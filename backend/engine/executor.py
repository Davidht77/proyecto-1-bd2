"""Planificador y ejecutor de consultas.

El planificador inspecciona la clausula WHERE y elige la ruta de acceso segun
las reglas del enunciado (3.4):

    igualdad + indice HASH o BTREE   -> IndexScan
    rango    + indice BTREE          -> IndexRangeScan
    igualdad sobre la PK (Sequential)-> BinarySearch
    rango    sobre la PK (Sequential)-> BinaryRangeScan
    cualquier otro caso              -> SeqScan (Full Table Scan)

Con varios predicados (Entregable 2) se elige el que admite la mejor ruta de
acceso y el resto se evalua como filtro residual sobre las filas recuperadas.

Consultas vectoriales (ORDER BY <col> <->|<=> '[...]' LIMIT k), enunciado 5.4:

    sin WHERE, con IVF               -> IVFScan (ANN, `SET ivf.probes`)
    sin WHERE, sin IVF               -> KnnSeqScan (k-NN exacto, Max-Heap)
    WHERE indexable, sin IVF         -> PreFilter (indice relacional + k-NN exacto)
    WHERE no indexable, con IVF      -> PostFilter (candidatos ANN + filtro)
    WHERE indexable y con IVF        -> PreFilter o PostFilter segun hybrid.choose_strategy

Cada ejecucion reporta el desglose exacto de bloques leidos y escritos (por
separado los del indice y los de la tabla) y los tiempos de parseo y ejecucion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.engine.catalog import Catalog
from backend.engine.hybrid import HybridEstimate, choose_strategy
from backend.engine.errors import (
    ColumnNotFound,
    NotImplementedFeature,
    SQLRuntimeError,
    TableNotFound,
)
from backend.engine.parser import (
    Condition,
    CreateIndex,
    CreateTable,
    Delete,
    DistanceExpr,
    DropTable,
    Insert,
    ScoreExpr,
    Select,
    SetOption,
    SpatialRadius,
    Statement,
    TextMatch,
    parse,
)
from backend.index.hash import RID as HashRID
from backend.storage.page import page_capacity
from backend.storage.schema import ColumnType, Schema, parse_vector
from backend.vector.ivf import IVFIndex, VRID
from backend.vector.knn import knn_linear


@dataclass
class Plan:
    access: str          # SeqScan | BinarySearch | BinaryRangeScan | IndexScan | ...
    table: str | None
    reason: str
    estimated_reads: int | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "access": self.access,
            "table": self.table,
            "reason": self.reason,
            "estimated_reads": self.estimated_reads,
            **self.detail,
        }


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    plan: Plan
    disk_reads: int
    disk_writes: int
    parse_ms: float
    exec_ms: float
    message: str | None = None
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "plan": self.plan.as_dict(),
            "metrics": {
                "disk_reads": self.disk_reads,
                "disk_writes": self.disk_writes,
                "parse_ms": round(self.parse_ms, 3),
                "exec_ms": round(self.exec_ms, 3),
                "total_ms": round(self.parse_ms + self.exec_ms, 3),
            },
            "message": self.message,
            "truncated": self.truncated,
        }


MAX_ROWS = 500  # el visor de resultados pagina; no tiene sentido enviar 500k filas

# Opciones de sesion que acepta SET. Cada una: (valor por defecto, validador).
DEFAULT_SETTINGS: dict[str, Any] = {
    "ivf.probes": 4,             # listas que escanea el IVF por consulta
    "hybrid.strategy": "AUTO",   # AUTO | PRE | POST (forzar para experimentos)
}

# Orden de preferencia entre rutas de acceso cuando hay varios predicados.
_ACCESS_RANK = {
    "IndexScan": 0, "BinarySearch": 1, "IndexRangeScan": 2, "BinaryRangeScan": 3, "SeqScan": 9,
}

# Tope del conteo de filas candidatas que hace el planificador hibrido: pasado
# este numero el pre-filtrado ya no puede ganar y no vale la pena seguir contando.
ESTIMATE_CAP = 2000


class Executor:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)

    # ------------------------------------------------------------- entrada

    def execute(self, sql: str, max_rows: int = MAX_ROWS) -> QueryResult:
        t0 = time.perf_counter()
        stmt = parse(sql)
        parse_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        result = self._dispatch(stmt, max_rows)
        result.exec_ms = (time.perf_counter() - t1) * 1000.0
        result.parse_ms = parse_ms
        return result

    def _dispatch(self, stmt: Statement, max_rows: int) -> QueryResult:
        if isinstance(stmt, CreateTable):
            return self._create_table(stmt)
        if isinstance(stmt, DropTable):
            return self._drop_table(stmt)
        if isinstance(stmt, Insert):
            return self._insert(stmt)
        if isinstance(stmt, Select):
            return self._select(stmt, max_rows)
        if isinstance(stmt, Delete):
            return self._delete(stmt)
        if isinstance(stmt, CreateIndex):
            return self._create_index(stmt)
        if isinstance(stmt, SetOption):
            return self._set(stmt)
        raise SQLRuntimeError(f"sentencia no soportada: {type(stmt).__name__}")

    # ------------------------------------------------------------- DDL

    def _create_table(self, stmt: CreateTable) -> QueryResult:
        schema = Schema(stmt.columns, pk_index=stmt.pk_index)
        self.catalog.create_table(stmt.table, schema, stmt.engine)
        info = self.catalog.describe(stmt.table)
        return _ok(
            Plan("DDL", stmt.table, f"CREATE TABLE USING {stmt.engine}"),
            message=(
                f"Tabla «{stmt.table}» creada con motor {stmt.engine.lower()}. "
                f"Registro de {schema.record_size} B por fila, "
                f"bloques de {info.page_size // 1024} KB."
            ),
            writes=1,
        )

    def _drop_table(self, stmt: DropTable) -> QueryResult:
        self.catalog.drop_table(stmt.table)
        return _ok(
            Plan("DDL", stmt.table, "DROP TABLE"),
            message=f"Tabla «{stmt.table}» eliminada.",
        )

    def _create_index(self, stmt: CreateIndex) -> QueryResult:
        if not self.catalog.exists(stmt.table):
            raise TableNotFound(f"la tabla «{stmt.table}» no existe")
        if stmt.kind not in ("HASH", "BTREE", "IVF"):
            dueno = {"RTREE": "el módulo espacial (R-Tree)",
                     "INVERTED": "el módulo textual (índice invertido)",
                     "HNSW": "una segunda estructura ANN (se implementó IVF)"}.get(stmt.kind, "")
            raise NotImplementedFeature(
                f"CREATE INDEX ... USING {stmt.kind} todavía no está conectado al motor"
                + (f": corresponde a {dueno}." if dueno else ".")
            )
        if stmt.options and stmt.kind != "IVF":
            raise SQLRuntimeError(f"USING {stmt.kind} no acepta opciones WITH (...)")
        _validar_columna(self.catalog.open_table(stmt.table).schema, stmt.column, stmt.table)
        self.catalog.create_index(stmt.table, stmt.name, stmt.column, stmt.kind, stmt.options)
        if stmt.kind == "IVF":
            st = self.catalog.open_index(stmt.table, stmt.name).stats()
            mensaje = (
                f"Índice «{stmt.name}» creado (IVF, {st['metric']}) sobre "
                f"{stmt.table}.{stmt.column}: {st['n_vectors']} vectores de dimensión "
                f"{st['dim']} en {st['n_lists']} celdas de Voronoi, {st['list_pages']} "
                f"páginas de listas ({st['records_per_page']} vectores por página)."
            )
        elif stmt.kind == "HASH":
            mensaje = f"Índice «{stmt.name}» creado sobre {stmt.table}({stmt.column}) usando HASH."
        else:
            tree = self.catalog.open_index(stmt.table, stmt.name)
            mensaje = (
                f"Índice «{stmt.name}» creado (BTREE) sobre {stmt.table}.{stmt.column}. "
                f"Altura {tree.height}, fan-out {tree.fanout}."
            )
        return _ok(
            Plan("DDL", stmt.table, f"CREATE INDEX USING {stmt.kind}"),
            message=mensaje,
            writes=1,
        )

    def _set(self, stmt: SetOption) -> QueryResult:
        if stmt.name not in DEFAULT_SETTINGS:
            raise SQLRuntimeError(
                f"opción desconocida: {stmt.name}. Disponibles: {', '.join(DEFAULT_SETTINGS)}"
            )
        valor = stmt.value
        if stmt.name == "ivf.probes":
            if not isinstance(valor, int) or valor < 1:
                raise SQLRuntimeError("ivf.probes debe ser un entero >= 1")
        elif stmt.name == "hybrid.strategy":
            valor = str(valor).upper()
            if valor not in ("AUTO", "PRE", "POST"):
                raise SQLRuntimeError("hybrid.strategy debe ser AUTO, PRE o POST")
        self.settings[stmt.name] = valor
        return _ok(Plan("SET", None, f"{stmt.name} = {valor}"), message=f"{stmt.name} = {valor}")

    # ------------------------------------------------------------- DML

    def _insert(self, stmt: Insert) -> QueryResult:
        sf = self.catalog.open_table(stmt.table)
        if len(stmt.values) != len(sf.schema.columns):
            raise SQLRuntimeError(
                f"la tabla «{stmt.table}» tiene {len(sf.schema.columns)} columnas, "
                f"se dieron {len(stmt.values)} valores"
            )
        # Los literales vectoriales llegan como string '[...]': se convierten una
        # sola vez para que la tabla y el IVF reciban el mismo valor numerico.
        valores = list(stmt.values)
        for i, col in enumerate(sf.schema.columns):
            if col.type is ColumnType.VECTOR:
                try:
                    valores[i] = parse_vector(valores[i], col.length)
                except (ValueError, TypeError) as exc:
                    raise SQLRuntimeError(f"columna «{col.name}»: {exc}") from None
        es_seq = self.catalog.engine_of(stmt.table) == "SEQUENTIAL"
        sf.counter.reset()
        paginas_antes = sf.page_count
        rid = sf.insert(tuple(valores))
        self._index_on_insert(stmt.table, sf.schema, valores, rid)
        sf.flush()

        mensaje = "1 registro insertado."
        if es_seq and sf.page_count != paginas_antes:
            mensaje += (
                f" Se disparó una reorganización automática: el área de overflow "
                f"superó su límite de {sf.overflow_cap} registros."
            )
        razon = (
            "Inserción en el área principal o, si la página está llena, en su "
            "cadena de overflow."
            if es_seq
            else "Inserción en la primera página con espacio libre, localizada por "
            "la free-list del archivo."
        )
        return _ok(
            Plan("Insert", stmt.table, razon),
            message=mensaje,
            reads=sf.counter.disk_reads,
            writes=sf.counter.disk_writes,
        )

    def _delete(self, stmt: Delete) -> QueryResult:
        # NOTA: esto no mantiene los indices BTREE de la tabla. HeapFile
        # borra con move-the-last (el ultimo registro del archivo hereda el
        # RID del borrado) y SequentialFile puede desplazar overflow, asi
        # que un borrado invalida RIDs de formas que un simple "quita esta
        # clave del arbol" no repara. Falta una pasada de reindexado
        # (o reconstruir los indices con Catalog.create_index) tras un
        # DELETE si la tabla tiene indices BTREE activos.
        sf = self.catalog.open_table(stmt.table)
        if stmt.where is None:
            raise SQLRuntimeError(
                "Un DELETE sin WHERE borraría la tabla entera. Usa DROP TABLE si es lo que quieres."
            )
        filtros = stmt.filters or [stmt.where]
        for f in filtros:
            _validar_columna(sf.schema, f.column, stmt.table)
        acceso, plan = self._choose_access(sf, stmt.table, filtros, "Delete")
        residuales = [f for f in filtros if f is not acceso]

        sf.counter.reset()
        if plan.access == "BinarySearch" and not residuales:
            claves = [acceso.lo] if sf.search(acceso.lo) is not None else []
        else:
            claves = [
                sf.schema.key_of(row)
                for row in self._rows_for(sf, acceso, plan, limit=None, table=stmt.table)
                if all(_matches(sf.schema, row, f) for f in residuales)
            ]

        indices_hash = self._hash_indexes_of(stmt.table)
        indices_ivf = self._ivf_indexes_of(stmt.table)
        es_heap = self.catalog.engine_of(stmt.table) == "HEAP"
        borradas = 0
        for k in claves:
            row = sf.search(k)
            rid = _rid_of(sf, k)
            if row is None or rid is None:
                continue
            if sf.delete(k):
                borradas += 1
                hash_rid = HashRID.from_engine_rid(rid)
                for info, col_idx in indices_hash:
                    idx = self.catalog.open_index(stmt.table, info["name"])
                    idx.delete(row[col_idx], rid=hash_rid)
                    idx.flush()
                for idx, col_idx in indices_ivf:
                    idx.delete(k, row[col_idx])
                if es_heap and indices_ivf:
                    self._ivf_follow_move_the_last(sf, rid, k, indices_ivf)
                for idx, _ in indices_ivf:
                    idx.flush()
        sf.flush()

        return _ok(
            plan,
            message=f"{borradas} registro{'' if borradas == 1 else 's'} eliminado{'' if borradas == 1 else 's'}.",
            reads=sf.counter.disk_reads,
            writes=sf.counter.disk_writes,
        )

    def _hash_indexes_of(self, table: str) -> list[tuple[dict, int]]:
        sf = self.catalog.open_table(table)
        out = []
        for info in self.catalog.indexes_of(table):
            if info["kind"] != "HASH":
                continue
            col_idx = next(i for i, c in enumerate(sf.schema.columns) if c.name.lower() == info["column"].lower())
            out.append((info, col_idx))
        return out

    def _ivf_indexes_of(self, table: str) -> list[tuple[IVFIndex, int]]:
        sf = self.catalog.open_table(table)
        out = []
        for info in self.catalog.indexes_of(table):
            if info["kind"] != "IVF":
                continue
            col_idx = next(i for i, c in enumerate(sf.schema.columns)
                           if c.name.lower() == info["column"].lower())
            out.append((self.catalog.open_index(table, info["name"]), col_idx))
        return out

    def _ivf_follow_move_the_last(self, sf, rid, borrada, indices_ivf) -> None:
        """El Heap File borra con move-the-last: la fila que estaba al final del
        archivo ahora ocupa el RID de la borrada. Se actualiza su entrada en el
        IVF para que el indice no quede apuntando a un RID viejo."""
        page = sf.pager.read_page(rid.page_id)
        if not page.is_occupied(rid.slot):
            return
        movida = sf.schema.unpack(page.read_record(rid.slot))
        pk = sf.schema.key_of(movida)
        if pk == borrada:
            return
        for idx, col_idx in indices_ivf:
            idx.relocate(pk, movida[col_idx], rid)

    def _index_on_insert(self, table: str, schema: Schema, values: list, rid) -> None:
        hash_rid = HashRID.from_engine_rid(rid)
        for info, col_idx in self._hash_indexes_of(table):
            idx = self.catalog.open_index(table, info["name"])
            idx.insert(values[col_idx], hash_rid)
            idx.flush()
        for idx, col_idx in self._ivf_indexes_of(table):
            idx.insert(values[schema.pk_index], rid, values[col_idx])
            idx.flush()

    # ------------------------------------------------------------- SELECT

    def _select(self, stmt: Select, max_rows: int) -> QueryResult:
        sf = self.catalog.open_table(stmt.table)
        schema = sf.schema

        if stmt.special:
            pred = stmt.special[0]
            modulo = ("el módulo textual (índice invertido + COSINE_SCORE)"
                      if isinstance(pred, TextMatch) else "el módulo espacial (R-Tree)")
            raise NotImplementedFeature(
                f"El predicado {'MATCH' if isinstance(pred, TextMatch) else 'WITHIN_RADIUS'} "
                f"ya se parsea, pero su ejecución corresponde a {modulo}, que aún no está "
                f"conectado al planificador."
            )
        if any(isinstance(c, ScoreExpr) for c in stmt.columns):
            raise SQLRuntimeError("COSINE_SCORE()/BM25_SCORE() solo tienen sentido junto a un predicado MATCH")
        for f in stmt.filters:
            _validar_columna(schema, f.column, stmt.table)

        if stmt.distance is not None:
            return self._select_vector(stmt, sf, max_rows)

        proyeccion, columnas = _projection(schema, stmt.columns, stmt.table)
        orden = self._order_column(schema, stmt)
        acceso, plan = self._choose_access(sf, stmt.table, stmt.filters, "Select")
        residuales = [f for f in stmt.filters if f is not acceso]
        if residuales:
            plan.detail["residual_filters"] = [f.column for f in residuales]

        # El LIMIT solo puede cortar el recorrido si nada lo reordena o descarta despues.
        corte = stmt.limit if (orden is None and not residuales) else None
        io = self._io_snapshot(sf, stmt.table)
        filas_todas = (
            row for row in self._rows_for(sf, acceso, plan, corte, table=stmt.table)
            if all(_matches(schema, row, f) for f in residuales)
        )
        if orden is not None:
            col_idx, desc = orden
            filas_todas = iter(sorted(filas_todas, key=lambda r: r[col_idx], reverse=desc))
            plan.detail["sort"] = f"{schema.columns[col_idx].name} {'DESC' if desc else 'ASC'} (en memoria)"
        filas: list[list[Any]] = []
        total = 0
        for row in filas_todas:
            if stmt.limit is not None and total >= stmt.limit:
                break
            total += 1
            if len(filas) < max_rows:
                filas.append([row[i] for i in proyeccion])
        reads, writes = self._io_finish(io, plan)

        return QueryResult(
            columns=columnas,
            rows=filas,
            row_count=total,
            plan=plan,
            disk_reads=reads,
            disk_writes=writes,
            parse_ms=0.0,
            exec_ms=0.0,
            truncated=total > len(filas),
        )

    def _order_column(self, schema: Schema, stmt: Select) -> tuple[int, bool] | None:
        if stmt.order_by is None:
            return None
        _validar_columna(schema, stmt.order_by.key, stmt.table)
        idx = next(i for i, c in enumerate(schema.columns) if c.name.lower() == stmt.order_by.key.lower())
        if schema.columns[idx].type is ColumnType.VECTOR:
            raise SQLRuntimeError("no se puede ordenar por un VECTOR; usa una distancia (<-> o <=>)")
        return idx, stmt.order_by.descending

    def _choose_access(self, sf, table: str, filters: list[Condition], kind: str):
        """Entre varios predicados, el que admite la ruta de acceso mas barata."""
        if not filters:
            return None, self._plan(sf, table, None, kind)
        candidatos = [(f, self._plan(sf, table, f, kind)) for f in filters]
        return min(candidatos, key=lambda fp: _ACCESS_RANK.get(fp[1].access, 5))

    # ------------------------------------------------ telemetria indice/tabla

    def _index_counters(self, table: str) -> list:
        counters = []
        for info in self.catalog.indexes_of(table):
            if info.get("implicit"):
                continue
            idx = self.catalog.open_index(table, info["name"])
            if hasattr(idx, "counter"):
                counters.append(idx.counter)
            else:  # ExtendibleHash: un contador por archivo
                counters.extend([idx.dir_pager.counter, idx.bkt_pager.counter])
        return counters

    def _io_snapshot(self, sf, table: str) -> dict:
        counters = self._index_counters(table)
        sf.counter.reset()
        return {"sf": sf, "counters": counters,
                "r0": sum(c.disk_reads for c in counters), "w0": sum(c.disk_writes for c in counters)}

    def _io_finish(self, io: dict, plan: Plan) -> tuple[int, int]:
        sf, counters = io["sf"], io["counters"]
        idx_r = sum(c.disk_reads for c in counters) - io["r0"]
        idx_w = sum(c.disk_writes for c in counters) - io["w0"]
        plan.detail["io"] = {
            "index_reads": idx_r, "table_reads": sf.counter.disk_reads,
            "index_writes": idx_w, "table_writes": sf.counter.disk_writes,
        }
        return sf.counter.disk_reads + idx_r, sf.counter.disk_writes + idx_w

    # ------------------------------------------------- consultas vectoriales

    def _select_vector(self, stmt: Select, sf, max_rows: int) -> QueryResult:
        schema = sf.schema
        dist = stmt.distance
        _validar_columna(schema, dist.column, stmt.table)
        vcol = next(i for i, c in enumerate(schema.columns) if c.name.lower() == dist.column.lower())
        col = schema.columns[vcol]
        if col.type is not ColumnType.VECTOR:
            raise SQLRuntimeError(f"«{col.name}» no es una columna VECTOR: no admite {dist.operator}")
        try:
            q = np.asarray(parse_vector(dist.vector, col.length), dtype=np.float32)
        except ValueError as exc:
            raise SQLRuntimeError(f"vector de consulta: {exc}") from None

        # El ORDER BY puede nombrar la distancia por su alias o repetir la expresion.
        ok = stmt.order_by
        por_distancia = ok is not None and (
            isinstance(ok.key, DistanceExpr)
            or ok.key.lower() in {(dist.alias or "").lower(), dist.label.lower()}
        )
        if ok is not None and not por_distancia:
            raise SQLRuntimeError(
                "en una consulta vectorial el ORDER BY debe ser la distancia (o su alias)"
            )
        es_knn = por_distancia and not ok.descending and stmt.limit is not None

        items, columnas = _vector_projection(schema, stmt.columns, dist, stmt.table)
        ivf = self._ivf_for(stmt.table, col.name, dist.metric)
        io = self._io_snapshot(sf, stmt.table)

        if es_knn:
            plan, resultados = self._knn(stmt, sf, vcol, q, dist, ivf)
        else:
            plan, resultados = self._vector_full(stmt, sf, vcol, q, dist, por_distancia)

        filas = []
        for d, row in resultados[:max_rows]:
            filas.append([round(d, 6) if it == "__dist__" else _jsonable(row[it]) for it in items])
        reads, writes = self._io_finish(io, plan)
        return QueryResult(
            columns=columnas, rows=filas, row_count=len(resultados), plan=plan,
            disk_reads=reads, disk_writes=writes, parse_ms=0.0, exec_ms=0.0,
            truncated=len(resultados) > len(filas),
        )

    def _ivf_for(self, table: str, column: str, metric: str) -> tuple[dict, IVFIndex] | None:
        for info in self.catalog.indexes_of(table):
            if info["kind"] == "IVF" and info["column"].lower() == column.lower() and info["metric"] == metric:
                return info, self.catalog.open_index(table, info["name"])
        return None

    def _vector_full(self, stmt, sf, vcol, q, dist, ordenar):
        """Sin LIMIT (o con DESC) no hay k-NN: se calcula la distancia de todas las
        filas que cumplen el WHERE y, si se pidio, se ordena."""
        acceso, plan = self._choose_access(sf, stmt.table, stmt.filters, "Select")
        residuales = [f for f in stmt.filters if f is not acceso]
        filas = [r for r in self._rows_for(sf, acceso, plan, None, table=stmt.table)
                 if all(_matches(sf.schema, r, f) for f in residuales)]
        from backend.vector.distance import distances
        d = distances(q, np.asarray([r[vcol] for r in filas], dtype=np.float32), dist.metric) if filas else []
        res = list(zip([float(x) for x in d], filas))
        if ordenar:
            res.sort(key=lambda e: e[0], reverse=stmt.order_by.descending)
        if stmt.limit is not None:
            res = res[: stmt.limit]
        plan.detail["vector"] = {"metric": dist.metric, "mode": "distancia exacta por fila"}
        return plan, res

    def _knn(self, stmt, sf, vcol, q, dist, ivf):
        table, k, filtros = stmt.table, stmt.limit, stmt.filters
        P = max(1, sf.page_count)
        base = {"pages": P, "engine": self.catalog.engine_of(table),
                "vector": {"metric": dist.metric, "k": k, "column": sf.schema.columns[vcol].name}}
        acceso, pre_plan = self._choose_access(sf, table, filtros, "Select")
        indexable = acceso is not None and pre_plan.access != "SeqScan"

        if not filtros:
            estrategia = "IVF" if ivf else "SEQ"
        elif ivf is None:
            estrategia = "PRE" if indexable else "SEQ"
        elif not indexable:
            estrategia = "POST"
        else:
            forzada = self.settings["hybrid.strategy"]
            if forzada in ("PRE", "POST"):
                estrategia, razon = forzada, f"forzada con SET hybrid.strategy = {forzada}"
            else:
                est = self._estimate(sf, table, acceso, pre_plan, k, ivf[1])
                estrategia, razon = choose_strategy(est)
                base["estimate"] = est.as_dict()
            base["decision"] = razon

        if estrategia in ("SEQ", "PRE"):
            residuales = [f for f in filtros if f is not acceso] if estrategia == "PRE" else filtros
            fuente = self._rows_for(sf, acceso if estrategia == "PRE" else None,
                                    pre_plan if estrategia == "PRE" else self._plan(sf, table, None, "Select"),
                                    None, table=table)
            candidatas = (r for r in fuente if all(_matches(sf.schema, r, f) for f in residuales))
            res = knn_linear(candidatas, lambda r: r[vcol], q, dist.metric, k)
            if estrategia == "SEQ":
                plan = Plan("KnnSeqScan", table,
                            f"k-NN exacto: recorre las {P} páginas, calcula la distancia "
                            f"{dist.metric} de cada fila{' que cumple el WHERE' if filtros else ''} "
                            f"y conserva las {k} mejores en un Max-Heap.",
                            estimated_reads=P, detail=base)
            else:
                plan = Plan("PreFilter", table,
                            f"Pre-filtering: {pre_plan.access} sobre «{acceso.column}» entrega solo "
                            f"las filas que cumplen el filtro y sobre ellas se hace k-NN exacto "
                            f"(Max-Heap de {k}).",
                            detail={**base, "prefilter_access": pre_plan.access,
                                    **{k_: v for k_, v in pre_plan.detail.items() if k_ in ("index_name", "kind")}})
            return plan, [(d, r) for d, r in res]

        info, idx = ivf
        return self._ivf_scan(stmt, sf, vcol, q, idx, info, base, estrategia == "POST")

    def _ivf_scan(self, stmt, sf, vcol, q, idx: IVFIndex, info, base, post: bool):
        table, k = stmt.table, stmt.limit
        probes = max(1, min(self.settings["ivf.probes"], idx.n_lists))
        orden = idx.probe_order(q)
        escaneadas, ampliaciones, revisados, obsoletos = 0, 0, 0, 0
        aceptadas: list[tuple[float, tuple]] = []
        lote = probes
        while escaneadas < len(orden):
            cands = idx.scan_lists(q, orden[escaneadas : escaneadas + lote])
            escaneadas += lote
            for c in cands:
                # Los candidatos llegan ordenados por distancia: los primeros k que
                # pasan el filtro son el top-k de las celdas escaneadas.
                if len(aceptadas) >= k:
                    break
                row, stale = self._fetch_candidate(sf, table, c)
                revisados += 1
                obsoletos += stale
                if row is None:
                    continue
                if post and not all(_matches(sf.schema, row, f) for f in stmt.filters):
                    continue
                aceptadas.append((c.distance, row))
            if len(aceptadas) >= k or not post:
                break
            # Post-filtering con filtro muy selectivo: no alcanzaron las k filas.
            # Se duplica el sondeo sobre las celdas siguientes (iterative deepening).
            ampliaciones += 1
            lote *= 2
        aceptadas.sort(key=lambda e: e[0])
        detalle = {**base, "index_name": info["name"], "n_lists": idx.n_lists,
                   "probes": probes, "lists_scanned": min(escaneadas, len(orden)),
                   "candidates_checked": revisados}
        if ampliaciones:
            detalle["expansions"] = ampliaciones
        if obsoletos:
            detalle["stale_rids"] = obsoletos
        if post:
            plan = Plan("PostFilter", table,
                        f"Post-filtering: el IVF «{info['name']}» propone candidatos en orden de "
                        f"distancia ({probes} de {idx.n_lists} celdas) y a cada uno se le aplica el "
                        f"WHERE hasta reunir {k}"
                        + (f"; hubo que ampliar el sondeo {ampliaciones} vez/veces." if ampliaciones else "."),
                        detail=detalle)
        else:
            plan = Plan("IVFScan", table,
                        f"Búsqueda aproximada IVF «{info['name']}»: escanea solo las {probes} celdas "
                        f"de Voronoi más cercanas de {idx.n_lists} (ivf.probes = {probes}).",
                        estimated_reads=round(probes * idx.stats()["avg_pages_per_list"]) + k,
                        detail=detalle)
        return plan, aceptadas[:k]

    def _fetch_candidate(self, sf, table: str, c) -> tuple[tuple | None, int]:
        """Lee la fila de un candidato IVF verificando que el RID siga siendo suyo.

        Devuelve (fila, 1 si el RID estaba obsoleto). Un RID queda obsoleto si la
        tabla movio el registro (reorganizacion del Sequential, desplazamiento de
        slots): en ese caso se recupera por clave primaria."""
        try:
            row = self.catalog.row_at(table, HashRID(c.rid.page_id, c.rid.slot, c.rid.area))
        except (IndexError, ValueError):
            row = None
        if row is not None and sf.schema.key_of(row) == c.pk:
            return row, 0
        return sf.search(c.pk), 1

    def _estimate(self, sf, table: str, cond: Condition, plan: Plan, k: int, ivf: IVFIndex) -> HybridEstimate:
        """Cuenta las filas que cumplen `cond` recorriendo solo el indice
        relacional, con un tope: el planificador paga unas pocas paginas de
        indice para decidir con datos y no con una selectividad inventada."""
        cap = max(ESTIMATE_CAP, 10 * k)
        n = 0
        detalle = plan.detail
        if plan.access in ("IndexScan", "IndexRangeScan") and detalle.get("kind") == "HASH":
            n = len(self.catalog.open_index(table, detalle["index_name"]).search(cond.lo))
        elif plan.access in ("IndexScan", "IndexRangeScan"):
            tree = self.catalog.open_index(table, detalle["index_name"])
            lo = cond.lo if cond.lo is not None else _MIN
            hi = cond.hi if cond.hi is not None else _MAX
            for _ in tree.range_search(lo, hi):
                n += 1
                if n >= cap:
                    break
        else:  # BinarySearch / BinaryRangeScan sobre la clave del Sequential
            for _ in self._rows_for(sf, cond, plan, cap, table=table):
                n += 1
        st = ivf.stats()
        return HybridEstimate(
            est_matches=min(n, cap), capped=n >= cap, n_rows=sf.n_records, k=k,
            probes=max(1, min(self.settings["ivf.probes"], ivf.n_lists)),
            n_lists=ivf.n_lists, avg_pages_per_list=st["avg_pages_per_list"],
            rows_per_page=page_capacity(sf.page_size, sf.schema.record_size),
        )

    def _row_at(self, sf, rid):
        """Lee una fila de la tabla por su RID fisico (page_id, slot).

        El RID que guarda el indice BTREE es el mismo (page_id, slot) que
        usa la tabla, asi que basta con leer esa pagina directo por el
        Pager de la tabla, sin pasar por scan()/logical numbering. El BTREE
        solo indexa el area principal (ver Catalog.create_index), asi que
        aqui alcanza con el pager del Heap File o el area main del
        Sequential File.
        """
        page = sf.pager.read_page(rid.page_id) if hasattr(sf, "pager") else sf.main.read_page(rid.page_id)
        if not page.is_occupied(rid.slot):
            return None
        return sf.schema.unpack(page.read_record(rid.slot))

    # ------------------------------------------------- planificador de acceso

    def _plan(self, sf, table: str, where: Condition | None, kind: str) -> Plan:
        """Selecciona la ruta de acceso segun el enunciado 3.4."""
        P = max(1, sf.page_count)
        engine = self.catalog.engine_of(table)
        indices = self.catalog.indexes_of(table)
        es_pk = where is not None and where.column.lower() == sf.schema.pk_column.name.lower()
        base = {"pages": P, "engine": engine}

        if where is None:
            return Plan(
                "SeqScan", table,
                "Sin WHERE: hay que recorrer la tabla completa.",
                estimated_reads=P,
                detail=base,
            )

        hash_idx = next((i for i in indices if i["kind"] == "HASH" and i["column"].lower() == where.column.lower()), None)
        btree_idx = next((i for i in indices if i["kind"] == "BTREE" and i["column"].lower() == where.column.lower()), None)

        if where.is_equality and hash_idx:
            return Plan(
                "IndexScan", table, f"igualdad con indice HASH '{hash_idx['name']}'", 1,
                detail={**base, "kind": "HASH", "index_name": hash_idx["name"]},
            )
        if btree_idx:
            acceso = "IndexScan" if where.is_equality else "IndexRangeScan"
            tree = self.catalog.open_index(table, btree_idx["name"])
            razon = (
                f"igualdad con indice BTREE '{btree_idx['name']}': {tree.height} "
                f"transferencias (altura del arbol)"
                if where.is_equality else
                f"rango con indice BTREE '{btree_idx['name']}': desciende a la "
                f"primera hoja ({tree.height} transferencias) y recorre next_leaf_id"
            )
            return Plan(
                acceso, table, razon,
                estimated_reads=tree.height if where.is_equality else None,
                detail={**base, "index_name": btree_idx["name"], "index_height": tree.height, "fanout": tree.fanout},
            )

        if engine == "SEQUENTIAL" and es_pk:
            import math

            h = math.ceil(math.log2(max(2, P)))
            detalle = {**base, "overflow_records": sf.n_overflow}
            if where.is_equality:
                return Plan(
                    "BinarySearch", table,
                    f"Igualdad sobre la clave primaria de un Sequential File: "
                    f"búsqueda binaria sobre {P} páginas.",
                    estimated_reads=h,
                    detail=detalle,
                )
            return Plan(
                "BinaryRangeScan", table,
                f"Rango sobre la clave primaria: búsqueda binaria hasta el límite "
                f"inferior y avance secuencial por las {P} páginas del archivo.",
                estimated_reads=h,
                detail=detalle,
            )

        if engine == "HEAP":
            return Plan(
                "SeqScan", table,
                f"Un Heap File no mantiene orden físico, así que ni siquiera una "
                f"igualdad sobre la clave primaria puede evitar el recorrido de "
                f"las {P} páginas. Es el costo que un índice vendría a eliminar.",
                estimated_reads=P,
                detail=base,
            )

        return Plan(
            "SeqScan", table,
            f"El WHERE filtra por «{where.column}», que no es la clave primaria "
            f"ni tiene índice: recorrido secuencial completo.",
            estimated_reads=P,
            detail=base,
        )

    # ------------------------------------------------------------- ejecucion

    def _rows_for(self, sf, where: Condition | None, plan: Plan, limit: int | None, table: str | None = None):
        if plan.access == "IndexScan" and plan.detail.get("kind") == "HASH":
            idx = self.catalog.open_index(table, plan.detail["index_name"])
            n = 0
            for rid in idx.search(where.lo):
                row = self.catalog.row_at(table, rid)
                if row is None:
                    continue
                yield row
                n += 1
                if limit is not None and n >= limit:
                    return
            return

        if plan.access in ("IndexScan", "IndexRangeScan"):
            table = plan.table
            index_name = plan.detail.get("index_name")
            if index_name is None:
                raise NotImplementedFeature(
                    "El planificador eligió un IndexScan pero no hay un índice BTREE "
                    "aplicable (falta el hashing dinámico para índices HASH)."
                )
            tree = self.catalog.open_index(table, index_name)
            n = 0
            if plan.access == "IndexScan":
                rid = tree.search(where.lo)
                if rid is not None:
                    row = self._row_at(sf, rid)
                    if row is not None:
                        yield row
                return
            # IndexRangeScan
            lo = where.lo if where.lo is not None else _MIN
            hi = where.hi if where.hi is not None else _MAX
            for rid in tree.range_search(lo, hi):
                row = self._row_at(sf, rid)
                if row is None:
                    continue
                col_idx = next(
                    i for i, c in enumerate(sf.schema.columns) if c.name.lower() == where.column.lower()
                )
                v = row[col_idx]
                if not where.lo_inclusive and v == where.lo:
                    continue
                if not where.hi_inclusive and v == where.hi:
                    continue
                yield row
                n += 1
                if limit is not None and n >= limit:
                    return
            return

        n = 0
        if plan.access == "BinarySearch":
            row = sf.search(where.lo)
            if row is not None:
                yield row
            return

        if plan.access == "BinaryRangeScan":
            lo = where.lo if where.lo is not None else _MIN
            hi = where.hi if where.hi is not None else _MAX
            for row in sf.range_search(lo, hi):
                k = sf.schema.key_of(row)
                if not where.lo_inclusive and k == where.lo:
                    continue
                if not where.hi_inclusive and k == where.hi:
                    continue
                yield row
                n += 1
                if limit is not None and n >= limit:
                    return
            return

        # SeqScan
        for row in sf.scan():
            if where is not None and not _matches(sf.schema, row, where):
                continue
            yield row
            n += 1
            if limit is not None and n >= limit:
                return


_MIN = -(2**63)
_MAX = 2**63 - 1


def _matches(schema: Schema, row: tuple, where: Condition) -> bool:
    idx = next(
        (i for i, c in enumerate(schema.columns) if c.name.lower() == where.column.lower()),
        None,
    )
    if idx is None:
        return False
    v = row[idx]
    try:
        if where.lo is not None:
            if where.lo_inclusive and v < where.lo:
                return False
            if not where.lo_inclusive and v <= where.lo:
                return False
        if where.hi is not None:
            if where.hi_inclusive and v > where.hi:
                return False
            if not where.hi_inclusive and v >= where.hi:
                return False
    except TypeError:
        # Comparar CHAR con un numero, por ejemplo: el predicado no aplica.
        return False
    return True


def _projection(schema: Schema, columns: list, tabla: str) -> tuple[list[int], list[str]]:
    nombres = [c.name for c in schema.columns]
    if not columns:
        return list(range(len(nombres))), nombres
    proyeccion = []
    for col in columns:
        _validar_columna(schema, col, tabla)
        proyeccion.append(next(i for i, n in enumerate(nombres) if n.lower() == col.lower()))
    return proyeccion, [nombres[i] for i in proyeccion]


def _vector_projection(schema: Schema, columns: list, dist: DistanceExpr, tabla: str):
    """Como _projection, pero la expresion de distancia se marca con '__dist__'."""
    nombres = [c.name for c in schema.columns]
    if not columns:
        return list(range(len(nombres))) + ["__dist__"], nombres + [dist.label]
    items, etiquetas = [], []
    for col in columns:
        if isinstance(col, DistanceExpr):
            items.append("__dist__")
            etiquetas.append(col.label)
            continue
        _validar_columna(schema, col, tabla)
        i = next(i for i, n in enumerate(nombres) if n.lower() == col.lower())
        items.append(i)
        etiquetas.append(nombres[i])
    return items, etiquetas


def _jsonable(value):
    return list(value) if isinstance(value, tuple) else value


def _rid_of(sf, key):
    hallazgo = sf._find_rid(key)
    if hallazgo is None:
        return None
    return hallazgo if hasattr(hallazgo, "_fields") else hallazgo[0]


def _validar_columna(schema: Schema, nombre: str, tabla: str) -> None:
    if not any(c.name.lower() == nombre.lower() for c in schema.columns):
        disponibles = ", ".join(c.name for c in schema.columns)
        raise ColumnNotFound(
            f"la columna «{nombre}» no existe en «{tabla}». Columnas disponibles: {disponibles}"
        )


def _ok(plan: Plan, message: str, reads: int = 0, writes: int = 0) -> QueryResult:
    return QueryResult(
        columns=[], rows=[], row_count=0, plan=plan,
        disk_reads=reads, disk_writes=writes,
        parse_ms=0.0, exec_ms=0.0, message=message,
    )