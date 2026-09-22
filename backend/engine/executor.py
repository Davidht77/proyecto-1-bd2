"""Planificador y ejecutor de consultas.

El planificador inspecciona la clausula WHERE y elige la ruta de acceso segun
las reglas del enunciado (3.4):

    igualdad + indice HASH o BTREE   -> IndexScan
    rango    + indice BTREE          -> IndexRangeScan
    igualdad sobre la PK (Sequential)-> BinarySearch
    rango    sobre la PK (Sequential)-> BinaryRangeScan
    cualquier otro caso              -> SeqScan (Full Table Scan)

Cada ejecucion reporta el desglose exacto de bloques leidos y escritos, y los
tiempos de parseo y ejecucion en milisegundos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from backend.engine.catalog import Catalog
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
    DropTable,
    Insert,
    Select,
    Statement,
    parse,
)
from backend.index.hash import RID as HashRID
from backend.storage.schema import Schema


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


class Executor:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

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
        if stmt.kind != "HASH":
            raise NotImplementedFeature(
                f"CREATE INDEX ... USING {stmt.kind} todavía no está implementado. "
                "El árbol B+ es trabajo pendiente del equipo. Mientras tanto, las "
                "consultas de igualdad y de rango sobre la clave primaria de una "
                "tabla SEQUENTIAL ya resuelven por búsqueda binaria, sin índice."
            )
        _validar_columna(self.catalog.open_table(stmt.table).schema, stmt.column, stmt.table)
        self.catalog.create_index(stmt.table, stmt.name, stmt.column, stmt.kind)
        return _ok(
            Plan("DDL", stmt.table, f"CREATE INDEX USING {stmt.kind}"),
            message=f"Índice «{stmt.name}» creado sobre {stmt.table}({stmt.column}) usando HASH.",
            writes=1,
        )

    # ------------------------------------------------------------- DML

    def _insert(self, stmt: Insert) -> QueryResult:
        sf = self.catalog.open_table(stmt.table)
        if len(stmt.values) != len(sf.schema.columns):
            raise SQLRuntimeError(
                f"la tabla «{stmt.table}» tiene {len(sf.schema.columns)} columnas, "
                f"se dieron {len(stmt.values)} valores"
            )
        es_seq = self.catalog.engine_of(stmt.table) == "SEQUENTIAL"
        sf.counter.reset()
        paginas_antes = sf.page_count
        rid = sf.insert(tuple(stmt.values))
        self._index_on_insert(stmt.table, sf.schema, stmt.values, rid)
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
        sf = self.catalog.open_table(stmt.table)
        if stmt.where is None:
            raise SQLRuntimeError(
                "Un DELETE sin WHERE borraría la tabla entera. Usa DROP TABLE si es lo que quieres."
            )
        _validar_columna(sf.schema, stmt.where.column, stmt.table)
        plan = self._plan(sf, stmt.table, stmt.where, "Delete")

        sf.counter.reset()
        if plan.access == "BinarySearch":
            claves = [stmt.where.lo] if sf.search(stmt.where.lo) is not None else []
        else:
            claves = [
                sf.schema.key_of(row)
                for row in self._rows_for(sf, stmt.where, plan, limit=None, table=stmt.table)
            ]

        indices_hash = self._hash_indexes_of(stmt.table)
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

    def _index_on_insert(self, table: str, schema: Schema, values: list, rid) -> None:
        hash_rid = HashRID.from_engine_rid(rid)
        for info, col_idx in self._hash_indexes_of(table):
            idx = self.catalog.open_index(table, info["name"])
            idx.insert(values[col_idx], hash_rid)
            idx.flush()

    # ------------------------------------------------------------- SELECT

    def _select(self, stmt: Select, max_rows: int) -> QueryResult:
        sf = self.catalog.open_table(stmt.table)
        schema = sf.schema

        nombres = [c.name for c in schema.columns]
        if stmt.columns:
            for col in stmt.columns:
                _validar_columna(schema, col, stmt.table)
            proyeccion = [
                next(i for i, n in enumerate(nombres) if n.lower() == c.lower())
                for c in stmt.columns
            ]
            columnas = [nombres[i] for i in proyeccion]
        else:
            proyeccion = list(range(len(nombres)))
            columnas = nombres

        if stmt.where is not None:
            _validar_columna(schema, stmt.where.column, stmt.table)

        plan = self._plan(sf, stmt.table, stmt.where, "Select")

        limite = stmt.limit if stmt.limit is not None else None
        sf.counter.reset()
        filas: list[list[Any]] = []
        total = 0
        for row in self._rows_for(sf, stmt.where, plan, limite, table=stmt.table):
            total += 1
            if len(filas) < max_rows:
                filas.append([row[i] for i in proyeccion])
        reads, writes = sf.counter.disk_reads, sf.counter.disk_writes

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

    # ------------------------------------------------- planificador de acceso

    def _plan(self, sf, table: str, where: Condition | None, kind: str) -> Plan:
        """Selecciona la ruta de acceso.

        El orden de preferencia sigue el enunciado 3.4. Como todavia no hay
        indices BTREE ni HASH, las dos primeras ramas no se alcanzan; quedan
        escritas para que quien los implemente solo tenga que conectarlos aqui.
        """
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
            return Plan(acceso, table, f"indice BTREE '{btree_idx['name']}'", None)

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
            raise NotImplementedFeature(
                "El planificador eligió un IndexScan, pero el árbol B+ todavía no "
                "está implementado."
            )

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
