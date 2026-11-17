"""Catalogo de tablas.

No hay archivo de catalogo. El esquema de cada tabla ya vive en la pagina 0 de
su propio archivo de datos, asi que el catalogo se reconstruye escaneando el
directorio y leyendo esas cabeceras. Eso elimina la posibilidad de que el
catalogo y los datos se desincronicen, y de paso evita introducir un
serializador de alto nivel, que el enunciado prohibe.

El motor de almacenamiento se codifica en el sufijo del archivo:
    <tabla>.seq.dat   Sequential File
    <tabla>.heap.dat  Heap File

Los indices secundarios siguen la misma logica autodescriptiva: un BTREE de
la tabla <t> con nombre <n> vive en <t>__<n>.btree.dat, y ese archivo ya
guarda su propio esquema de clave (ver BPlusTree), asi que el catalogo no
necesita un archivo de metadatos aparte para reconstruirlos: alcanza con
listar los que hagan match con <t>__*.btree.dat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.engine.errors import (
    ColumnNotFound,
    NotImplementedFeature,
    SQLRuntimeError,
    TableAlreadyExists,
    TableNotFound,
)
from backend.index.btree import BPlusTree
from backend.index.btree import RID as BTreeRID
from backend.index.hash import ExtendibleHash
from backend.index.hash import RID as HashRID
from backend.index.rtree import RTree, RTreeRID
from backend.storage.pager import peek_header
from backend.storage.schema import Column, ColumnType, Schema
from backend.structures.heap_file import HeapFile
from backend.structures.sequential_file import SequentialFile
from backend.vector.distance import METRICS
from backend.vector.ivf import IVFIndex

# Ambas organizaciones exponen la misma superficie: insert, search, delete,
# scan, range_search, bulk_load, flush y close. El motor las trata por igual;
# lo unico exclusivo del Sequential File es reorganize() y su overflow.
ENGINE_CLASS = {"SEQUENTIAL": SequentialFile, "HEAP": HeapFile}

ENGINE_SUFFIX = {"SEQUENTIAL": ".seq", "HEAP": ".heap"}
SUFFIX_ENGINE = {v: k for k, v in ENGINE_SUFFIX.items()}

INDEX_SUFFIX = "__"  # <tabla>__<indice>.btree.dat


@dataclass
class TableInfo:
    name: str
    engine: str
    schema: Schema
    n_records: int
    n_overflow: int
    page_count: int
    page_size: int
    record_size: int
    size_bytes: int
    indexes: list[dict]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "engine": self.engine,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type.name,
                    "length": c.length,
                    "primary_key": i == self.schema.pk_index,
                }
                for i, c in enumerate(self.schema.columns)
            ],
            "n_records": self.n_records,
            "n_overflow": self.n_overflow,
            "page_count": self.page_count,
            "page_size": self.page_size,
            "record_size": self.record_size,
            "size_bytes": self.size_bytes,
            "indexes": self.indexes,
        }


class Catalog:
    def __init__(self, data_dir: str = "data", page_size: int = 4096, cache_size: int = 0):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self.cache_size = cache_size
        self._open: dict[str, SequentialFile] = {}
        self._open_indexes: dict[
            tuple[str, str], ExtendibleHash | BPlusTree | IVFIndex | RTree
        ] = {}

    # ------------------------------------------------------------- rutas

    def _base(self, table: str) -> str:
        return str(self.data_dir / table)

    def _index_base(self, table: str, index_name: str) -> str:
        return str(self.data_dir / f"{table}{INDEX_SUFFIX}{index_name}")

    def _engine_of(self, table: str) -> str | None:
        for suffix, engine in SUFFIX_ENGINE.items():
            if (self.data_dir / f"{table}{suffix}.dat").exists():
                return engine
        return None

    def exists(self, table: str) -> bool:
        return self._engine_of(table) is not None

    # ------------------------------------------------------------- DDL

    def create_table(self, table: str, schema: Schema, engine: str) -> None:
        if self.exists(table):
            raise TableAlreadyExists(f"la tabla «{table}» ya existe")
        cls = ENGINE_CLASS[engine]
        f = cls(self._base(table), schema,
                page_size=self.page_size, cache_size=self.cache_size)
        f.close()

    def drop_table(self, table: str) -> None:
        engine = self._engine_of(table)
        if engine is None:
            raise TableNotFound(f"la tabla «{table}» no existe")
        self.close_table(table)
        suffix = ENGINE_SUFFIX[engine]
        for ext in (".dat", ".ovf"):
            path = self.data_dir / f"{table}{suffix}{ext}"
            if path.exists():
                path.unlink()
        # Los indices de la tabla quedan huerfanos sin su tabla: se eliminan
        # junto con ella.
        for idx in self.indexes_of(table):
            self.drop_index(table, idx["name"])

    # --------------------------------------------------------------- indices

    def create_index(
        self, table: str, index_name: str, column: str, kind: str, options: dict | None = None
    ) -> None:
        if kind not in ("HASH", "BTREE", "IVF", "RTREE"):
            raise NotImplementedFeature(f"CREATE INDEX ... USING {kind} todavía no está implementado.")
        if self._engine_of(table) is None:
            raise TableNotFound(f"la tabla «{table}» no existe")
        if any(i["name"] == index_name for i in self.indexes_of(table)):
            raise TableAlreadyExists(f"el indice «{index_name}» ya existe en «{table}»")

        f = self.open_table(table)
        col = next((c for c in f.schema.columns if c.name.lower() == column.lower()), None)
        if col is None:
            raise ColumnNotFound(f"la columna «{column}» no existe en «{table}»")
        col_idx = f.schema.columns.index(col)
        base = self._index_base(table, index_name)

        if kind == "RTREE":
            if col.type is not ColumnType.POINT:
                raise ColumnNotFound(
                    f"RTREE solo indexa columnas POINT; «{col.name}» es {col.type.name}"
                )
            tree = RTree(base, col, page_size=self.page_size, cache_size=self.cache_size)
            tree.bulk_load(
                (row[col_idx], RTreeRID.from_engine_rid(rid))
                for rid, row in f.scan_with_rid()
            )
            tree.flush()
            self._open_indexes[(table, index_name)] = tree
            return

        if kind == "IVF":
            self._create_ivf(f, base, col, col_idx, options or {})
            self._open_indexes[(table, index_name)] = IVFIndex(
                base, page_size=self.page_size, cache_size=self.cache_size)
            return
        if col.type is ColumnType.VECTOR:
            raise SQLRuntimeError(
                f"la columna «{col.name}» es VECTOR: indexala con USING IVF, no con {kind}"
            )

        if kind == "HASH":
            idx = ExtendibleHash(base, col, page_size=self.page_size, cache_size=self.cache_size)
            for rid, row in f.scan_with_rid():
                idx.insert(row[col_idx], HashRID.from_engine_rid(rid))
            idx.flush()
            self._open_indexes[(table, index_name)] = idx
            return

        tree = BPlusTree(base, col, page_size=self.page_size, cache_size=self.cache_size)
        # Construccion masiva: recorre la tabla una sola vez e inserta cada
        # (clave, RID) -- equivalente a un bulk_load para el indice. OJO:
        # la clave del indice es el valor de `column`, no la PK de la tabla
        # (key_from_bytes siempre extrae la PK; por eso se usa unpack()
        # completo y se proyecta la columna indexada).
        for rid_page in range(f.page_count):
            page = f._read_page(rid_page) if hasattr(f, "_read_page") else f._read_main(rid_page)
            for slot in range(getattr(page, "capacity", 0)):
                if page.is_occupied(slot):
                    raw = page.read_record(slot)
                    values = f.schema.unpack(raw)
                    tree.insert(values[col_idx], BTreeRID(page.page_id, slot))
        tree.flush()
        self._open_indexes[(table, index_name)] = tree

    def _create_ivf(self, f, base: str, col: Column, col_idx: int, options: dict) -> None:
        """Construye un IVF sobre la columna VECTOR leyendo la tabla una sola vez.

        Opciones (WITH ...): lists = C (por defecto floor(sqrt(N))),
        metric = 'L2' | 'COSINE' (por defecto L2), iters = iteraciones de K-Means.
        """
        if col.type is not ColumnType.VECTOR:
            raise SQLRuntimeError(f"IVF solo indexa columnas VECTOR; «{col.name}» es {col.type.name}")
        metric = str(options.get("metric", "L2")).upper()
        if metric not in METRICS:
            raise SQLRuntimeError(f"metric = '{metric}' no soportada. Usa 'L2' o 'COSINE'.")
        lists = options.get("lists")
        if lists is not None and (not isinstance(lists, int) or lists <= 0):
            raise SQLRuntimeError(f"lists debe ser un entero positivo, no {lists!r}")
        iters = options.get("iters", 20)
        desconocidas = set(options) - {"metric", "lists", "iters"}
        if desconocidas:
            raise SQLRuntimeError(f"opciones de IVF desconocidas: {', '.join(sorted(desconocidas))}")

        pk = f.schema.pk_index
        idx = IVFIndex(base, col, f.schema.pk_column, metric=metric,
                       page_size=self.page_size, cache_size=self.cache_size)
        try:
            idx.build(((row[pk], rid, row[col_idx]) for rid, row in f.scan_with_rid()),
                      n_lists=lists, n_iter=int(iters))
        finally:
            idx.close()

    def open_index(self, table: str, index_name: str) -> ExtendibleHash | BPlusTree | IVFIndex:
        key = (table, index_name)
        if key in self._open_indexes:
            return self._open_indexes[key]
        base = self._index_base(table, index_name)
        if Path(f"{base}.ivf.dir").exists():
            header = peek_header(f"{base}.ivf.dir")
            idx = IVFIndex(base, page_size=header["page_size"], cache_size=self.cache_size)
            self._open_indexes[key] = idx
            return idx
        if Path(f"{base}.hash.dir").exists():
            header = peek_header(f"{base}.hash.dir")
            idx = ExtendibleHash(base, header["schema"].columns[0], page_size=header["page_size"], cache_size=self.cache_size)
            self._open_indexes[key] = idx
            return idx
        if Path(f"{base}.rtree.dat").exists():
            header = peek_header(f"{base}.rtree.dat")
            # La entrada del R-Tree mide 8*dim (MBR float32) + 10 (RID).
            dim = max(1, (header["record_size"] - 10) // 8)
            idx = RTree(
                base, header["schema"].columns[0],
                page_size=header["page_size"], cache_size=self.cache_size, dim=dim,
            )
            self._open_indexes[key] = idx
            return idx
        path = Path(f"{base}.btree.dat")
        if not path.exists():
            raise TableNotFound(f"el indice «{index_name}» no existe en «{table}»")
        header = peek_header(str(path))
        tree = BPlusTree(base, header["schema"].columns[0], page_size=header["page_size"], cache_size=self.cache_size)
        self._open_indexes[key] = tree
        return tree

    def close_index(self, table: str, index_name: str) -> None:
        idx = self._open_indexes.pop((table, index_name), None)
        if idx is not None:
            idx.close()

    def drop_index(self, table: str, index_name: str) -> None:
        self.close_index(table, index_name)
        base = self._index_base(table, index_name)
        for path in (Path(f"{base}.hash.dir"), Path(f"{base}.hash.bkt"), Path(f"{base}.btree.dat"),
                     Path(f"{base}.ivf.dir"), Path(f"{base}.ivf.lst"), Path(f"{base}.rtree.dat")):
            if path.exists():
                path.unlink()

    def row_at(self, table: str, rid: HashRID) -> tuple | None:
        f = self.open_table(table)
        if self.engine_of(table) == "HEAP":
            page = f.pager.read_page(rid.page_id)
        else:
            page = (f.main if rid.area == 0 else f.ovf).read_page(rid.page_id)
        if not page.is_occupied(rid.slot):
            return None
        return f.schema.unpack(page.read_record(rid.slot))

    # ------------------------------------------------------------- acceso

    def engine_of(self, table: str) -> str:
        engine = self._engine_of(table)
        if engine is None:
            raise TableNotFound(f"la tabla «{table}» no existe")
        return engine

    def open_table(self, table: str):
        if table in self._open:
            return self._open[table]
        engine = self.engine_of(table)

        # El esquema y el page_size se leen de la pagina 0: el archivo es
        # autodescriptivo, asi que una tabla creada con otro tamano de bloque
        # se reabre correctamente sin metadatos externos.
        header = peek_header(f"{self._base(table)}{ENGINE_SUFFIX[engine]}.dat")
        f = ENGINE_CLASS[engine](
            self._base(table), header["schema"],
            page_size=header["page_size"], cache_size=self.cache_size,
        )
        self._open[table] = f
        return f

    def close_table(self, table: str) -> None:
        sf = self._open.pop(table, None)
        if sf is not None:
            sf.close()

    def close_all(self) -> None:
        for table in list(self._open):
            self.close_table(table)
        for table, index_name in list(self._open_indexes):
            self.close_index(table, index_name)

    # ------------------------------------------------------------- listado

    def table_names(self) -> list[str]:
        nombres = set()
        for path in self.data_dir.glob("*.dat"):
            stem = path.name
            for suffix in SUFFIX_ENGINE:
                if stem.endswith(f"{suffix}.dat"):
                    nombres.add(stem[: -len(f"{suffix}.dat")])
        return sorted(nombres)

    def describe(self, table: str) -> TableInfo:
        engine = self._engine_of(table)
        if engine is None:
            raise TableNotFound(f"la tabla «{table}» no existe")
        f = self.open_table(table)
        suffix = ENGINE_SUFFIX[engine]
        size = sum(
            (self.data_dir / f"{table}{suffix}{ext}").stat().st_size
            for ext in (".dat", ".ovf")
            if (self.data_dir / f"{table}{suffix}{ext}").exists()
        )
        return TableInfo(
            name=table,
            engine=engine,
            schema=f.schema,
            n_records=f.n_records,
            # El Heap File no tiene area de overflow: es un concepto exclusivo
            # del Sequential File.
            n_overflow=getattr(f, "n_overflow", 0),
            page_count=f.page_count,
            page_size=f.page_size,
            record_size=f.schema.record_size,
            size_bytes=size,
            indexes=self.indexes_of(table),
        )

    def indexes_of(self, table: str) -> list[dict]:
        """Indices secundarios de una tabla.

        El Sequential File tiene un indice primario implicito: la ordenacion
        fisica por clave, sobre la que hace busqueda binaria. El Heap File no
        tiene ninguno, y por eso toda consulta sobre el es un full scan.
        Los indices BTREE y HASH se descubren igual que las tablas: escaneando
        el directorio en busca de <tabla>__<nombre>.btree.dat o
        <tabla>__<nombre>.hash.dir y leyendo su columna clave con
        peek_header, sin metadatos externos.
        """
        out = []
        if self._engine_of(table) == "SEQUENTIAL":
            f = self.open_table(table)
            out.append({
                "name": f"pk_{table}",
                "column": f.schema.pk_column.name,
                "kind": "SEQUENTIAL",
                "implicit": True,
            })

        prefix = f"{table}{INDEX_SUFFIX}"
        for path in self.data_dir.glob(f"{prefix}*.hash.dir"):
            index_name = path.name[len(prefix) : -len(".hash.dir")]
            header = peek_header(str(path))
            out.append({
                "name": index_name,
                "column": header["schema"].columns[0].name,
                "kind": "HASH",
                "implicit": False,
            })
        for path in self.data_dir.glob(f"{prefix}*.ivf.dir"):
            index_name = path.name[len(prefix) : -len(".ivf.dir")]
            stats = self.open_index(table, index_name).stats()
            out.append({
                "name": index_name,
                "column": peek_header(str(path))["schema"].columns[0].name,
                "kind": "IVF",
                "implicit": False,
                **{k: stats[k] for k in ("metric", "dim", "n_lists", "n_vectors")},
            })
        for path in self.data_dir.glob(f"{prefix}*.rtree.dat"):
            index_name = path.name[len(prefix) : -len(".rtree.dat")]
            header = peek_header(str(path))
            out.append({
                "name": index_name,
                "column": header["schema"].columns[0].name,
                "kind": "RTREE",
                "implicit": False,
            })
        for path in self.data_dir.glob(f"{prefix}*.btree.dat"):
            index_name = path.name[len(prefix) : -len(".btree.dat")]
            header = peek_header(str(path))
            out.append({
                "name": index_name,
                "column": header["schema"].columns[0].name,
                "kind": "BTREE",
                "implicit": False,
            })
        return out

    def list_tables(self) -> list[TableInfo]:
        return [self.describe(t) for t in self.table_names()]