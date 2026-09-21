"""Catalogo de tablas.

No hay archivo de catalogo. El esquema de cada tabla ya vive en la pagina 0 de
su propio archivo de datos, asi que el catalogo se reconstruye escaneando el
directorio y leyendo esas cabeceras. Eso elimina la posibilidad de que el
catalogo y los datos se desincronicen, y de paso evita introducir un
serializador de alto nivel, que el enunciado prohibe.

El motor de almacenamiento se codifica en el sufijo del archivo:
    <tabla>.seq.dat   Sequential File
    <tabla>.heap.dat  Heap File
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.engine.errors import TableAlreadyExists, TableNotFound
from backend.storage.pager import peek_header
from backend.storage.schema import Schema
from backend.structures.heap_file import HeapFile
from backend.structures.sequential_file import SequentialFile

# Ambas organizaciones exponen la misma superficie: insert, search, delete,
# scan, range_search, bulk_load, flush y close. El motor las trata por igual;
# lo unico exclusivo del Sequential File es reorganize() y su overflow.
ENGINE_CLASS = {"SEQUENTIAL": SequentialFile, "HEAP": HeapFile}

ENGINE_SUFFIX = {"SEQUENTIAL": ".seq", "HEAP": ".heap"}
SUFFIX_ENGINE = {v: k for k, v in ENGINE_SUFFIX.items()}


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

    # ------------------------------------------------------------- rutas

    def _base(self, table: str) -> str:
        return str(self.data_dir / table)

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
        Los BTREE y HASH apareceran aqui cuando se implementen.
        """
        if self._engine_of(table) != "SEQUENTIAL":
            return []
        f = self.open_table(table)
        return [
            {
                "name": f"pk_{table}",
                "column": f.schema.pk_column.name,
                "kind": "SEQUENTIAL",
                "implicit": True,
            }
        ]

    def list_tables(self) -> list[TableInfo]:
        return [self.describe(t) for t in self.table_names()]
