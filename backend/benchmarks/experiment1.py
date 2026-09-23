"""Experimento 1 (enunciado 4.1): Costo de Inserción Masiva.

Compara el tiempo total y las escrituras I/O al insertar lotes crecientes de
tuplas en cada motor de almacenamiento (Heap, Sequential con y sin
reorganizacion automatica) y claves en el Hash Dinamico (indice, no motor
de tabla: se mide el insert directo sobre el indice con RIDs sinteticos,
sin pasar por una tabla real).

El Sequential se construye con SequentialFile directo (no via Catalog):
Catalog.create_table no expone auto_reorganize, y el flag no se guarda en
el header del archivo, asi que reabrir con Catalog siempre volveria a
True.

PENDIENTE (no cubierto todavia por este modulo):
  - Arbol B+: a diferencia de Heap/Sequential/Hash, no expone un solo
    DiskCounter (falta decidir si se mide sobre una tabla real o, como el
    Hash aqui, con claves sinteticas directo al indice).
  - Exportar resultados a CSV/grafica para el informe.
  - Conectar esto a POST /api/benchmarks/run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.index.hash import ExtendibleHash
from backend.index.hash import RID as HashRID
from backend.storage.schema import Column, ColumnType, Schema
from backend.structures.sequential_file import SequentialFile

N_VALUES = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000]

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)


@dataclass
class InsertResult:
    engine: str
    n: int
    total_ms: float
    disk_writes: int


def _generar_filas(n: int) -> list[tuple]:
    ids = list(range(n))
    random.shuffle(ids)  # inserción en orden aleatorio, no ya ordenada por PK
    return [(i, f"reg{i}", round(random.uniform(1, 10_000), 2)) for i in ids]


def _insertar(data_dir: Path, engine: str, filas: list[tuple]) -> InsertResult:
    cat = Catalog(str(data_dir), page_size=4096)
    table = f"bench_{engine.lower()}"
    cat.create_table(table, SCHEMA, engine)
    f = cat.open_table(table)

    t0 = time.perf_counter()
    for fila in filas:
        f.insert(fila)
    f.flush()
    total_ms = (time.perf_counter() - t0) * 1000.0

    resultado = InsertResult(engine, len(filas), total_ms, f.counter.disk_writes)
    cat.close_all()
    return resultado


def _insertar_sequential(data_dir: Path, filas: list[tuple], auto_reorganize: bool) -> InsertResult:
    sf = SequentialFile(
        str(data_dir / "bench_sequential"), SCHEMA,
        page_size=4096, auto_reorganize=auto_reorganize,
    )

    t0 = time.perf_counter()
    for fila in filas:
        sf.insert(fila)
    sf.flush()
    total_ms = (time.perf_counter() - t0) * 1000.0

    sufijo = "CON_REORG" if auto_reorganize else "SIN_REORG"
    resultado = InsertResult(f"SEQUENTIAL_{sufijo}", len(filas), total_ms, sf.counter.disk_writes)
    sf.close()
    return resultado


def _insertar_hash(data_dir: Path, filas: list[tuple]) -> InsertResult:
    idx = ExtendibleHash(str(data_dir / "bench_hash"), SCHEMA.columns[0], page_size=4096)

    t0 = time.perf_counter()
    for i, fila in enumerate(filas):
        idx.insert(fila[0], HashRID(i, 0))
    idx.flush()
    total_ms = (time.perf_counter() - t0) * 1000.0

    writes = idx.dir_pager.counter.disk_writes + idx.bkt_pager.counter.disk_writes
    resultado = InsertResult("HASH", len(filas), total_ms, writes)
    idx.close()
    return resultado


def run(data_dir: str = "data/benchmarks/exp1") -> list[InsertResult]:
    """Corre el Experimento 1 para HEAP, SEQUENTIAL (con y sin reorg) y HASH."""
    base = Path(data_dir)
    resultados = []
    for n in N_VALUES:
        filas = _generar_filas(n)

        run_dir = base / "heap" / str(n)
        run_dir.mkdir(parents=True, exist_ok=True)
        resultados.append(_insertar(run_dir, "HEAP", filas))

        for auto_reorganize in (True, False):
            sufijo = "con_reorg" if auto_reorganize else "sin_reorg"
            run_dir = base / f"sequential_{sufijo}" / str(n)
            run_dir.mkdir(parents=True, exist_ok=True)
            resultados.append(_insertar_sequential(run_dir, filas, auto_reorganize))

        run_dir = base / "hash" / str(n)
        run_dir.mkdir(parents=True, exist_ok=True)
        resultados.append(_insertar_hash(run_dir, filas))
    return resultados
