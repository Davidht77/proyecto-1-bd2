"""Experimento 1 (enunciado 4.1): Costo de Inserción Masiva.

Compara el tiempo total y las escrituras I/O al insertar lotes crecientes de
tuplas en cada motor de almacenamiento.

PENDIENTE (no cubierto todavia por este modulo):
  - Arbol B+ y Hash Dinamico: a diferencia de Heap/Sequential, no son un
    motor de tabla; hay que crear la tabla + CREATE INDEX y medir el costo
    de construir el indice, no solo el insert.
  - Sequential File con y sin reorganizacion (el enunciado pide comparar
    ambos casos).
  - Exportar resultados a CSV/grafica para el informe.
  - Conectar esto a POST /api/benchmarks/run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.storage.schema import Column, ColumnType, Schema

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


def run(data_dir: str = "data/benchmarks/exp1") -> list[InsertResult]:
    """Corre el Experimento 1 para HEAP y SEQUENTIAL sobre cada N de N_VALUES."""
    base = Path(data_dir)
    resultados = []
    for n in N_VALUES:
        filas = _generar_filas(n)
        for engine in ("HEAP", "SEQUENTIAL"):
            run_dir = base / engine.lower() / str(n)
            run_dir.mkdir(parents=True, exist_ok=True)
            resultados.append(_insertar(run_dir, engine, filas))
    return resultados
