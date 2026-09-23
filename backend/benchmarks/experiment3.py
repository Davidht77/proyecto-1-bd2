"""Experimento 3 (enunciado 4.3): Búsquedas por Rango con Selectividad Variable.

Evalúa el costo de I/O y tiempo al consultar rangos que representan
0.1%, 1%, 5%, 10% y 25% del total de tuplas. Contrasta Árbol B+ frente a
Sequential File y Full Scan (Heap). Hash Dinámico no entra: no soporta
búsqueda por rango (enunciado 3.3.2), solo igualdad.

PENDIENTE: exportar resultados a CSV/gráfica, conectar a
POST /api/benchmarks/run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.index.btree import BPlusTree
from backend.index.btree import RID as BTreeRID
from backend.storage.schema import Column, ColumnType, Schema
from backend.structures.sequential_file import SequentialFile

N = 100_000
SELECTIVIDADES = [0.001, 0.01, 0.05, 0.10, 0.25]

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)


@dataclass
class RangeStats:
    access: str
    selectividad: float
    n_esperado: int
    n_obtenido: int
    disk_reads: int
    latency_ms: float


def _generar_filas(n: int) -> list[tuple]:
    ids = list(range(n))
    random.shuffle(ids)  # inserción en orden aleatorio, no ya ordenada por PK
    return [(i, f"reg{i}", round(random.uniform(1, 10_000), 2)) for i in ids]


def _rangos(n: int) -> list[tuple[float, int, int]]:
    rangos = []
    for s in SELECTIVIDADES:
        ancho = max(1, int(n * s))
        lo = random.randint(0, n - ancho)
        rangos.append((s, lo, lo + ancho - 1))
    return rangos


def _medir_heap(data_dir: Path, filas: list[tuple], rangos: list[tuple[float, int, int]]) -> list[RangeStats]:
    data_dir.mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(data_dir), page_size=4096)
    cat.create_table("bench_heap", SCHEMA, "HEAP")
    f = cat.open_table("bench_heap")
    for fila in filas:
        f.insert(fila)
    f.flush()

    stats = []
    for s, lo, hi in rangos:
        f.counter.reset()
        t0 = time.perf_counter()
        n_obtenido = sum(1 for fila in f.scan() if lo <= fila[0] <= hi)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        stats.append(RangeStats("FULL_SCAN_HEAP", s, hi - lo + 1, n_obtenido, f.counter.disk_reads, latency_ms))
    cat.close_all()
    return stats


def _medir_sequential(data_dir: Path, filas: list[tuple], rangos: list[tuple[float, int, int]]) -> list[RangeStats]:
    data_dir.mkdir(parents=True, exist_ok=True)
    sf = SequentialFile(str(data_dir / "bench_sequential"), SCHEMA, page_size=4096)
    for fila in filas:
        sf.insert(fila)
    sf.flush()

    stats = []
    for s, lo, hi in rangos:
        sf.counter.reset()
        t0 = time.perf_counter()
        n_obtenido = sum(1 for _ in sf.range_search(lo, hi))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        stats.append(RangeStats("SEQUENTIAL", s, hi - lo + 1, n_obtenido, sf.counter.disk_reads, latency_ms))
    sf.close()
    return stats


def _medir_btree(data_dir: Path, filas: list[tuple], rangos: list[tuple[float, int, int]]) -> list[RangeStats]:
    data_dir.mkdir(parents=True, exist_ok=True)
    tree = BPlusTree(str(data_dir / "bench_btree"), SCHEMA.columns[0], page_size=4096)
    for i, fila in enumerate(filas):
        tree.insert(fila[0], BTreeRID(i, 0))
    tree.flush()

    stats = []
    for s, lo, hi in rangos:
        r0 = tree.counter.disk_reads
        t0 = time.perf_counter()
        n_obtenido = sum(1 for _ in tree.range_search(lo, hi))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        stats.append(RangeStats("BTREE", s, hi - lo + 1, n_obtenido, tree.counter.disk_reads - r0, latency_ms))
    tree.close()
    return stats


def run(data_dir: str = "data/benchmarks/exp3") -> list[RangeStats]:
    """Corre el Experimento 3: un rango por cada selectividad de SELECTIVIDADES."""
    base = Path(data_dir)
    filas = _generar_filas(N)
    rangos = _rangos(N)

    resultados = []
    resultados += _medir_heap(base / "heap", filas, rangos)
    resultados += _medir_sequential(base / "sequential", filas, rangos)
    resultados += _medir_btree(base / "btree", filas, rangos)
    return resultados
