"""Experimento 2 (enunciado 4.2): Búsquedas Puntuales de Igualdad.

Ejecuta N_QUERIES consultas aleatorias de igualdad sobre N registros y
compara promedio/desviación estándar de I/O reads y latencia (ms) entre:
Full Scan (Heap), Búsqueda Binaria (Sequential), Árbol B+ y Hash Dinámico.

PENDIENTE: exportar resultados a CSV/gráfica, conectar a
POST /api/benchmarks/run.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from backend.engine.catalog import Catalog
from backend.index.btree import BPlusTree
from backend.index.btree import RID as BTreeRID
from backend.index.hash import ExtendibleHash
from backend.index.hash import RID as HashRID
from backend.storage.schema import Column, ColumnType, Schema
from backend.structures.sequential_file import SequentialFile

N = 100_000
N_QUERIES = 1_000

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)


@dataclass
class SearchStats:
    access: str
    avg_reads: float
    std_reads: float
    avg_latency_ms: float
    std_latency_ms: float


def _generar_filas(n: int) -> list[tuple]:
    ids = list(range(n))
    random.shuffle(ids)  # inserción en orden aleatorio, no ya ordenada por PK
    return [(i, f"reg{i}", round(random.uniform(1, 10_000), 2)) for i in ids]


def _medir(access: str, queries: list[int], buscar: Callable[[int], None], leer_reads: Callable[[], int]) -> SearchStats:
    reads: list[int] = []
    latencias: list[float] = []
    for key in queries:
        r0 = leer_reads()
        t0 = time.perf_counter()
        buscar(key)
        latencias.append((time.perf_counter() - t0) * 1000.0)
        reads.append(leer_reads() - r0)
    return SearchStats(
        access=access,
        avg_reads=statistics.mean(reads),
        std_reads=statistics.pstdev(reads),
        avg_latency_ms=statistics.mean(latencias),
        std_latency_ms=statistics.pstdev(latencias),
    )


def _heap_full_scan(data_dir: Path, filas: list[tuple], queries: list[int]) -> SearchStats:
    data_dir.mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(data_dir), page_size=4096)
    cat.create_table("bench_heap", SCHEMA, "HEAP")
    f = cat.open_table("bench_heap")
    for fila in filas:
        f.insert(fila)
    f.flush()

    stats = _medir("FULL_SCAN_HEAP", queries, lambda k: f.search(k), lambda: f.counter.disk_reads)
    cat.close_all()
    return stats


def _sequential_binary_search(data_dir: Path, filas: list[tuple], queries: list[int]) -> SearchStats:
    data_dir.mkdir(parents=True, exist_ok=True)
    sf = SequentialFile(str(data_dir / "bench_sequential"), SCHEMA, page_size=4096)
    for fila in filas:
        sf.insert(fila)
    sf.flush()

    stats = _medir("BINARY_SEARCH_SEQUENTIAL", queries, lambda k: sf.search(k), lambda: sf.counter.disk_reads)
    sf.close()
    return stats


def _btree_search(data_dir: Path, filas: list[tuple], queries: list[int]) -> SearchStats:
    data_dir.mkdir(parents=True, exist_ok=True)
    tree = BPlusTree(str(data_dir / "bench_btree"), SCHEMA.columns[0], page_size=4096)
    for i, fila in enumerate(filas):
        tree.insert(fila[0], BTreeRID(i, 0))
    tree.flush()

    stats = _medir("BTREE", queries, lambda k: tree.search(k), lambda: tree.counter.disk_reads)
    tree.close()
    return stats


def _hash_search(data_dir: Path, filas: list[tuple], queries: list[int]) -> SearchStats:
    data_dir.mkdir(parents=True, exist_ok=True)
    idx = ExtendibleHash(str(data_dir / "bench_hash"), SCHEMA.columns[0], page_size=4096)
    for i, fila in enumerate(filas):
        idx.insert(fila[0], HashRID(i, 0))
    idx.flush()

    leer = lambda: idx.dir_pager.counter.disk_reads + idx.bkt_pager.counter.disk_reads
    stats = _medir("HASH", queries, lambda k: idx.search(k), leer)
    idx.close()
    return stats


def run(data_dir: str = "data/benchmarks/exp2") -> list[SearchStats]:
    """Corre el Experimento 2: N_QUERIES igualdades aleatorias sobre N filas."""
    base = Path(data_dir)
    filas = _generar_filas(N)
    queries = random.sample(range(N), N_QUERIES)

    return [
        _heap_full_scan(base / "heap", filas, queries),
        _sequential_binary_search(base / "sequential", filas, queries),
        _btree_search(base / "btree", filas, queries),
        _hash_search(base / "hash", filas, queries),
    ]
