"""Experimento 4 (enunciado 4.4): Sensibilidad al Tamaño de Bloque.

Varía el tamaño de página B en PAGE_SIZES y mide su efecto en el fan-out,
la altura h del Árbol B+ y el número total de I/Os al insertar N claves.

PENDIENTE: exportar resultados a CSV/gráfica, conectar a
POST /api/benchmarks/run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.index.btree import BPlusTree
from backend.index.btree import RID as BTreeRID
from backend.storage.schema import Column, ColumnType

N = 100_000
PAGE_SIZES = [1024, 2048, 4096, 8192]

KEY_COLUMN = Column("id", ColumnType.INT)


@dataclass
class BlockSizeStats:
    page_size: int
    fanout: int
    height: int
    disk_reads: int
    disk_writes: int
    total_ms: float


def _generar_claves(n: int) -> list[int]:
    claves = list(range(n))
    random.shuffle(claves)  # inserción en orden aleatorio, no ya ordenada
    return claves


def _medir(data_dir: Path, page_size: int, claves: list[int]) -> BlockSizeStats:
    data_dir.mkdir(parents=True, exist_ok=True)
    tree = BPlusTree(str(data_dir / "bench_btree"), KEY_COLUMN, page_size=page_size)

    t0 = time.perf_counter()
    for i, key in enumerate(claves):
        tree.insert(key, BTreeRID(i, 0))
    tree.flush()
    total_ms = (time.perf_counter() - t0) * 1000.0

    stats = BlockSizeStats(
        page_size=page_size,
        fanout=tree.fanout,
        height=tree.height,
        disk_reads=tree.counter.disk_reads,
        disk_writes=tree.counter.disk_writes,
        total_ms=total_ms,
    )
    tree.close()
    return stats


def run(data_dir: str = "data/benchmarks/exp4") -> list[BlockSizeStats]:
    """Corre el Experimento 4 para cada tamaño de página de PAGE_SIZES."""
    base = Path(data_dir)
    claves = _generar_claves(N)
    return [_medir(base / str(page_size), page_size, claves) for page_size in PAGE_SIZES]
