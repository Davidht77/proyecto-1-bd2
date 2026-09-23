"""Experimento 4 (enunciado 4.4): Sensibilidad al Tamaño de Bloque.

Variar el tamaño de página B in [1024, 2048, 4096, 8192] bytes y analizar
el efecto en el factor de ramificación (fan-out), la altura h del árbol B+
y el número total de I/Os transferidos (inserción).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.index.btree import BPlusTree
from backend.index.btree import RID as BTreeRID
from backend.storage.schema import Column, ColumnType, Schema

N_RECORDS = 100_000
PAGE_SIZES = [1024, 2048, 4096, 8192]

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)

@dataclass
class SizeResult:
    page_size: int
    fan_out: int
    height: int
    total_ms: float
    disk_writes: int
    disk_reads: int
    
    def as_dict(self):
        return {
            "page_size": self.page_size,
            "fan_out": self.fan_out,
            "height": self.height,
            "total_ms": round(self.total_ms, 3),
            "disk_writes": self.disk_writes,
            "disk_reads": self.disk_reads,
            "total_ios": self.disk_writes + self.disk_reads
        }


def _test_size(data_dir: Path, page_size: int, ids: list[int]) -> SizeResult:
    idx = BPlusTree(str(data_dir / "bench_btree"), SCHEMA.columns[0], page_size=page_size)
    
    t0 = time.perf_counter()
    for i, pk in enumerate(ids):
        # We insert the primary key and a synthetic RID
        idx.insert(pk, BTreeRID(i // 100, i % 100))
    idx.flush()
    ms = (time.perf_counter() - t0) * 1000.0
    
    writes = idx.pager.counter.disk_writes
    reads = idx.pager.counter.disk_reads
    fan_out = idx.fanout
    height = idx.height
    
    idx.close()
    
    return SizeResult(page_size, fan_out, height, ms, writes, reads)


def run(data_dir: str = "data/benchmarks/exp4") -> list[dict]:
    base = Path(data_dir)
    
    ids = list(range(N_RECORDS))
    random.shuffle(ids)
    
    resultados = []
    
    for ps in PAGE_SIZES:
        run_dir = base / f"size_{ps}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        res = _test_size(run_dir, ps, ids)
        resultados.append(res.as_dict())
        
    return resultados
