"""Experimento 3 (enunciado 4.3): Búsquedas por Rango con Selectividad Variable.

Evaluar el costo de I/O y tiempo al consultar rangos que representan el 
0.1 %, 1 %, 5 %, 10 % y 25 % del total de tuplas.
Contrastar el Árbol B+ frente al Sequential File y al Full Scan (Heap).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.storage.schema import Column, ColumnType, Schema

N_RECORDS = 100_000
SELECTIVITIES = [0.001, 0.01, 0.05, 0.10, 0.25]

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)

@dataclass
class RangeResult:
    engine: str
    selectivity: float
    total_ms: float
    disk_reads: int
    
    def as_dict(self):
        return {
            "engine": self.engine,
            "selectivity_pct": round(self.selectivity * 100, 1),
            "total_ms": round(self.total_ms, 3),
            "disk_reads": self.disk_reads,
        }

def _generar_filas(n: int) -> list[tuple]:
    ids = list(range(n))
    random.shuffle(ids)
    return [(i, f"reg{i}", round(random.uniform(1, 10_000), 2)) for i in ids]


def _test_engine(data_dir: Path, base_engine: str, index_kind: str | None, label: str, filas: list[tuple]) -> list[RangeResult]:
    cat = Catalog(str(data_dir), page_size=4096)
    table = f"bench_{base_engine.lower()}"
    cat.create_table(table, SCHEMA, base_engine)
    
    if index_kind:
        cat.create_index(table, "idx", "id", index_kind)
        
    f = cat.open_table(table)
    
    for fila in filas:
        f.insert(fila)
    f.flush()
    cat.close_all()
    
    cat3 = Catalog(str(data_dir), page_size=4096)
    f3 = cat3.open_table(table)
    idx3 = cat3.open_index(table, "idx") if index_kind else None
    
    resultados = []
    
    for sel in SELECTIVITIES:
        # we know keys are 0 to N_RECORDS-1
        range_size = int(N_RECORDS * sel)
        lo = random.randint(0, N_RECORDS - range_size - 1)
        hi = lo + range_size - 1
        
        f3.counter.reset()
        if idx3:
            if hasattr(idx3, "pager"):
                idx3.pager.counter.reset()
                
        t0 = time.perf_counter()
        
        if index_kind:
            # BTREE range search
            for rid in idx3.range_search(lo, hi):
                if base_engine == "HEAP":
                    page = f3.pager.read_page(rid.page_id)
                else:
                    page = f3.main.read_page(rid.page_id)
                page.read_record(rid.slot)
        else:
            if base_engine == "SEQUENTIAL":
                for row in f3.range_search(lo, hi):
                    pass # sequential iterates rows
            else: # HEAP
                for row in f3.scan():
                    if lo <= row[0] <= hi:
                        pass
                        
        ms = (time.perf_counter() - t0) * 1000.0
        
        q_reads = f3.counter.disk_reads
        if idx3:
            if hasattr(idx3, "pager"):
                q_reads += idx3.pager.counter.disk_reads
                
        resultados.append(RangeResult(label, sel, ms, q_reads))
        
    cat3.close_all()
    
    return resultados

def run(data_dir: str = "data/benchmarks/exp3") -> list[dict]:
    base = Path(data_dir)
    filas = _generar_filas(N_RECORDS)
    
    resultados = []
    
    engines = [
        ("HEAP", None, "Full Scan (Heap)"), 
        ("SEQUENTIAL", None, "Sequential File"),
        ("HEAP", "BTREE", "Árbol B+")
    ]
               
    for base_engine, index_kind, label in engines:
        run_dir = base / label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        res_list = _test_engine(run_dir, base_engine, index_kind, label, filas)
        for r in res_list:
            resultados.append(r.as_dict())
            
    return resultados
