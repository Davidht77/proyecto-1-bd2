"""Experimento 2 (enunciado 4.2): Búsquedas Puntuales de Igualdad.

Ejecutar un conjunto de 1000 consultas aleatorias de igualdad sobre N = 100,000 registros.
Comparar el promedio y desviación estándar de I/O reads y latencia (ms) entre:
Full Scan (Heap) vs Búsqueda Binaria (Sequential) vs Árbol B+ vs Hash Dinámico.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.storage.schema import Column, ColumnType, Schema

N_RECORDS = 100_000
N_QUERIES = 1000

SCHEMA = Schema(
    [
        Column("id", ColumnType.INT),
        Column("nombre", ColumnType.CHAR, length=20),
        Column("monto", ColumnType.FLOAT),
    ],
    pk_index=0,
)

@dataclass
class QueryResult:
    engine: str
    avg_ms: float
    std_ms: float
    avg_reads: float
    std_reads: float
    
    def as_dict(self):
        return {
            "engine": self.engine,
            "avg_ms": round(self.avg_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "avg_reads": round(self.avg_reads, 2),
            "std_reads": round(self.std_reads, 2),
        }

def _generar_filas(n: int) -> list[tuple]:
    ids = list(range(n))
    random.shuffle(ids)
    return [(i, f"reg{i}", round(random.uniform(1, 10_000), 2)) for i in ids]

def _calcular_stats(latencies: list[float], reads: list[int]) -> tuple[float, float, float, float]:
    n = len(latencies)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    avg_ms = sum(latencies) / n
    std_ms = math.sqrt(sum((x - avg_ms) ** 2 for x in latencies) / n) if n > 1 else 0.0
    
    avg_reads = sum(reads) / n
    std_reads = math.sqrt(sum((x - avg_reads) ** 2 for x in reads) / n) if n > 1 else 0.0
    
    return avg_ms, std_ms, avg_reads, std_reads

def _test_engine(data_dir: Path, base_engine: str, index_kind: str | None, label: str, filas: list[tuple], queries: list[int]) -> QueryResult:
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
    
    latencies = []
    reads = []
    
    for q in queries:
        f3.counter.reset()
        if idx3:
            if hasattr(idx3, "pager"):
                idx3.pager.counter.reset()
            elif hasattr(idx3, "dir_pager"):
                idx3.dir_pager.counter.reset()
                idx3.bkt_pager.counter.reset()
                
        t0 = time.perf_counter()
        
        if index_kind:
            if index_kind == "HASH":
                rids = idx3.search(q)
                if rids:
                    for rid in rids:
                        cat3.row_at(table, rid)
            else: # BTREE
                rid = idx3.search(q)
                if rid:
                    if base_engine == "HEAP":
                        page = f3.pager.read_page(rid.page_id)
                    else:
                        page = f3.main.read_page(rid.page_id)
                    page.read_record(rid.slot)
        else:
            if base_engine == "SEQUENTIAL":
                f3.search(q)
            else: # HEAP
                for row in f3.scan():
                    if row[0] == q:
                        break
                        
        ms = (time.perf_counter() - t0) * 1000.0
        
        q_reads = f3.counter.disk_reads
        if idx3:
            if hasattr(idx3, "pager"):
                q_reads += idx3.pager.counter.disk_reads
            elif hasattr(idx3, "dir_pager"):
                q_reads += idx3.dir_pager.counter.disk_reads + idx3.bkt_pager.counter.disk_reads
                
        latencies.append(ms)
        reads.append(q_reads)
        
    cat3.close_all()
    
    avg_ms, std_ms, avg_reads, std_reads = _calcular_stats(latencies, reads)
    
    return QueryResult(label, avg_ms, std_ms, avg_reads, std_reads)

def run(data_dir: str = "data/benchmarks/exp2") -> list[dict]:
    base = Path(data_dir)
    filas = _generar_filas(N_RECORDS)
    queries = [random.randint(0, N_RECORDS - 1) for _ in range(N_QUERIES)]
    
    resultados = []
    
    engines = [
        ("HEAP", None, "Full Scan (Heap)"), 
        ("SEQUENTIAL", None, "Búsqueda Binaria (Sequential)"),
        ("HEAP", "BTREE", "Árbol B+"),
        ("HEAP", "HASH", "Hash Dinámico")
    ]
               
    for base_engine, index_kind, label in engines:
        run_dir = base / label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        res = _test_engine(run_dir, base_engine, index_kind, label, filas, queries)
        resultados.append(res.as_dict())
        
    return resultados
