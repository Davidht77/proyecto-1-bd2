"""Experimento 5: indice espacial R-Tree.

Dos suites:

1. ``spatial``: para N puntos 2D, compara el R-Tree contra el escaneo
   secuencial de un Heap File en consultas WITHIN_RADIUS (varias
   selectividades) y KNN (varios k).
2. ``dimension``: construye R-Trees genericos de dimension D = 2, 4, 8, ...,
   128 y compara su costo de I/O contra el costo analitico de un full scan.
   Muestra la maldicion de la dimensionalidad: al crecer D, los MBRs se
   solapan tanto que el arbol termina leyendo casi todas las hojas.

El resultado es una lista de ``SpatialStats`` que ``run_all.py`` vuelca a
``benchmarks/results/exp5_espacial.csv`` y ``exp5_dimension.csv``.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from backend.index import geo
from backend.index.rtree import RTree, RTreeRID
from backend.storage.page import page_capacity
from backend.storage.schema import Column, ColumnType, Schema
from backend.structures.heap_file import HeapFile

PAGE_SIZE = 4096
N_SIZES = [10_000, 50_000, 100_000]
SELECTIVITIES = [0.001, 0.01, 0.05, 0.10, 0.25]
K_VALUES = [5, 20, 50]
N_QUERIES = 30

DIM_NS = 10_000
DIMS = [2, 4, 8, 16, 32, 64, 128]
DIM_SELECTIVITY = 0.01
DIM_QUERIES = 20

# Caja geografica alrededor de Lima usada para generar los puntos.
LON_MIN, LON_MAX = -77.35, -76.70
LAT_MIN, LAT_MAX = -12.43, -11.84


@dataclass
class SpatialStats:
    suite: str
    structure: str
    n: int
    dim: int
    query_type: str
    selectivity: float
    k: int
    avg_reads: float
    std_reads: float
    avg_ms: float
    std_ms: float
    avg_results: float


def _points(n: int, seed: int = 7) -> list[tuple[float, float]]:
    rnd = random.Random(seed)
    return [
        (rnd.uniform(LON_MIN, LON_MAX), rnd.uniform(LAT_MIN, LAT_MAX))
        for _ in range(n)
    ]


def _radius_for_selectivity(selectivity: float) -> float:
    """Radio aproximado (km) cuya area cubre ``selectivity`` de la caja."""
    box_area = (LON_MAX - LON_MIN) * (LAT_MAX - LAT_MIN)  # grados^2
    radius_deg = math.sqrt(max(0.0, selectivity) * box_area / math.pi)
    return radius_deg * geo.KM_PER_DEGREE_LAT


def _stats(r_list, t_list, counts) -> tuple[float, float, float, float, float]:
    return (
        statistics.mean(r_list),
        statistics.pstdev(r_list),
        statistics.mean(t_list),
        statistics.pstdev(t_list),
        statistics.mean(counts),
    )


# --------------------------------------------------------------- suite 2D


def _build_heap(base: Path, pts: list[tuple[float, float]]) -> tuple[HeapFile, Schema]:
    schema = Schema(
        [Column("id", ColumnType.INT), Column("punto", ColumnType.POINT)],
        pk_index=0,
    )
    base.mkdir(parents=True, exist_ok=True)
    heap = HeapFile(str(base / "datos"), schema, page_size=PAGE_SIZE)
    heap.bulk_load((i, pts[i]) for i in range(len(pts)))
    heap.flush()
    return heap, schema


def _build_rtree(base: Path, heap: HeapFile, schema: Schema, pts):
    base.mkdir(parents=True, exist_ok=True)
    tree = RTree(str(base / "geo"), schema.columns[1], page_size=PAGE_SIZE)
    rid_to_id: dict[RTreeRID, int] = {}
    entries = []
    for rid, row in heap.scan_with_rid():
        spatial_rid = RTreeRID.from_engine_rid(rid)
        rid_to_id[spatial_rid] = row[0]
        entries.append((row[1], spatial_rid))
    tree.bulk_load(entries)
    tree.flush()
    return tree, rid_to_id


def _spatial_suite(data_dir: Path) -> list[SpatialStats]:
    out: list[SpatialStats] = []
    for n in N_SIZES:
        pts = _points(n)
        base = data_dir / f"n{n}"
        heap, schema = _build_heap(base / "heap", pts)
        tree, rid_to_id = _build_rtree(base / "rtree", heap, schema, pts)
        centers = random.Random(11).sample(pts, min(N_QUERIES, n))

        for selectivity in SELECTIVITIES:
            radius_km = _radius_for_selectivity(selectivity)

            r_list, t_list, counts = [], [], []
            for center in centers:
                r0 = tree.counter.disk_reads
                t0 = time.perf_counter()
                window = geo.radius_bbox(center[0], center[1], radius_km)
                count = 0
                for rid in tree.search_window(window, collect_mbrs=False):
                    idx = rid_to_id.get(rid)
                    if idx is None:
                        continue
                    p = pts[idx]
                    if geo.point_in_radius(p[0], p[1], center[0], center[1], radius_km):
                        count += 1
                t_list.append((time.perf_counter() - t0) * 1000.0)
                r_list.append(tree.counter.disk_reads - r0)
                counts.append(count)
            out.append(SpatialStats(
                "spatial", "RTREE", n, 2, "radius", selectivity, 0,
                *_stats(r_list, t_list, counts),
            ))

            r_list, t_list, counts = [], [], []
            for center in centers:
                r0 = heap.counter.disk_reads
                t0 = time.perf_counter()
                count = 0
                for row in heap.scan():
                    p = row[1]
                    if geo.point_in_radius(p[0], p[1], center[0], center[1], radius_km):
                        count += 1
                t_list.append((time.perf_counter() - t0) * 1000.0)
                r_list.append(heap.counter.disk_reads - r0)
                counts.append(count)
            out.append(SpatialStats(
                "spatial", "SEQUENTIAL", n, 2, "radius", selectivity, 0,
                *_stats(r_list, t_list, counts),
            ))

        for k in K_VALUES:
            r_list, t_list, counts = [], [], []
            for center in centers:
                r0 = tree.counter.disk_reads
                t0 = time.perf_counter()
                count = len(tree.knn(center, k, collect_mbrs=False))
                t_list.append((time.perf_counter() - t0) * 1000.0)
                r_list.append(tree.counter.disk_reads - r0)
                counts.append(count)
            out.append(SpatialStats(
                "spatial", "RTREE", n, 2, "knn", 0.0, k,
                *_stats(r_list, t_list, counts),
            ))

            r_list, t_list, counts = [], [], []
            for center in centers:
                r0 = heap.counter.disk_reads
                t0 = time.perf_counter()
                scored = sorted(
                    (
                        geo.haversine(center[0], center[1], row[1][0], row[1][1]),
                        row[0],
                    )
                    for row in heap.scan()
                )
                count = len(scored[:k])
                t_list.append((time.perf_counter() - t0) * 1000.0)
                r_list.append(heap.counter.disk_reads - r0)
                counts.append(count)
            out.append(SpatialStats(
                "spatial", "SEQUENTIAL", n, 2, "knn", 0.0, k,
                *_stats(r_list, t_list, counts),
            ))

        tree.close()
        heap.close()
    return out


# ---------------------------------------------------------- suite dimensional


def _dim_points(n: int, dim: int, seed: int = 23) -> list[tuple[float, ...]]:
    rnd = random.Random(seed)
    return [tuple(rnd.random() for _ in range(dim)) for _ in range(n)]


def _dimension_suite(data_dir: Path) -> list[SpatialStats]:
    out: list[SpatialStats] = []
    for dim in DIMS:
        pts = _dim_points(DIM_NS, dim)
        base = data_dir / f"dim{dim}"
        base.mkdir(parents=True, exist_ok=True)
        tree = RTree(str(base / "rtree"), page_size=PAGE_SIZE, dim=dim)
        tree.bulk_load(
            (p, RTreeRID(i // 1000, i % 1000, 0)) for i, p in enumerate(pts)
        )
        tree.flush()

        side = DIM_SELECTIVITY ** (1.0 / dim)
        centers = random.Random(31).sample(pts, min(DIM_QUERIES, DIM_NS))

        r_list, t_list, counts = [], [], []
        for center in centers:
            r0 = tree.counter.disk_reads
            t0 = time.perf_counter()
            window = tuple(c - side / 2 for c in center) + tuple(
                c + side / 2 for c in center
            )
            count = len(tree.search_window(window, collect_mbrs=False))
            t_list.append((time.perf_counter() - t0) * 1000.0)
            r_list.append(tree.counter.disk_reads - r0)
            counts.append(count)
        out.append(SpatialStats(
            "dimension", "RTREE", DIM_NS, dim, "window", DIM_SELECTIVITY, 0,
            *_stats(r_list, t_list, counts),
        ))

        # Costo analitico de recorrer una tabla con N registros de dim*8+4 B.
        record_size = 4 + 8 * dim
        capacity = page_capacity(PAGE_SIZE, record_size)
        seq_pages = math.ceil(DIM_NS / capacity)
        out.append(SpatialStats(
            "dimension", "SEQUENTIAL", DIM_NS, dim, "window", DIM_SELECTIVITY, 0,
            float(seq_pages), 0.0, 0.0, 0.0, float(DIM_NS * DIM_SELECTIVITY),
        ))
        tree.close()
    return out


def run(data_dir: str = "data/benchmarks/exp5") -> list[SpatialStats]:
    base = Path(data_dir)
    # El benchmark se puede repetir: se parte siempre de archivos vacios.
    if base.exists():
        import shutil

        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return _spatial_suite(base / "spatial") + _dimension_suite(base / "dimension")


def write_csvs(rows: list[SpatialStats], results_dir: Path) -> list[Path]:
    """Escribe las dos salidas del Experimento 5."""
    import csv
    from dataclasses import asdict

    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suite, name in (("spatial", "exp5_espacial"), ("dimension", "exp5_dimension")):
        subset = [r for r in rows if r.suite == suite]
        if not subset:
            continue
        path = results_dir / f"{name}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(subset[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in subset)
        paths.append(path)
    return paths


if __name__ == "__main__":
    filas = run()
    for path in write_csvs(filas, Path("benchmarks/results")):
        print(f"-> {path} ({len(filas)} filas)")

