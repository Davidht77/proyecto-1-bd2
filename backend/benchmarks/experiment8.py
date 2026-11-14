"""Experimento 8 (enunciado 6.4): compromiso eficiencia vs calidad en ANN.

8a. Recall@10 vs latencia vs bloques leidos al variar `ivf.probes`, tomando el
    k-NN lineal exacto (KnnSeqScan) como verdad de campo.
8b. Planificador hibrido: costo de Pre-filtering vs Post-filtering al variar la
    selectividad del filtro relacional (B+ sobre `precio`). Sirve para calibrar
    hybrid.choose_strategy y para justificar su umbral en el informe.

Todo pasa por el Executor (SQL real), asi que las metricas son las mismas que
ve el cliente web, con el I/O desglosado en indice y tabla.

Uso:
    python -m backend.benchmarks.experiment8                 # sintetico, N=20k, D=128
    python -m backend.benchmarks.experiment8 --n 50000
    python -m backend.benchmarks.experiment8 --npy emb.npy   # embeddings reales (N, D)
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
from dataclasses import dataclass

import numpy as np

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor

PROBES = [1, 2, 5, 10, 20, 50]
SELECTIVIDADES = [0.001, 0.01, 0.05, 0.25, 1.0]
K = 10


@dataclass
class AnnRow:
    method: str          # "exact" | "ivf"
    probes: int          # 0 para el exacto
    n: int
    dim: int
    n_lists: int
    recall_at_10: float
    avg_ms: float
    std_ms: float
    avg_reads: float
    avg_index_reads: float
    avg_table_reads: float


@dataclass
class HybridRow:
    selectivity: float
    strategy: str        # "PRE" | "POST"
    probes: int
    recall_at_10: float
    avg_ms: float
    avg_reads: float
    avg_index_reads: float
    avg_table_reads: float
    avg_expansions: float


def sintetico(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Mezcla de 64 gaussianas: estructura de clusters como la de embeddings reales."""
    rng = np.random.default_rng(seed)
    centros = rng.normal(size=(64, dim)) * 3
    return (centros[rng.integers(64, size=n)] + rng.normal(size=(n, dim))).astype(np.float32)


def _vec(v) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in v) + "]"


def _cargar(ex: Executor, X: np.ndarray) -> None:
    n, dim = X.shape
    ex.execute(f"CREATE TABLE items (id INT PRIMARY KEY, precio FLOAT, emb VECTOR({dim})) USING HEAP")
    sf = ex.catalog.open_table("items")
    # precio = permutacion de 0..n-1: una ventana [0, s*n) selecciona exactamente s.
    precios = np.random.default_rng(1).permutation(n)
    sf.bulk_load((i, float(precios[i]), tuple(map(float, X[i]))) for i in range(n))
    sf.flush()


def _medir(ex: Executor, sql: str) -> tuple[list[int], dict]:
    r = ex.execute(sql)
    io = r.plan.detail.get("io", {})
    return [row[0] for row in r.rows], {
        "ms": r.exec_ms, "reads": r.disk_reads,
        "index": io.get("index_reads", 0), "table": io.get("table_reads", 0),
        "expansions": r.plan.detail.get("expansions", 0), "access": r.plan.access,
    }


def _resumen(ms, reads, idx, tab):
    return (statistics.mean(ms), statistics.pstdev(ms), statistics.mean(reads),
            statistics.mean(idx), statistics.mean(tab))


def run(n: int = 20_000, dim: int = 128, n_queries: int = 100, npy: str | None = None,
        seed: int = 0) -> tuple[list[AnnRow], list[HybridRow]]:
    X = np.load(npy).astype(np.float32) if npy else sintetico(n, dim, seed)
    n, dim = X.shape
    rng = np.random.default_rng(seed + 1)
    Q = X[rng.integers(n, size=n_queries)] + rng.normal(size=(n_queries, dim)).astype(np.float32) * 0.3

    tmp = tempfile.mkdtemp(prefix="exp8_")
    cat = Catalog(tmp, page_size=4096, cache_size=0)
    ex = Executor(cat)
    try:
        _cargar(ex, X)
        print(f"  tabla cargada: N={n}, D={dim}", flush=True)

        # ---- verdad de campo: k-NN exacto
        verdad, ms, rd, ir, tr = [], [], [], [], []
        for q in Q:
            ids, m = _medir(ex, f"SELECT id FROM items ORDER BY emb <-> '{_vec(q)}' LIMIT {K}")
            assert m["access"] == "KnnSeqScan"
            verdad.append(set(ids))
            ms.append(m["ms"]); rd.append(m["reads"]); ir.append(m["index"]); tr.append(m["table"])
        ann: list[AnnRow] = [AnnRow("exact", 0, n, dim, 0, 1.0, *_resumen(ms, rd, ir, tr))]
        print(f"  exacto: {ann[0].avg_ms:.1f} ms, {ann[0].avg_reads:.0f} lecturas", flush=True)

        ex.execute("CREATE INDEX iv ON items (emb) USING IVF WITH (metric = 'L2')")
        n_lists = cat.open_index("items", "iv").n_lists

        for p in [p for p in PROBES if p <= n_lists]:
            ex.execute(f"SET ivf.probes = {p}")
            rec, ms, rd, ir, tr = [], [], [], [], []
            for q, gt in zip(Q, verdad):
                ids, m = _medir(ex, f"SELECT id FROM items ORDER BY emb <-> '{_vec(q)}' LIMIT {K}")
                rec.append(len(gt & set(ids)) / K)
                ms.append(m["ms"]); rd.append(m["reads"]); ir.append(m["index"]); tr.append(m["table"])
            fila = AnnRow("ivf", p, n, dim, n_lists, statistics.mean(rec), *_resumen(ms, rd, ir, tr))
            ann.append(fila)
            print(f"  probes={p:>3}: recall@10={fila.recall_at_10:.3f}  {fila.avg_ms:.1f} ms  "
                  f"{fila.avg_reads:.0f} lecturas", flush=True)

        # ---- 8b: planificador hibrido
        ex.execute("CREATE INDEX ip ON items (precio) USING BTREE")
        ex.execute("SET ivf.probes = 5")
        hib: list[HybridRow] = []
        Qh = Q[: max(10, n_queries // 5)]
        for s in SELECTIVIDADES:
            tope = max(1, int(s * n))
            where = f"precio < {tope}"
            mask = np.zeros(n, dtype=bool)
            precios = np.random.default_rng(1).permutation(n)
            mask[precios < tope] = True
            verdad_f = []
            for q in Qh:
                d = np.where(mask, np.linalg.norm(X - q, axis=1), np.inf)
                verdad_f.append({int(i) for i in np.argsort(d)[:K] if np.isfinite(d[i])})
            for estrategia in ("PRE", "POST"):
                ex.execute(f"SET hybrid.strategy = '{estrategia}'")
                rec, ms, rd, ir, tr, exp = [], [], [], [], [], []
                for q, gt in zip(Qh, verdad_f):
                    ids, m = _medir(ex, f"SELECT id FROM items WHERE {where} "
                                        f"ORDER BY emb <-> '{_vec(q)}' LIMIT {K}")
                    rec.append(len(gt & set(ids)) / max(1, len(gt)))
                    ms.append(m["ms"]); rd.append(m["reads"]); ir.append(m["index"])
                    tr.append(m["table"]); exp.append(m["expansions"])
                fila = HybridRow(s, estrategia, 5, statistics.mean(rec), statistics.mean(ms),
                                 statistics.mean(rd), statistics.mean(ir), statistics.mean(tr),
                                 statistics.mean(exp))
                hib.append(fila)
                print(f"  sel={s:>6.1%} {estrategia:<4}: recall={fila.recall_at_10:.3f} "
                      f"{fila.avg_ms:.1f} ms  {fila.avg_reads:.0f} lecturas "
                      f"(indice {fila.avg_index_reads:.0f} / tabla {fila.avg_table_reads:.0f})", flush=True)
        return ann, hib
    finally:
        cat.close_all()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    from backend.benchmarks.run_all import _volcar

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--npy", default=None, help="archivo .npy (N, D) con embeddings reales")
    args = ap.parse_args()
    ann, hib = run(args.n, args.dim, args.queries, args.npy)
    _volcar("exp8_ann", ann)
    _volcar("exp8_hibrido", hib)


if __name__ == "__main__":
    main()
