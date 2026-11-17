"""Experimento 7 (enunciado 6.3): paradigma clasico (BoVW) vs. embeddings densos.

7a. Imagenes (Caltech-101). Para un conjunto de consultas de distintas
    categorias se compara:
      - BoVW: SIFT -> codebook de K palabras (K = 128, 256, 512) -> indice
        invertido en disco con TF-IDF + coseno,
      - ResNet-18 (512-D): k-NN exacto (KnnSeqScan) e IVF en disco (probes variable).
    Metricas: Precision@5 y Precision@10 (acierto = misma categoria que la
    consulta, excluyendo la propia imagen), tiempo de extraccion por imagen,
    latencia de consulta y bloques leidos (indice / tabla).

7b. Texto (Wikipedia, opcional con --text). Busqueda de item conocido: la
    consulta es el titulo de un articulo y el acierto es recuperar ese articulo.
    Compara TF-IDF + coseno y BM25 (indice invertido) contra SBERT exacto e IVF.

Las caracteristicas de las imagenes se cachean en data/cache/ (misma cache que
usa la aplicacion), asi que repetir el experimento solo paga las consultas.

Uso:
    python -m backend.benchmarks.experiment7                       # 7a completo
    python -m backend.benchmarks.experiment7 --ks 256 --n-cats 10  # version corta
    python -m backend.benchmarks.experiment7 --text --text-n 4000  # 7a + 7b
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.multimedia import bovw, datasets
from backend.multimedia.service import MultimediaStore, extract_features
from backend.multimedia.tfidf import TfidfCosine, build as build_tfidf

KS = [128, 256, 512]
PROBES = [1, 5, 20]
K_EVAL = 10


@dataclass
class ImageRow:
    method: str            # "bovw" | "resnet_exact" | "resnet_ivf"
    param: int             # K del codebook | 0 | probes
    n_images: int
    n_categories: int
    n_queries: int
    p_at_5: float
    p_at_10: float
    extract_ms_per_image: float
    avg_query_ms: float
    std_query_ms: float
    avg_reads: float
    avg_index_reads: float
    avg_table_reads: float
    build_s: float
    index_bytes: int


@dataclass
class TextRow:
    method: str            # "tfidf" | "bm25" | "sbert_exact" | "sbert_ivf"
    param: int
    n_docs: int
    n_queries: int
    hit_at_1: float
    hit_at_10: float
    mrr_at_10: float
    avg_query_ms: float
    avg_reads: float
    avg_index_reads: float
    avg_table_reads: float
    extract_ms_per_query: float


def _precision(hits: list[dict], categoria: str, n: int) -> float:
    top = hits[:n]
    return sum(h["categoria"] == categoria for h in top) / n if top else 0.0


def measure_extraction(root: str, items, codebooks: dict, n: int = 300, seed: int = 0) -> tuple[dict, float]:
    """ms por imagen de cada paradigma sobre la MISMA muestra y sin otra carga.

    Los tiempos que deja extract_features vienen de la construccion completa y
    pueden estar contaminados por otros procesos; este es el numero del informe.
    SIFT + cuantizacion se mide por K; ResNet-18 en lotes de 32, como en la carga.
    """
    from backend.multimedia.dense import ImageEmbedder

    base = datasets.categories_dir(root)
    rng = np.random.default_rng(seed)
    rutas = [str(items[int(i)].path(base)) for i in rng.choice(len(items), size=min(n, len(items)), replace=False)]
    sift = bovw.SiftExtractor()
    sift_ms = {}
    for k, cb in codebooks.items():
        t0 = time.perf_counter()
        for r in rutas:
            cb.tokens(sift(r))
        sift_ms[k] = 1000.0 * (time.perf_counter() - t0) / len(rutas)
    emb = ImageEmbedder()
    emb.embed(rutas[:8])  # carga del modelo fuera de la medicion
    t0 = time.perf_counter()
    emb.embed(rutas)
    return sift_ms, 1000.0 * (time.perf_counter() - t0) / len(rutas)


def _bytes(base: str) -> int:
    return sum(p.stat().st_size for p in Path(base).parent.glob(Path(base).name + ".*"))


# ------------------------------------------------------------------ 7a


def run_images(root: str, ks=KS, probes=PROBES, n_cats: int = 30, per_cat: int = 5,
               seed: int = 0, store_dir: str | None = None) -> list[ImageRow]:
    items = datasets.list_images(root)
    feats = extract_features(root, items, list(ks), cache_tag="caltech101_all",
                             progress=lambda s, a=0, b=0: print(f"  [{s}] {a}/{b}" if b else f"  [{s}] {a}", flush=True))
    n = len(items)
    cats = {}
    for i, it in enumerate(items):
        cats.setdefault(it.category, []).append(i)
    grandes = sorted(cats, key=lambda c: -len(cats[c]))[:n_cats]
    rng = np.random.default_rng(seed)
    consultas = [int(i) for c in grandes for i in rng.choice(cats[c], size=min(per_cat, len(cats[c])), replace=False)]
    print(f"  {n} imagenes, {len(cats)} categorias; {len(consultas)} consultas de {len(grandes)} categorias", flush=True)

    sift_ms, dense_ms = measure_extraction(root, items, feats.codebooks, seed=seed)
    print(f"  extraccion por imagen: SIFT+cuantizacion {sift_ms} ms, ResNet-18 {dense_ms:.1f} ms", flush=True)

    tmp = store_dir or tempfile.mkdtemp(prefix="exp7_")
    cat = Catalog(tmp, page_size=4096, cache_size=0)
    ex = Executor(cat)
    store = MultimediaStore(cat, ex, root)
    filas: list[ImageRow] = []
    try:
        t0 = time.perf_counter()
        info = store.build(k_words=ks[0], features=feats)
        print(f"  tienda construida en {time.perf_counter() - t0:.0f}s: {info}", flush=True)
        sf = cat.open_table("imagenes")
        rids = [rid for rid, _ in sf.scan_with_rid()]  # bulk_load conserva el orden de `items`

        # ---- BoVW por cada K (el de ks[0] ya lo construyo la tienda)
        for k in ks:
            if k == ks[0]:
                base, build_s = store.bovw_base, info["bovw_build_s"]
                idx = store.tfidf()
            else:
                base = str(store.text_dir / f"imagenes__bovw{k}")
                t0 = time.perf_counter()
                build_tfidf([(r.page_id, r.slot, feats.tokens[k][i]) for i, r in enumerate(rids)],
                            base, tokenize=bovw.identity_tokens)
                build_s = time.perf_counter() - t0
                idx = TfidfCosine(base, tokenize_query=bovw.identity_tokens)
            store._tfidf = idx
            p5, p10, ms, rd, ir, tr = [], [], [], [], [], []
            for q in consultas:
                r = store.search_bovw(feats.tokens[k][q], K_EVAL + 1)
                hits = [h for h in r["hits"] if h["id"] != q][:K_EVAL]
                p5.append(_precision(hits, items[q].category, 5)); p10.append(_precision(hits, items[q].category, 10))
                m = r["metrics"]; ms.append(m["exec_ms"]); rd.append(m["disk_reads"]); ir.append(m["index_reads"]); tr.append(m["table_reads"])
            fila = ImageRow("bovw", k, n, len(cats), len(consultas), statistics.mean(p5), statistics.mean(p10),
                            sift_ms[k], statistics.mean(ms), statistics.pstdev(ms), statistics.mean(rd),
                            statistics.mean(ir), statistics.mean(tr), round(build_s, 2), _bytes(base))
            filas.append(fila)
            print(f"  bovw K={k:<4} P@5={fila.p_at_5:.3f} P@10={fila.p_at_10:.3f} {fila.avg_query_ms:.1f} ms "
                  f"{fila.avg_reads:.0f} lecturas", flush=True)
            if k != ks[0]:
                idx.close()

        # ---- ResNet-18: exacto e IVF
        ivf_base = cat._index_base("imagenes", "idx_emb")
        for metodo, param in [("resnet_exact", 0)] + [("resnet_ivf", p) for p in probes]:
            p5, p10, ms, rd, ir, tr = [], [], [], [], [], []
            for q in consultas:
                r = store.search_dense(feats.dense[q], K_EVAL + 1, probes=param or None, exact=(metodo == "resnet_exact"))
                hits = [h for h in r["hits"] if h["id"] != q][:K_EVAL]
                p5.append(_precision(hits, items[q].category, 5)); p10.append(_precision(hits, items[q].category, 10))
                m = r["metrics"]; ms.append(m["exec_ms"]); rd.append(m["disk_reads"]); ir.append(m["index_reads"]); tr.append(m["table_reads"])
            fila = ImageRow(metodo, param, n, len(cats), len(consultas), statistics.mean(p5), statistics.mean(p10),
                            dense_ms, statistics.mean(ms), statistics.pstdev(ms), statistics.mean(rd),
                            statistics.mean(ir), statistics.mean(tr), info["ivf_build_s"] if param else 0.0,
                            _bytes(ivf_base + ".ivf") if param else 0)
            filas.append(fila)
            print(f"  {metodo:<12} {param:<4} P@5={fila.p_at_5:.3f} P@10={fila.p_at_10:.3f} {fila.avg_query_ms:.1f} ms "
                  f"{fila.avg_reads:.0f} lecturas", flush=True)
        return filas
    finally:
        store._tfidf = None
        cat.close_all()
        if not store_dir:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 7b


def run_text(csv_path: str, n_docs: int = 4000, n_queries: int = 300, probes=PROBES, seed: int = 0) -> list[TextRow]:
    from backend.rag.corpus import Corpus
    from backend.rag.pipeline import RagService
    from backend.text.preprocess import preprocess_query

    tmp = tempfile.mkdtemp(prefix="exp7t_")
    cat = Catalog(tmp, page_size=4096, cache_size=0)
    ex = Executor(cat)
    corp = Corpus(cat, ex)
    rag = RagService(cat, ex, corpus=corp)
    filas: list[TextRow] = []
    try:
        info = corp.build(csv_path, limit=n_docs)
        print(f"  corpus: {info}", flush=True)
        sf = cat.open_table("documentos")
        docs = list(sf.scan())
        rng = np.random.default_rng(seed)
        qs = [docs[int(i)] for i in rng.choice(len(docs), size=min(n_queries, len(docs)), replace=False)]

        def evaluar(metodo, param, buscar):
            h1, h10, mrr, ms, rd, ir, tr, ext = [], [], [], [], [], [], [], []
            for d in qs:
                t0 = time.perf_counter()
                r = buscar(d[2])
                ids = [h["id"] for h in r["hits"]]
                pos = ids.index(d[0]) + 1 if d[0] in ids else None
                h1.append(pos == 1); h10.append(pos is not None); mrr.append(1 / pos if pos else 0.0)
                m = r["metrics"]; ms.append(m["exec_ms"]); rd.append(m["disk_reads"]); ir.append(m["index_reads"]); tr.append(m["table_reads"])
                ext.append(m.get("embed_ms", 0.0))
            fila = TextRow(metodo, param, len(docs), len(qs), statistics.mean(h1), statistics.mean(h10), statistics.mean(mrr),
                           statistics.mean(ms), statistics.mean(rd), statistics.mean(ir), statistics.mean(tr), statistics.mean(ext))
            filas.append(fila)
            print(f"  {metodo:<11} {param:<3} hit@1={fila.hit_at_1:.3f} hit@10={fila.hit_at_10:.3f} MRR={fila.mrr_at_10:.3f} "
                  f"{fila.avg_query_ms:.1f} ms {fila.avg_reads:.0f} lecturas", flush=True)

        # costo de "extraer" la consulta lexica: el preprocesamiento
        t0 = time.perf_counter()
        for d in qs:
            preprocess_query(d[2])
        pre_ms = 1000.0 * (time.perf_counter() - t0) / len(qs)
        evaluar("tfidf", 0, lambda q: rag.retrieve_lexical(q, K_EVAL, "tfidf"))
        evaluar("bm25", 0, lambda q: rag.retrieve_lexical(q, K_EVAL, "bm25"))
        for f in filas:
            f.extract_ms_per_query = round(pre_ms, 3)
        evaluar("sbert_exact", 0, lambda q: rag.retrieve_semantic(q, K_EVAL, exact=True))
        for p in probes:
            evaluar("sbert_ivf", p, lambda q, p=p: rag.retrieve_semantic(q, K_EVAL, probes=p))
        return filas
    finally:
        rag.invalidate()
        cat.close_all()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    from backend.benchmarks.run_all import _volcar

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/datasets", help="carpeta con caltech101/")
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--probes", type=int, nargs="+", default=PROBES)
    ap.add_argument("--n-cats", type=int, default=30)
    ap.add_argument("--per-cat", type=int, default=5)
    ap.add_argument("--text", action="store_true", help="correr tambien 7b (texto)")
    ap.add_argument("--text-csv", default="data/datasets/wiki_15k.csv")
    ap.add_argument("--text-n", type=int, default=4000)
    ap.add_argument("--text-queries", type=int, default=300)
    args = ap.parse_args()
    if not datasets.is_ready(args.root):
        datasets.download(args.root)
    print("[exp7_imagenes] arrancando...", flush=True)
    _volcar("exp7_imagenes", run_images(args.root, args.ks, args.probes, args.n_cats, args.per_cat))
    if args.text:
        print("[exp7_texto] arrancando...", flush=True)
        _volcar("exp7_texto", run_text(args.text_csv, args.text_n, args.text_queries, args.probes))


if __name__ == "__main__":
    main()
