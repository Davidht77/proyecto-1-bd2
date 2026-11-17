"""Tienda multimedia: tablas e indices en el motor para los dos paradigmas.

Todo lo persistente vive en el motor del proyecto:

    imagenes       (id INT PK, categoria CHAR(20), archivo CHAR(64), emb VECTOR(512)) USING HEAP
    idx_emb        IVF COSINE sobre imagenes.emb                       (paradigma moderno)
    codebook_bovw  (id INT PK, centro VECTOR(128)) USING HEAP          (palabras visuales)
    <text_dir>/imagenes__bovw.*   vocabulario B+ + postings + normas   (paradigma clasico)

El indice BoVW se guarda en un subdirectorio (`data/text/`) porque el catalogo
descubre los indices relacionales por el nombre de archivo en `data/` y un
`.btree.dat` de vocabulario no debe aparecer como un B+ de la tabla.

Extraccion de caracteristicas (`extract_features`): una pasada de SIFT sobre
una muestra para entrenar el/los codebooks y una segunda pasada completa que
cuantiza cada imagen y, en paralelo, calcula el embedding ResNet-18. Las
representaciones ya calculadas se guardan en `data/cache/` como `.npy` (es
una cache de ML, no almacenamiento del gestor: borrarla solo obliga a
recalcular) para que el Experimento 7 pueda repetirse sin pagar otra vez los
minutos de CNN.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.index.hash import RID as HashRID
from backend.multimedia import bovw, datasets
from backend.multimedia.dense import ImageEmbedder
from backend.multimedia.tfidf import TfidfCosine, build as build_tfidf
from backend.storage.disk_counter import DiskCounter

TABLE = "imagenes"
CODEBOOK_TABLE = "codebook_bovw"
DENSE_INDEX = "idx_emb"
BOVW_NAME = "bovw"
CAT_LEN, FILE_LEN = 20, 64


# ------------------------------------------------------------------ features


@dataclass
class Features:
    items: list[datasets.ImageItem]
    tokens: dict[int, list[list[str]]]      # K -> lista de palabras visuales por imagen
    dense: np.ndarray                        # (N, 512)
    sift_ms_per_image: float                 # extraccion + cuantizacion
    dense_ms_per_image: float                # forward de ResNet-18 (por lote)
    codebooks: dict[int, bovw.Codebook] = field(default_factory=dict)
    codebook_fit_s: dict[int, float] = field(default_factory=dict)


def _cache_dir(root: str | Path, tag: str) -> Path:
    d = Path(root).parent / "cache" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_features(root: str | Path, items: list[datasets.ImageItem], ks: list[int],
                     cache_tag: str | None = None, sample_images: int = 1500, progress=None,
                     sift: bovw.SiftExtractor | None = None, embedder: ImageEmbedder | None = None,
                     seed: int = 0) -> Features:
    """Palabras visuales (por cada K) y embeddings densos de todas las imagenes."""
    base = datasets.categories_dir(root)
    rutas = [str(it.path(base)) for it in items]
    cache = _cache_dir(root, cache_tag) if cache_tag else None
    sift = sift or bovw.SiftExtractor()
    embedder = embedder or ImageEmbedder()
    log = progress or (lambda *a: None)

    # -- codebooks: K-Means sobre una muestra estratificada de descriptores
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rutas))[: min(sample_images, len(rutas))]
    codebooks: dict[int, bovw.Codebook] = {}
    fit_s: dict[int, float] = {}
    pendientes = [k for k in ks if not (cache and (cache / f"codebook_{k}.npy").exists())]
    if pendientes:
        log("sift-sample", 0, len(idx))
        muestra, _ = bovw.sample_descriptors(sift, [rutas[i] for i in idx], per_image=80, max_total=250_000, seed=seed)
        for k in pendientes:
            t0 = time.perf_counter()
            codebooks[k] = bovw.Codebook.fit(muestra, k, seed=seed)
            fit_s[k] = time.perf_counter() - t0
            if cache:
                np.save(cache / f"codebook_{k}.npy", codebooks[k].centers)
            log("kmeans", k, 0)
    for k in ks:
        if k not in codebooks:
            codebooks[k] = bovw.Codebook(np.load(cache / f"codebook_{k}.npy"))

    # -- pasada completa: SIFT + cuantizacion (todas las K a la vez)
    tokens: dict[int, list[list[str]]] = {k: [] for k in ks}
    faltan = [k for k in ks if not (cache and (cache / f"tokens_{k}.json").exists())]
    sift_ms = 0.0
    if faltan:
        t_sift = 0.0
        for i, ruta in enumerate(rutas):
            t0 = time.perf_counter()
            d = sift(ruta)
            for k in faltan:
                tokens[k].append(codebooks[k].tokens(d))
            t_sift += time.perf_counter() - t0
            if i % 200 == 0:
                log("sift", i, len(rutas))
        sift_ms = 1000.0 * t_sift / max(1, len(rutas))
        if cache:
            for k in faltan:
                (cache / f"tokens_{k}.json").write_text(json.dumps(tokens[k]), encoding="utf-8")
            (cache / "sift_ms.json").write_text(json.dumps(sift_ms))
    for k in ks:
        if k not in faltan:
            tokens[k] = json.loads((cache / f"tokens_{k}.json").read_text(encoding="utf-8"))
    if not faltan and cache and (cache / "sift_ms.json").exists():
        sift_ms = json.loads((cache / "sift_ms.json").read_text())

    # -- embeddings densos
    if cache and (cache / "dense.npy").exists():
        dense = np.load(cache / "dense.npy")
        dense_ms = json.loads((cache / "dense_ms.json").read_text()) if (cache / "dense_ms.json").exists() else 0.0
    else:
        t0 = time.perf_counter()
        dense = embedder.embed(rutas, progress=lambda a, b: log("resnet", a, b))
        dense_ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(rutas))
        if cache:
            np.save(cache / "dense.npy", dense)
            (cache / "dense_ms.json").write_text(json.dumps(dense_ms))
    return Features(items, tokens, dense, sift_ms, dense_ms, codebooks, fit_s)


# --------------------------------------------------------------------- store


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


class MultimediaStore:
    """Construye y consulta la tienda de imagenes sobre un Catalog/Executor."""

    def __init__(self, catalog: Catalog, executor: Executor, dataset_root: str | Path,
                 text_dir: str | Path | None = None):
        self.catalog = catalog
        self.executor = executor
        self.root = Path(dataset_root)
        self.text_dir = Path(text_dir) if text_dir else Path(catalog.data_dir) / "text"
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self._sift = bovw.SiftExtractor()
        self._embedder = ImageEmbedder()
        self._codebook: bovw.Codebook | None = None
        self._tfidf: TfidfCosine | None = None
        self.progress: dict = {"building": False, "stage": None, "done": 0, "total": 0, "error": None}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- estado

    @property
    def bovw_base(self) -> str:
        return str(self.text_dir / f"{TABLE}__{BOVW_NAME}")

    def is_ready(self) -> bool:
        return (self.catalog.exists(TABLE) and self.catalog.exists(CODEBOOK_TABLE)
                and Path(f"{self.bovw_base}.norms.dat").exists()
                and any(i["name"] == DENSE_INDEX for i in self.catalog.indexes_of(TABLE)))

    def status(self) -> dict:
        listo = self.is_ready()
        out = {"ready": listo, "dataset_ready": datasets.is_ready(self.root), **self.progress}
        if listo:
            info = self.catalog.describe(TABLE)
            out.update({
                "n_images": info.n_records, "table_pages": info.page_count,
                "dense_index": next(i for i in info.indexes if i["name"] == DENSE_INDEX),
                "k_words": self.codebook().k,
                "bovw": self.tfidf().stats(),
            })
        return out

    def _set(self, stage, done=0, total=0):
        self.progress.update({"stage": stage, "done": int(done), "total": int(total)})

    # ------------------------------------------------------------ build

    def build(self, k_words: int = 256, max_per_category: int | None = None, lists: int | None = None,
              features: Features | None = None) -> dict:
        """Construye todo desde el dataset. Bloqueante; `build_async` lo envuelve."""
        with self._lock:
            self.progress.update({"building": True, "error": None})
            try:
                return self._build(k_words, max_per_category, lists, features)
            except Exception as exc:  # noqa: BLE001 - se reporta en /status
                self.progress["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.progress["building"] = False

    def build_async(self, **kw) -> threading.Thread:
        t = threading.Thread(target=lambda: self.build(**kw), daemon=True)
        t.start()
        return t

    def _build(self, k_words, max_per_category, lists, features) -> dict:
        t_total = time.perf_counter()
        self._set("dataset")
        if not datasets.is_ready(self.root):
            datasets.download(self.root)
        items = datasets.list_images(self.root, max_per_category=max_per_category)
        if features is None:
            tag = f"caltech101_all" if not max_per_category else f"caltech101_max{max_per_category}"
            features = extract_features(self.root, items, [k_words], cache_tag=tag,
                                        progress=self._set, sift=self._sift, embedder=self._embedder)
        self._set("tabla")
        self.drop()
        ex = self.executor
        ex.execute(f"CREATE TABLE {TABLE} (id INT PRIMARY KEY, categoria CHAR({CAT_LEN}), "
                   f"archivo CHAR({FILE_LEN}), emb VECTOR({features.dense.shape[1]})) USING HEAP")
        sf = self.catalog.open_table(TABLE)
        sf.bulk_load((i, it.category[:CAT_LEN], it.relpath[:FILE_LEN], tuple(map(float, features.dense[i])))
                     for i, it in enumerate(items))
        sf.flush()

        self._set("ivf")
        t0 = time.perf_counter()
        extra = f", lists = {int(lists)}" if lists else ""
        ex.execute(f"CREATE INDEX {DENSE_INDEX} ON {TABLE} (emb) USING IVF WITH (metric = 'COSINE'{extra})")
        ivf_s = time.perf_counter() - t0

        self._set("codebook")
        cb = features.codebooks[k_words]
        ex.execute(f"CREATE TABLE {CODEBOOK_TABLE} (id INT PRIMARY KEY, centro VECTOR({cb.centers.shape[1]})) USING HEAP")
        cbt = self.catalog.open_table(CODEBOOK_TABLE)
        cbt.bulk_load((i, tuple(map(float, c))) for i, c in enumerate(cb.centers))
        cbt.flush()
        self._codebook = cb

        self._set("bovw")
        t0 = time.perf_counter()
        docs = [(rid.page_id, rid.slot, features.tokens[k_words][row[0]]) for rid, row in sf.scan_with_rid()]
        counter = DiskCounter()
        st = build_tfidf(docs, self.bovw_base, tokenize=bovw.identity_tokens, counter=counter)
        bovw_s = time.perf_counter() - t0
        self._tfidf = None
        self._set("listo")
        return {
            "n_images": len(items), "k_words": k_words, "ivf_build_s": round(ivf_s, 2),
            "bovw_build_s": round(bovw_s, 2), "bovw_build_writes": counter.disk_writes,
            "sift_ms_per_image": round(features.sift_ms_per_image, 2),
            "dense_ms_per_image": round(features.dense_ms_per_image, 2),
            "codebook_fit_s": {k: round(v, 1) for k, v in features.codebook_fit_s.items()},
            "total_s": round(time.perf_counter() - t_total, 1), **st,
        }

    def drop(self) -> None:
        self._tfidf = None
        self._codebook = None
        for t in (TABLE, CODEBOOK_TABLE):
            if self.catalog.exists(t):
                self.catalog.drop_table(t)
        for p in self.text_dir.glob(f"{TABLE}__{BOVW_NAME}.*"):
            p.unlink()

    # ---------------------------------------------------------- acceso

    def codebook(self) -> bovw.Codebook:
        if self._codebook is None:
            filas = sorted(self.catalog.open_table(CODEBOOK_TABLE).scan(), key=lambda r: r[0])
            self._codebook = bovw.Codebook(np.asarray([r[1] for r in filas], dtype=np.float32))
        return self._codebook

    def tfidf(self) -> TfidfCosine:
        if self._tfidf is None:
            self._tfidf = TfidfCosine(self.bovw_base, page_size=self.catalog.page_size,
                                      cache_size=self.catalog.cache_size, tokenize_query=bovw.identity_tokens)
        return self._tfidf

    def image_path(self, image_id: int) -> Path | None:
        row = self.catalog.open_table(TABLE).search(int(image_id))
        if row is None:
            return None
        return datasets.categories_dir(self.root) / row[2]

    def sample(self, n: int = 12, seed: int | None = None) -> list[dict]:
        """Imagenes al azar del dataset indexado, para usarlas como consulta en la demo."""
        sf = self.catalog.open_table(TABLE)
        rng = np.random.default_rng(seed)
        ids = rng.choice(sf.n_records, size=min(n, sf.n_records), replace=False)
        out = []
        for i in sorted(int(x) for x in ids):
            row = sf.search(i)
            if row:
                out.append({"id": row[0], "categoria": row[1], "archivo": row[2]})
        return out

    # -------------------------------------------------------- consultas

    def search_dense(self, vec, k: int = 10, probes: int | None = None, exact: bool = False) -> dict:
        """Top-k por ResNet: IVF (`IVFScan`) o, con exact=True, k-NN lineal."""
        ex = self.executor
        if probes is not None:
            ex.execute(f"SET ivf.probes = {int(probes)}")
        sql = (f"SELECT id, categoria, archivo, emb <=> '{_vec(vec)}' AS dist FROM {TABLE} "
               f"ORDER BY dist ASC LIMIT {int(k)}")
        # exact: SET vector.scan = 'EXACT' hace que el planificador ignore el IVF y
        # resuelva el k-NN exacto (KnnSeqScan) con la misma metrica coseno.
        ex.execute(f"SET vector.scan = '{'EXACT' if exact else 'AUTO'}'")
        try:
            r = ex.execute(sql)
        finally:
            ex.execute("SET vector.scan = 'AUTO'")
        return {
            "hits": [{"id": row[0], "categoria": row[1], "archivo": row[2], "score": row[3]} for row in r.rows],
            "metrics": _metrics(r),
        }

    def search_bovw(self, tokens: list[str], k: int = 10) -> dict:
        idx = self.tfidf()
        t0 = time.perf_counter()
        idx.counter.reset()
        sf = self.catalog.open_table(TABLE)
        sf.counter.reset()
        top = idx.search(None, k=k, tokens=tokens)
        index_reads = idx.counter.disk_reads
        hits = []
        for (page, slot), score in top:
            row = self.catalog.row_at(TABLE, HashRID(page, slot, 0))
            if row is not None:
                hits.append({"id": row[0], "categoria": row[1], "archivo": row[2], "score": round(score, 6)})
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "hits": hits,
            "metrics": {
                "access": "InvertedIndex", "exec_ms": round(ms, 3),
                "disk_reads": index_reads + sf.counter.disk_reads, "index_reads": index_reads,
                "table_reads": sf.counter.disk_reads, "terms": len(set(tokens)),
            },
        }

    def search_image(self, image, k: int = 10, probes: int = 5) -> dict:
        """Consulta por los dos paradigmas a partir de una ruta o de los bytes de una imagen."""
        t0 = time.perf_counter()
        desc = self._sift(image)
        tokens = self.codebook().tokens(desc)
        sift_ms = (time.perf_counter() - t0) * 1000.0
        vec, dense_ms = self._embedder.time_one(image)
        clasico = self.search_bovw(tokens, k)
        clasico["extract_ms"] = round(sift_ms, 2)
        clasico["n_descriptors"] = int(len(desc))
        moderno = self.search_dense(vec, k, probes=probes)
        moderno["extract_ms"] = round(dense_ms, 2)
        return {"bovw": clasico, "dense": moderno, "k": k, "probes": probes}


def _metrics(r) -> dict:
    io = r.plan.detail.get("io", {})
    return {
        "access": r.plan.access, "exec_ms": round(r.exec_ms, 3), "disk_reads": r.disk_reads,
        "index_reads": io.get("index_reads", 0), "table_reads": io.get("table_reads", 0),
        **{k: r.plan.detail[k] for k in ("probes", "n_lists", "candidates_checked") if k in r.plan.detail},
    }


def main() -> None:
    """python -m backend.multimedia.service [--k 256] [--max-per-category N] [--data data]"""
    import argparse

    ap = argparse.ArgumentParser(description="Construye la tienda multimedia (Caltech-101: BoVW + ResNet/IVF)")
    ap.add_argument("--k", type=int, default=256, help="palabras visuales del codebook")
    ap.add_argument("--max-per-category", type=int, default=None)
    ap.add_argument("--lists", type=int, default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--datasets", default=None, help="por defecto <data>/datasets")
    args = ap.parse_args()
    cat = Catalog(args.data, page_size=4096, cache_size=0)
    store = MultimediaStore(cat, Executor(cat), args.datasets or str(Path(args.data) / "datasets"))

    def log(stage, done=0, total=0):
        print(f"  [{stage}] {done}/{total}" if total else f"  [{stage}] {done}", flush=True)

    store._set = log  # progreso por consola
    try:
        print(json.dumps(store.build(k_words=args.k, max_per_category=args.max_per_category, lists=args.lists), indent=1))
    finally:
        cat.close_all()


if __name__ == "__main__":
    main()
