"""Corpus del RAG dentro del motor.

    documentos  (id INT PK, lang CHAR(2), titulo CHAR(80), texto CHAR(1000),
                 emb VECTOR(384)) USING HEAP
    idx_emb     IVF COSINE sobre documentos.emb                (recuperacion semantica)
    <text_dir>/documentos__texto.*   vocabulario B+ + postings + normas
                                     (recuperacion lexica: TF-IDF coseno y BM25)

El texto de cada articulo son sus primeros 1000 bytes (CHAR(1000)): es el
"fragmento" que el RAG cita. El embedding SBERT se calcula sobre
"titulo. texto" para que el titulo pese en la representacion.

El indice lexico es el SPIMI del modulo textual (Parte 3) con su pipeline de
preprocesamiento bilingue; sobre los mismos archivos corren BM25
(backend/text/ranking.py) y TF-IDF + coseno (backend/multimedia/tfidf.py).
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.multimedia.dense import TextEmbedder
from backend.multimedia.tfidf import build as build_tfidf
from backend.storage.disk_counter import DiskCounter

TABLE = "documentos"
DENSE_INDEX = "idx_emb"
TEXT_NAME = "texto"
TITLE_LEN, TEXT_LEN = 80, 1000


def _cut(s: str, n: int) -> str:
    return s.encode("utf-8")[:n].decode("utf-8", "ignore")


def read_csv(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    if limit:
        # mitad y mitad por idioma, para no quedarse solo con el primero
        por_lang: dict[str, list] = {}
        for r in filas:
            por_lang.setdefault(r["lang"], []).append(r)
        k = len(por_lang) or 1
        filas = [r for grupo in por_lang.values() for r in grupo[: max(1, limit // k)]]
    return filas


class Corpus:
    def __init__(self, catalog: Catalog, executor: Executor, text_dir: str | Path | None = None,
                 embedder: TextEmbedder | None = None):
        self.catalog = catalog
        self.executor = executor
        self.text_dir = Path(text_dir) if text_dir else Path(catalog.data_dir) / "text"
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or TextEmbedder()
        self.progress: dict = {"building": False, "stage": None, "done": 0, "total": 0, "error": None}
        self._lock = threading.Lock()

    @property
    def text_base(self) -> str:
        return str(self.text_dir / f"{TABLE}__{TEXT_NAME}")

    def is_ready(self) -> bool:
        return (self.catalog.exists(TABLE) and Path(f"{self.text_base}.norms.dat").exists()
                and any(i["name"] == DENSE_INDEX for i in self.catalog.indexes_of(TABLE)))

    def status(self) -> dict:
        out = {"ready": self.is_ready(), **self.progress}
        if out["ready"]:
            info = self.catalog.describe(TABLE)
            out.update({"n_docs": info.n_records, "table_pages": info.page_count,
                        "dense_index": next(i for i in info.indexes if i["name"] == DENSE_INDEX)})
        return out

    def _set(self, stage, done=0, total=0):
        self.progress.update({"stage": stage, "done": int(done), "total": int(total)})

    # ---------------------------------------------------------------- build

    def build(self, csv_path: str | Path, limit: int | None = None, lists: int | None = None) -> dict:
        with self._lock:
            self.progress.update({"building": True, "error": None})
            try:
                return self._build(csv_path, limit, lists)
            except Exception as exc:  # noqa: BLE001
                self.progress["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.progress["building"] = False

    def build_async(self, **kw) -> threading.Thread:
        t = threading.Thread(target=lambda: self.build(**kw), daemon=True)
        t.start()
        return t

    def _build(self, csv_path, limit, lists) -> dict:
        t_total = time.perf_counter()
        self._set("csv")
        filas = read_csv(csv_path, limit)
        if not filas:
            raise ValueError(f"{csv_path} no tiene filas")

        self._set("sbert", 0, len(filas))
        t0 = time.perf_counter()
        textos = [f"{r['title']}. {r['text']}" for r in filas]
        E = self.embedder.embed(textos, progress=lambda a, b: self._set("sbert", a, b))
        embed_s = time.perf_counter() - t0

        self._set("tabla")
        self.drop()
        ex = self.executor
        ex.execute(f"CREATE TABLE {TABLE} (id INT PRIMARY KEY, lang CHAR(2), titulo CHAR({TITLE_LEN}), "
                   f"texto CHAR({TEXT_LEN}), emb VECTOR({E.shape[1]})) USING HEAP")
        sf = self.catalog.open_table(TABLE)
        sf.bulk_load((i, r["lang"][:2], _cut(r["title"], TITLE_LEN), _cut(r["text"], TEXT_LEN),
                      tuple(map(float, E[i]))) for i, r in enumerate(filas))
        sf.flush()

        self._set("ivf")
        t0 = time.perf_counter()
        extra = f", lists = {int(lists)}" if lists else ""
        ex.execute(f"CREATE INDEX {DENSE_INDEX} ON {TABLE} (emb) USING IVF WITH (metric = 'COSINE'{extra})")
        ivf_s = time.perf_counter() - t0

        self._set("spimi")
        t0 = time.perf_counter()
        docs = [(rid.page_id, rid.slot, f"{row[2]} {row[3]}") for rid, row in sf.scan_with_rid()]
        counter = DiskCounter()
        st = build_tfidf(docs, self.text_base, counter=counter)
        spimi_s = time.perf_counter() - t0
        self._set("listo")
        return {"n_docs": len(filas), "embed_s": round(embed_s, 1), "ivf_build_s": round(ivf_s, 1),
                "inverted_build_s": round(spimi_s, 1), "inverted_build_writes": counter.disk_writes,
                "total_s": round(time.perf_counter() - t_total, 1), **st}

    def drop(self) -> None:
        if self.catalog.exists(TABLE):
            self.catalog.drop_table(TABLE)
        for p in self.text_dir.glob(f"{TABLE}__{TEXT_NAME}.*"):
            p.unlink()

    def index_size_bytes(self) -> dict:
        tam = {p.name.split(".", 1)[1]: p.stat().st_size for p in self.text_dir.glob(f"{TABLE}__{TEXT_NAME}.*")}
        return tam


def main() -> None:
    """python -m backend.rag.corpus [--csv data/datasets/wiki_15k.csv] [--limit N] [--data data]"""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Carga el corpus RAG en el motor (tabla + IVF + indice invertido)")
    ap.add_argument("--csv", default="data/datasets/wiki_15k.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lists", type=int, default=None)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    cat = Catalog(args.data, page_size=4096, cache_size=0)
    corp = Corpus(cat, Executor(cat))
    try:
        print(json.dumps(corp.build(args.csv, limit=args.limit, lists=args.lists), indent=1))
    finally:
        cat.close_all()


if __name__ == "__main__":
    main()
