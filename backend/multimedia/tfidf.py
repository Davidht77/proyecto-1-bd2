"""TF-IDF con similitud de coseno (CosineScore) sobre el indice invertido en disco.

Pesos del enunciado (5.2.3):

    w_{t,d} = (1 + log10 tf_{t,d}) * log10(N / df_t)
    score(q, d) = sum_t w_{t,q} * w_{t,d} / (||d|| * ||q||)

El indice es el mismo par (vocabulario B+, postings) que construye SPIMI
(backend/text/spimi.py), asi que BM25 y TF-IDF conviven sobre los mismos
archivos. Lo unico que agrega este modulo es la **norma euclidiana de cada
documento**, que el enunciado pide precalcular y guardar en disco: va en
`<base>.norms.dat`, un archivo paginado por el Pager con registros
<page, slot, norma> (el RID del documento y su ||d||).

Las normas se leen una sola vez al abrir el indice (I/O contado) y quedan en
RAM, igual que los centroides del IVF: son N floats y releerlas en cada
consulta solo sumaria ruido a la comparacion de rutas.

En una consulta solo se leen las listas de posteo de los terminos de q, se
acumulan los productos parciales en un diccionario y el top-k sale de un heap.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable, Iterable
from operator import itemgetter

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema
from backend.text import spimi
from backend.text.preprocess import preprocess, preprocess_query

NORM = Schema([Column("page", ColumnType.INT), Column("slot", ColumnType.INT),
               Column("norm", ColumnType.FLOAT)], pk_index=0)

Tokenizer = Callable[[object], list[str]]


def _write_norms(path: str, norms: Iterable[tuple[int, int, float]], page_size: int,
                 counter: DiskCounter | None) -> None:
    pager = Pager(path, record_size=NORM.record_size, schema=NORM, page_size=page_size, counter=counter)
    cap = page_capacity(page_size, NORM.record_size)
    page = None
    try:
        for i, rec in enumerate(norms):
            if i % cap == 0:
                if page is not None:
                    pager.write_page(page)
                page = pager.new_page(PageType.DATA)
            n = page.record_count
            page.write_record(n, NORM.pack(rec))
            page.set_occupied(n, True)
            page.record_count = n + 1
        if page is not None:
            pager.write_page(page)
    finally:
        pager.close()


def build(docs: list[tuple[int, int, object]], base: str, tokenize: Tokenizer = preprocess,
          budget: int = 50_000, page_size: int = 4096, counter: DiskCounter | None = None) -> dict:
    """Construye vocabulario + postings (SPIMI) y el archivo de normas.

    `docs` = [(page, slot, contenido)], donde `contenido` es lo que `tokenize`
    espera (texto para el modulo lexico, lista de palabras visuales para BoVW).
    Se materializa como lista porque hacen falta dos pasadas: una para df
    (necesario para el idf de las normas) y otra para el indice.
    """
    df: Counter = Counter()
    n_docs = 0
    for _, _, contenido in docs:
        df.update(set(tokenize(contenido)))
        n_docs += 1
    spimi.build_index(docs, base, budget=budget, page_size=page_size, counter=counter, tokenize=tokenize)

    def normas():
        for page, slot, contenido in docs:
            acc = 0.0
            for t, tf in Counter(tokenize(contenido)).items():
                idf = math.log10(n_docs / df[t]) if df[t] else 0.0
                w = (1 + math.log10(tf)) * idf
                acc += w * w
            yield page, slot, math.sqrt(acc)

    _write_norms(f"{base}.norms.dat", normas(), page_size, counter)
    return {"n_docs": n_docs, "vocab_size": len(df)}


class TfidfCosine:
    """Consulta top-k por coseno sobre un indice construido con `build`."""

    def __init__(self, base: str, page_size: int = 4096, cache_size: int = 0,
                 counter: DiskCounter | None = None, tokenize_query: Tokenizer = preprocess_query):
        self.base = base
        self.counter = counter if counter is not None else DiskCounter()
        self.tokenize_query = tokenize_query
        self.vocab, self.post = spimi.open_index(base, page_size=page_size, cache_size=cache_size,
                                                 counter=self.counter)
        self._norms = Pager(f"{base}.norms.dat", record_size=NORM.record_size, schema=NORM,
                            page_size=page_size, cache_size=cache_size, counter=self.counter)
        self.norms: dict[tuple[int, int], float] = {}
        for pid in range(1, self._norms.page_count):
            page = self._norms.read_page(pid)
            for slot in range(page.record_count):
                p, s, nrm = NORM.unpack(page.read_record(slot))
                self.norms[(p, s)] = nrm
        self.n_docs = self.post.pager.user_a

    # -------------------------------------------------------------- consulta

    def weights(self, tokens: list[str]) -> dict[str, tuple[float, int, int]]:
        """Para cada termino de la consulta presente en el vocabulario:
        (w_{t,q}, inicio de su lista, df). Cuesta solo lecturas del B+."""
        out = {}
        for t, tfq in Counter(tokens).items():
            entry = self.vocab.search(t)
            if entry is None:
                continue
            start, df = entry
            idf = math.log10(self.n_docs / df) if df else 0.0
            out[t] = ((1 + math.log10(tfq)) * idf, start, df)
        return out

    def search(self, query, k: int = 10, tokens: list[str] | None = None) -> list[tuple[tuple[int, int], float]]:
        """[((page, slot), score)] de los k documentos mas parecidos, descendente."""
        tokens = tokens if tokens is not None else self.tokenize_query(query)
        pesos = self.weights(tokens)
        if not pesos:
            return []
        qnorm = math.sqrt(sum(w * w for w, _, _ in pesos.values())) or 1.0
        scores: dict[tuple[int, int], float] = {}
        for t, (wq, start, df) in pesos.items():
            idf = math.log10(self.n_docs / df)
            for page, slot, tf, _dl in self.post.scan(start, df):
                scores[(page, slot)] = scores.get((page, slot), 0.0) + wq * (1 + math.log10(tf)) * idf
        for rid in scores:
            scores[rid] /= qnorm * (self.norms.get(rid) or 1.0)
        return heapq.nlargest(k, scores.items(), key=itemgetter(1))

    # ------------------------------------------------------------------ misc

    def stats(self) -> dict:
        return {
            "n_docs": self.n_docs,
            "vocab_pages": self.vocab.pager.page_count - 1,
            "postings_pages": self.post.pager.page_count - 1,
            "norm_pages": self._norms.page_count - 1,
            "postings": len(self.post),
        }

    def close(self) -> None:
        self.vocab.close()
        self.post.close()
        self._norms.close()

    def __enter__(self) -> "TfidfCosine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
