"""Indice invertido de una columna de texto, visto desde SQL (enunciado 5.2 y 5.4).

    CREATE INDEX idx_texto ON documentos (cuerpo) USING INVERTED
        WITH (scoring = 'TFIDF_COSINE');          -- o 'BM25'
    SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos
    WHERE cuerpo MATCH 'aprendizaje AND profundo AND NOT basico'
    ORDER BY relevancia DESC LIMIT 10;

Archivos (todos paginados por el Pager, descubiertos por nombre):

    <base>.inv.meta                 cabecera autodescriptiva: la columna indexada
                                    va como nombre de la unica columna del esquema,
                                    user_a = scoring por defecto, user_b = 1 si el
                                    indice quedo desactualizado tras un INSERT/DELETE
    <base>.inv.spimi.vocab.btree.dat  vocabulario: B+ termino -> (inicio, df)
    <base>.inv.spimi.post             postings <page, slot, tf, dl> ordenados por RID
    <base>.inv.norms.dat              ||d|| de cada documento (TF-IDF)

El vocabulario, los postings y las normas son los del modulo textual (SPIMI) y
de TF-IDF + coseno; este modulo solo los ata a una columna de una tabla y agrega
la evaluacion booleana y el score por documento que necesita el executor.

RIDs: el posting guarda (page, slot). Para el Sequential File el area de
overflow se codifica en el bit OVERFLOW_BIT de `page`, asi que un RID ordena
igual que el recorrido de la tabla (area principal y despues overflow).

Mantenimiento: SPIMI construye el indice de una vez. Un INSERT o DELETE solo lo
marca como desactualizado (escribir la cabecera cuesta una pagina) y la
siguiente consulta MATCH lo reconstruye antes de usarlo. Es el modelo de
reindexacion por lotes de los motores de busqueda; el costo queda visible en el
plan como escrituras del indice.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from backend.multimedia.tfidf import TfidfCosine
from backend.multimedia.tfidf import build as build_tfidf
from backend.storage.disk_counter import DiskCounter
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema
from backend.text import boolean
from backend.text.preprocess import preprocess, preprocess_query

SCORINGS = ("TFIDF_COSINE", "BM25")
SCORE_FUNCTION = {"COSINE_SCORE": "TFIDF_COSINE", "BM25_SCORE": "BM25"}
OVERFLOW_BIT = 1 << 30
BM25_K1, BM25_B = 1.2, 0.75

_META = ".inv.meta"
_TEXT = ".inv"
_TEXT_FILES = (".spimi.vocab.btree.dat", ".spimi.post", ".norms.dat")


def files_of(base: str) -> list[Path]:
    return [Path(base + _META)] + [Path(base + _TEXT + s) for s in _TEXT_FILES]


def encode_rid(rid) -> tuple[int, int]:
    """RID del motor -> (page, slot) del posting."""
    area = 1 if getattr(rid, "area", "main") in ("overflow", 1) else 0
    return rid.page_id | (OVERFLOW_BIT if area else 0), rid.slot


def decode_rid(page: int, slot: int) -> tuple[int, int, int]:
    """(page, slot) del posting -> (page_id, slot, area)."""
    return page & ~OVERFLOW_BIT, slot, 1 if page & OVERFLOW_BIT else 0


@dataclass
class MatchResult:
    """Lo que el indice responde a un MATCH, antes de tocar la tabla."""

    mode: str                                   # "ranked" | "boolean"
    scoring: str                                # "TFIDF_COSINE" | "BM25"
    rids: list[tuple[int, int]]                 # documentos que cumplen, por RID
    scores: dict[tuple[int, int], float]
    terms: list[str]                            # raices que suman score
    lists_read: int = 0
    postings_read: int = 0
    detail: dict = field(default_factory=dict)

    def ranked(self) -> list[tuple[tuple[int, int], float]]:
        """Documentos en orden de relevancia (empates por RID, estable)."""
        return sorted(((r, self.scores.get(r, 0.0)) for r in self.rids), key=lambda e: -e[1])


class InvertedIndex:
    def __init__(self, base: str, column: str | None = None, scoring: str = "TFIDF_COSINE",
                 page_size: int = 4096, cache_size: int = 0, counter: DiskCounter | None = None):
        self.base = base
        self.page_size = page_size
        self.cache_size = cache_size
        self.counter = counter if counter is not None else DiskCounter()
        nuevo = not Path(base + _META).exists()
        if nuevo and column is None:
            raise FileNotFoundError(f"no existe el indice invertido {base}")
        schema = Schema([Column(column, ColumnType.CHAR, 1)]) if nuevo else None
        self._meta = Pager(base + _META, record_size=1, schema=schema, page_size=page_size,
                           counter=self.counter)
        if nuevo:
            scoring = scoring.upper()
            if scoring not in SCORINGS:
                raise ValueError(f"scoring = '{scoring}' no soportado. Usa 'TFIDF_COSINE' o 'BM25'.")
            self._meta.user_a = SCORINGS.index(scoring)
            self._meta.user_b = 1  # sin construir todavia
        self.column = self._meta.schema.columns[0].name
        self._text: TfidfCosine | None = None

    # ------------------------------------------------------------- estado

    @property
    def scoring(self) -> str:
        return SCORINGS[self._meta.user_a]

    @property
    def stale(self) -> bool:
        return bool(self._meta.user_b)

    def mark_stale(self) -> None:
        if not self.stale:
            self._meta.user_b = 1
            self._meta.flush()

    # ---------------------------------------------------------- construccion

    def build(self, docs: Iterable[tuple[tuple[int, int], str]]) -> dict:
        """docs = [((page, slot), texto)] en el orden del recorrido de la tabla."""
        self._close_text()
        for p in files_of(self.base)[1:]:
            if p.exists():
                p.unlink()
        filas = [(page, slot, texto or "") for (page, slot), texto in docs]
        st = build_tfidf(filas, self.base + _TEXT, page_size=self.page_size, counter=self.counter)
        self._meta.user_b = 0
        self._meta.flush()
        return st

    def _index(self) -> TfidfCosine:
        if self._text is None:
            self._text = TfidfCosine(self.base + _TEXT, page_size=self.page_size,
                                     cache_size=self.cache_size, counter=self.counter)
        return self._text

    def stats(self) -> dict:
        if self.stale and not Path(self.base + _TEXT + ".spimi.post").exists():
            return {"scoring": self.scoring, "stale": True}
        st = self._index().stats()
        size = sum(p.stat().st_size for p in files_of(self.base) if p.exists())
        return {"scoring": self.scoring, "stale": self.stale, "size_bytes": size, **st}

    # -------------------------------------------------------------- consulta

    def match(self, query: str, scoring: str | None = None) -> MatchResult:
        """Evalua el MATCH solo con el indice: vocabulario + postings de las
        palabras de la consulta. Cada lista se lee una vez aunque la palabra
        aparezca en el filtro booleano y en el score."""
        scoring = (scoring or self.scoring).upper()
        idx = self._index()
        cache: dict[str, list[tuple]] = {}
        leidas = 0

        def postings(stem: str) -> list[tuple]:
            nonlocal leidas
            if stem not in cache:
                entry = idx.vocab.search(stem)
                if entry is None:
                    cache[stem] = []
                else:
                    start, df = entry
                    cache[stem] = idx.post.read(start, df)
                    leidas += 1
            return cache[stem]

        def lista_de(palabra: str) -> list | None:
            raices = preprocess_query(palabra)
            if not raices:
                return None  # stopword: no filtra
            out: list = []
            for r in raices:
                out = boolean.union(out, [(p[0], p[1]) for p in postings(r)])
            return out

        if boolean.is_boolean(query):
            arbol = boolean.parse(query)
            rids = boolean.evaluate(arbol, lista_de) or []
            palabras = boolean.positive_words(arbol)
            modo = "boolean"
        else:
            palabras = query.split()
            rids = []
            for p in palabras:
                rids = boolean.union(rids, lista_de(p) or [])
            modo = "ranked"

        terms = list(dict.fromkeys(r for p in palabras for r in preprocess_query(p)))
        scores = self._score(terms, set(rids), postings, scoring)
        return MatchResult(mode=modo, scoring=scoring, rids=rids, scores=scores, terms=terms,
                           lists_read=leidas, postings_read=sum(len(v) for v in cache.values()))

    def _score(self, terms, candidatos: set, postings, scoring: str) -> dict:
        idx = self._index()
        n = idx.n_docs or 1
        scores: dict[tuple[int, int], float] = {}
        if scoring == "BM25":
            avgdl = (idx.post.pager.user_b / n) if n else 1.0
            for t in terms:
                plist = postings(t)
                if not plist:
                    continue
                idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
                for page, slot, tf, dl in plist:
                    rid = (page, slot)
                    if rid in candidatos:
                        scores[rid] = scores.get(rid, 0.0) + idf * tf * (BM25_K1 + 1) / (
                            tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
            return scores
        # TF-IDF + coseno: w_tq = idf (cada raiz una vez en la consulta).
        pesos = {}
        for t in terms:
            plist = postings(t)
            if plist:
                pesos[t] = math.log10(n / len(plist))
        qnorm = math.sqrt(sum(w * w for w in pesos.values())) or 1.0
        for t, idf in pesos.items():
            for page, slot, tf, _dl in postings(t):
                rid = (page, slot)
                if rid in candidatos:
                    scores[rid] = scores.get(rid, 0.0) + idf * (1 + math.log10(tf)) * idf
        for rid in scores:
            scores[rid] /= qnorm * (idx.norms.get(rid) or 1.0)
        return scores

    # ------------------------------------------------------------------ misc

    def _close_text(self) -> None:
        if self._text is not None:
            self._text.close()
            self._text = None

    def flush(self) -> None:
        self._meta.flush()

    def close(self) -> None:
        self._close_text()
        self._meta.close()


# ------------------------------------------------- sin indice: Full Text Scan


def scan_match(docs: Iterable[tuple[object, str]], query: str, scoring: str = "TFIDF_COSINE") -> MatchResult:
    """Misma semantica que InvertedIndex.match, pero recorriendo la tabla.

    Es la linea base del Experimento 6: sin indice hay que leer todas las
    paginas, preprocesar cada documento y calcular df y normas sobre la marcha
    (el coseno necesita el idf de todos los terminos de cada documento).
    `docs` = [(rid, texto)]; el resultado usa esos mismos rids.
    """
    scoring = scoring.upper()
    bolsas: list[tuple[object, Counter]] = [(rid, Counter(preprocess(t or ""))) for rid, t in docs]
    n = len(bolsas) or 1
    df: Counter = Counter()
    for _, b in bolsas:
        df.update(b.keys())

    def lista_de(palabra: str) -> list | None:
        raices = preprocess_query(palabra)
        if not raices:
            return None
        return [i for i, (_, b) in enumerate(bolsas) if any(r in b for r in raices)]

    if boolean.is_boolean(query):
        arbol = boolean.parse(query)
        idxs = boolean.evaluate(arbol, lista_de) or []
        palabras, modo = boolean.positive_words(arbol), "boolean"
    else:
        palabras, modo = query.split(), "ranked"
        idxs = []
        for p in palabras:
            idxs = boolean.union(idxs, lista_de(p) or [])
    terms = list(dict.fromkeys(r for p in palabras for r in preprocess_query(p)))

    scores: dict = {}
    if scoring == "BM25":
        avgdl = sum(sum(b.values()) for _, b in bolsas) / n
        for i in idxs:
            rid, b = bolsas[i]
            dl = sum(b.values())
            s = 0.0
            for t in terms:
                tf = b.get(t, 0)
                if tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
            scores[rid] = s
    else:
        pesos = {t: math.log10(n / df[t]) for t in terms if df[t]}
        qnorm = math.sqrt(sum(w * w for w in pesos.values())) or 1.0
        for i in idxs:
            rid, b = bolsas[i]
            norma = math.sqrt(sum(((1 + math.log10(tf)) * math.log10(n / df[t])) ** 2 for t, tf in b.items()))
            s = sum(w * (1 + math.log10(b[t])) * w for t, w in pesos.items() if b.get(t))
            scores[rid] = s / (qnorm * (norma or 1.0))
    return MatchResult(mode=modo, scoring=scoring, rids=[bolsas[i][0] for i in idxs], scores=scores,
                       terms=terms)


def remove_files(base: str) -> None:
    for p in files_of(base):
        if p.exists():
            p.unlink()
    carpeta = Path(base).parent
    for p in carpeta.glob(Path(base).name + _TEXT + ".spimi.blk*"):
        os.remove(p)
