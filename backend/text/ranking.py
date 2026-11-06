"""Ranking BM25 top-k sobre el indice invertido en disco.

score(d) = sum por termino t de la consulta:
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

Solo se leen las listas de los terminos de la consulta: tf y dl vienen en cada
posting, y N y la suma de los largos en la cabecera del archivo de postings.
Paginacion: el heap guarda los mejores offset + k y se devuelven los ultimos k.
"""

import heapq
import math
from operator import itemgetter

from backend.index.btree import BPlusTree
from backend.text.postings import PostingsFile
from backend.text.preprocess import preprocess_query


def bm25(query: str, vocab: BPlusTree, post: PostingsFile, k: int = 10, offset: int = 0,
         k1: float = 1.2, b: float = 0.75) -> list[tuple[tuple[int, int], float]]:
    """Devuelve [((page, slot), score)] del puesto offset+1 al offset+k."""
    n_docs = post.pager.user_a
    avgdl = post.pager.user_b / n_docs if n_docs else 0
    scores: dict[tuple[int, int], float] = {}
    for t in preprocess_query(query):
        entry = vocab.search(t)
        if entry is None:
            continue
        start, df = entry
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for page, slot, tf, dl in post.scan(start, df):
            rid = (page, slot)
            scores[rid] = scores.get(rid, 0.0) + idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    return heapq.nlargest(offset + k, scores.items(), key=itemgetter(1))[offset:]
