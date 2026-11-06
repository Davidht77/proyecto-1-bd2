"""SPIMI (Single-Pass In-Memory Indexing): construccion externa del indice invertido.

Fase 1, invert_blocks: se recorren los documentos una sola vez armando un
diccionario termino -> postings en RAM. Cuando llega a `budget` postings, se
ordena por termino y se vuelca a un archivo de bloque; la RAM nunca guarda mas
de `budget` postings. Cada bloque es una secuencia de (termino, page, slot, tf, dl).

Fase 2, merge_blocks: merge k-vias de los bloques con un heap, leyendo una
pagina por bloque a la vez. Por cada termino sale una sola lista en
`<base>.spimi.post` (listas seguidas, sin hojas vacias) y una entrada en el
vocabulario, un B+ con termino -> RID(posicion inicial de la lista, df). N y la suma de los largos
quedan en user_a / user_b del archivo de postings, para el avgdl de BM25.
"""

import heapq
import os
from collections import Counter
from collections.abc import Iterable
from itertools import groupby
from operator import itemgetter

from backend.index.btree import RID, BPlusTree
from backend.storage.disk_counter import DiskCounter
from backend.storage.schema import Column, ColumnType, Schema
from backend.text.postings import POSTING, PostingsFile
from backend.text.preprocess import MAX_TERM_BYTES, preprocess

BLOCK = Schema([Column("term", ColumnType.CHAR, MAX_TERM_BYTES), *POSTING.columns])


def _write_block(path: str, block: dict[str, list], page_size: int, counter: DiskCounter | None) -> None:
    with PostingsFile(path, page_size=page_size, counter=counter, schema=BLOCK) as f:
        f.write((t, *p) for t in sorted(block) for p in block[t])


def invert_blocks(docs: Iterable[tuple[int, int, str]], base: str, budget: int = 50_000,
                  page_size: int = 4096, counter: DiskCounter | None = None) -> tuple[list[str], int, int]:
    """docs = (page, slot, texto) de cada fila. Devuelve (rutas de bloques, N, suma de largos)."""
    paths, block, n, n_docs, total_len = [], {}, 0, 0, 0
    for page, slot, text in docs:
        terms = preprocess(text)
        n_docs += 1
        total_len += len(terms)
        for term, tf in Counter(terms).items():
            block.setdefault(term, []).append((page, slot, tf, len(terms)))
            n += 1
            if n == budget:
                paths.append(f"{base}.spimi.blk{len(paths)}")
                _write_block(paths[-1], block, page_size, counter)
                block, n = {}, 0
    if block:
        paths.append(f"{base}.spimi.blk{len(paths)}")
        _write_block(paths[-1], block, page_size, counter)
    return paths, n_docs, total_len


def open_index(base: str, page_size: int = 4096, cache_size: int = 0,
               counter: DiskCounter | None = None) -> tuple[BPlusTree, PostingsFile]:
    vocab = BPlusTree(f"{base}.spimi.vocab", BLOCK.columns[0], page_size=page_size,
                      cache_size=cache_size, counter=counter)
    post = PostingsFile(f"{base}.spimi.post", page_size=page_size, cache_size=cache_size, counter=counter)
    return vocab, post


def merge_blocks(paths: list[str], base: str, n_docs: int, total_len: int,
                 page_size: int = 4096, counter: DiskCounter | None = None) -> None:
    blocks = [PostingsFile(p, page_size=page_size, counter=counter, schema=BLOCK) for p in paths]
    # cache_size=1: cada pagina nueva se escribe una sola vez, al salir del pool.
    vocab, post = open_index(base, page_size=page_size, cache_size=1, counter=counter)
    with vocab, post:
        # heapq.merge es estable: a igual termino sale primero el bloque anterior,
        # asi cada lista queda en orden de documento sin reordenarla.
        merged = heapq.merge(*(b.scan(0, len(b)) for b in blocks), key=itemgetter(0))
        for term, group in groupby(merged, key=itemgetter(0)):
            plist = [r[1:] for r in group]
            vocab.insert(term, RID(post.write(plist), len(plist)))
        post.pager.user_a, post.pager.user_b = n_docs, total_len
    for b, path in zip(blocks, paths):
        b.close()
        os.remove(path)


def build_index(docs: Iterable[tuple[int, int, str]], base: str, budget: int = 50_000,
                page_size: int = 4096, counter: DiskCounter | None = None) -> None:
    paths, n_docs, total_len = invert_blocks(docs, base, budget, page_size, counter)
    merge_blocks(paths, base, n_docs, total_len, page_size, counter)
