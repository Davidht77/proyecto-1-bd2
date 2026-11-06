"""SPIMI (Single-Pass In-Memory Indexing), fase 1: bloques ordenados en disco.

Se recorren los documentos una sola vez armando un diccionario termino ->
postings en RAM. Cuando llega a `budget` postings, se ordena por termino y se
vuelca a un archivo de bloque; la RAM nunca guarda mas de `budget` postings.
Cada bloque es una secuencia de registros (termino, page, slot, tf, dl).
"""

from collections import Counter
from collections.abc import Iterable

from backend.storage.disk_counter import DiskCounter
from backend.storage.schema import Column, ColumnType, Schema
from backend.text.postings import POSTING, PostingsFile
from backend.text.preprocess import MAX_TERM_BYTES, preprocess

BLOCK = Schema([Column("term", ColumnType.CHAR, MAX_TERM_BYTES), *POSTING.columns])


def _write_block(path: str, block: dict[str, list], page_size: int, counter: DiskCounter | None) -> None:
    with PostingsFile(path, page_size=page_size, counter=counter, schema=BLOCK) as f:
        f.write((t, *p) for t in sorted(block) for p in block[t])


def invert_blocks(docs: Iterable[tuple[int, int, str]], base: str, budget: int = 50_000,
                  page_size: int = 4096, counter: DiskCounter | None = None) -> list[str]:
    """docs = (page, slot, texto) de cada fila. Devuelve las rutas de los bloques."""
    paths, block, n = [], {}, 0
    for page, slot, text in docs:
        terms = preprocess(text)
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
    return paths
