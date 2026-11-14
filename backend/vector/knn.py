"""Busqueda exacta: k-NN lineal y busqueda por rango r (enunciado 5.3.2).

Es la linea base del Experimento 8 (verdad de campo para Recall@K) y la ruta
que usa el planificador cuando no hay indice IVF o cuando el pre-filtrado
relacional ya dejo pocos candidatos.

Los k mejores se mantienen en un Max-Heap de tamano k: la raiz es el peor de
los k actuales, asi que un candidato nuevo se compara contra ella en O(1) y
entra con O(log k). heapq es un min-heap; se guarda -distancia para invertirlo.
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Iterable, Iterator

import numpy as np

from backend.vector.distance import distances

_BATCH = 1024


def _batches(items: Iterable, vec_of, size: int) -> Iterator[tuple[list, np.ndarray]]:
    lote: list = []
    for it in items:
        lote.append(it)
        if len(lote) >= size:
            yield lote, np.asarray([vec_of(x) for x in lote], dtype=np.float32)
            lote = []
    if lote:
        yield lote, np.asarray([vec_of(x) for x in lote], dtype=np.float32)


def knn_linear(items: Iterable, vec_of, q, metric: str, k: int) -> list[tuple[float, object]]:
    """Los k items mas cercanos a q, ordenados por distancia ascendente.

    `items` es cualquier iterable (filas de un scan, candidatos de un indice);
    `vec_of(item)` extrae su vector. Se procesa por lotes para vectorizar el
    calculo de distancias sin materializar la tabla completa en RAM.
    """
    if k <= 0:
        return []
    heap: list[tuple[float, int, object]] = []  # (-dist, desempate, item)
    tie = count()
    for lote, X in _batches(items, vec_of, _BATCH):
        d = distances(q, X, metric)
        for dist, item in zip(d.tolist(), lote):
            if len(heap) < k:
                heapq.heappush(heap, (-dist, next(tie), item))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, next(tie), item))
    return [(-nd, item) for nd, _, item in sorted(heap, key=lambda e: (-e[0], e[1]))]


def range_linear(items: Iterable, vec_of, q, metric: str, r: float) -> list[tuple[float, object]]:
    """Todos los items a distancia <= r, ordenados por distancia."""
    out: list[tuple[float, object]] = []
    for lote, X in _batches(items, vec_of, _BATCH):
        d = distances(q, X, metric)
        out.extend((dist, item) for dist, item in zip(d.tolist(), lote) if dist <= r)
    out.sort(key=lambda e: e[0])
    return out
