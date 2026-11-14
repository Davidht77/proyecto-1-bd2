"""K-Means (Lloyd) con inicializacion k-means++ para el cuantizador del IVF.

Con metrica COSINE se usa K-Means esferico: los puntos y los centroides se
normalizan a norma 1 en cada iteracion, de modo que la celda de Voronoi se
define por angulo y no por magnitud, que es lo que compara el operador <=>.

Las asignaciones se calculan por bloques: con N = 100k y C = 316 la matriz de
distancias completa ocuparia 250 MB en float64.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 8192


def _normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Indice del centroide mas cercano (L2) para cada fila de X."""
    out = np.empty(len(X), dtype=np.int64)
    c2 = np.einsum("ij,ij->i", centroids, centroids)
    for s in range(0, len(X), _CHUNK):
        B = X[s : s + _CHUNK]
        # ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 no cambia el argmin.
        d = c2[None, :] - 2.0 * (B @ centroids.T)
        out[s : s + _CHUNK] = np.argmin(d, axis=1)
    return out


def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    centroids[0] = X[rng.integers(len(X))]
    d2 = np.einsum("ij,ij->i", X - centroids[0], X - centroids[0])
    for i in range(1, k):
        total = d2.sum()
        idx = rng.integers(len(X)) if total <= 0 else rng.choice(len(X), p=d2 / total)
        centroids[i] = X[idx]
        diff = X - centroids[i]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))
    return centroids


def kmeans(
    vectors,
    k: int,
    metric: str = "L2",
    n_iter: int = 20,
    seed: int = 0,
    max_train: int = 256,
) -> np.ndarray:
    """Entrena k centroides. Devuelve un arreglo (k, D) float32.

    Se entrena sobre una muestra de a lo sumo `max_train * k` puntos (la misma
    regla que usa FAISS): mas puntos casi no mueven los centroides y el costo
    de entrenamiento crece lineal con N.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if len(X) == 0:
        raise ValueError("no se puede entrenar K-Means sin vectores")
    k = max(1, min(k, len(X)))
    rng = np.random.default_rng(seed)
    if len(X) > max_train * k:
        X = X[rng.choice(len(X), size=max_train * k, replace=False)]
    spherical = metric == "COSINE"
    if spherical:
        X = _normalize(X)

    centroids = _kmeans_pp(X, k, rng)
    for _ in range(n_iter):
        labels = assign(X, centroids)
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, X)
        counts = np.bincount(labels, minlength=k)
        vacios = counts == 0
        nuevos = sums / np.maximum(counts, 1)[:, None]
        # Celda vacia: se reubica en un punto al azar en vez de dejarla muerta.
        if vacios.any():
            nuevos[vacios] = X[rng.integers(len(X), size=int(vacios.sum()))]
        if spherical:
            nuevos = _normalize(nuevos)
        if np.allclose(nuevos, centroids):
            centroids = nuevos
            break
        centroids = nuevos
    return centroids.astype(np.float32)
