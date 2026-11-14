"""Metricas de distancia vectorial (enunciado 5.3.2).

    L2      ||x - q||_2                     operador SQL  <->
    COSINE  1 - (x . q) / (||x|| ||q||)     operador SQL  <=>

Todo opera por lotes con numpy: el costo dominante de una busqueda por
similitud es la aritmetica O(N*D), no el I/O, y hacerla registro a registro
en Python puro haria que la latencia midiera el interprete y no el indice.
"""

from __future__ import annotations

import numpy as np

METRICS = ("L2", "COSINE")
METRIC_CODE = {"L2": 0, "COSINE": 1}
CODE_METRIC = {v: k for k, v in METRIC_CODE.items()}
OPERATOR_METRIC = {"<->": "L2", "<=>": "COSINE"}


def distances(q, X: np.ndarray, metric: str) -> np.ndarray:
    """Distancia de `q` a cada fila de `X`. Devuelve float64 de largo len(X)."""
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if metric == "L2":
        diff = X - q
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))
    if metric == "COSINE":
        qn = np.linalg.norm(q)
        xn = np.linalg.norm(X, axis=1)
        denom = xn * qn
        sim = np.divide(X @ q, denom, out=np.zeros(len(X)), where=denom > 0)
        # El vector nulo no tiene direccion: se le asigna similitud 0 (distancia 1).
        return 1.0 - sim
    raise ValueError(f"metrica desconocida: {metric}. Usa L2 o COSINE.")


def distance(q, x, metric: str) -> float:
    return float(distances(q, np.asarray([x]), metric)[0])
