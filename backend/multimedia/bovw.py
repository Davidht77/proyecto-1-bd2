"""Bag of Visual Words: SIFT -> K-Means -> histograma -> indice invertido.

Paradigma clasico del enunciado 5.3.1. Cada imagen produce unos cientos de
descriptores SIFT de 128 dimensiones; un codebook de K centroides (K-Means
sobre una muestra de descriptores) cuantiza cada descriptor a la "palabra
visual" mas cercana, y la imagen queda representada por la bolsa de palabras
`["vw0012", "vw0012", "vw0377", ...]`. Esa lista entra al mismo SPIMI que los
documentos de texto (con un tokenizador identidad: las palabras visuales no se
stemean ni se filtran) y se consulta con TF-IDF + coseno (tfidf.py).

El codebook se guarda en el motor como una tabla `(id INT, centro VECTOR(128))`
para no depender de un serializador externo.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

SIFT_DIM = 128
MAX_SIDE = 320


def load_gray(path_or_bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Imagen en escala de grises, reescalada para que el lado mayor sea max_side."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = np.frombuffer(bytes(path_or_bytes), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"no se pudo decodificar la imagen {path_or_bytes!r}"[:120])
    h, w = img.shape[:2]
    escala = max_side / max(h, w)
    if escala < 1.0:
        img = cv2.resize(img, (max(1, int(w * escala)), max(1, int(h * escala))), interpolation=cv2.INTER_AREA)
    return img


class SiftExtractor:
    def __init__(self, nfeatures: int = 400, max_side: int = MAX_SIDE):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.max_side = max_side

    def __call__(self, path_or_bytes) -> np.ndarray:
        """Descriptores (n, 128) float32; (0, 128) si la imagen no tiene puntos clave."""
        img = load_gray(path_or_bytes, self.max_side)
        _, desc = self.sift.detectAndCompute(img, None)
        if desc is None:
            return np.zeros((0, SIFT_DIM), dtype=np.float32)
        return desc.astype(np.float32)


def word(c: int) -> str:
    return f"vw{int(c):04d}"


def identity_tokens(tokens) -> list[str]:
    """Tokenizador para SPIMI: las palabras visuales ya vienen listas."""
    return list(tokens)


class Codebook:
    """K palabras visuales. `fit` entrena K-Means sobre una muestra de descriptores."""

    def __init__(self, centers: np.ndarray | None = None):
        self.centers = None if centers is None else np.asarray(centers, dtype=np.float32)

    @property
    def k(self) -> int:
        return 0 if self.centers is None else len(self.centers)

    @classmethod
    def fit(cls, descriptors: np.ndarray, k: int, seed: int = 0, batch: int = 4096) -> "Codebook":
        X = np.asarray(descriptors, dtype=np.float32)
        k = max(1, min(k, len(X)))
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch, n_init=3, max_iter=100)
        km.fit(X)
        return cls(km.cluster_centers_)

    def assign(self, desc: np.ndarray) -> np.ndarray:
        """Indice del centroide mas cercano a cada descriptor (por lotes, L2)."""
        if len(desc) == 0:
            return np.zeros(0, dtype=np.int64)
        C = self.centers
        cc = np.einsum("ij,ij->i", C, C)
        out = np.empty(len(desc), dtype=np.int64)
        for i in range(0, len(desc), 2048):
            X = desc[i : i + 2048]
            d = cc[None, :] - 2.0 * (X @ C.T)  # ||x||^2 es constante por fila: no cambia el argmin
            out[i : i + 2048] = np.argmin(d, axis=1)
        return out

    def tokens(self, desc: np.ndarray) -> list[str]:
        return [word(c) for c in self.assign(desc)]

    def histogram(self, desc: np.ndarray) -> np.ndarray:
        h = np.bincount(self.assign(desc), minlength=self.k).astype(np.float32)
        return h


def sample_descriptors(extractor: SiftExtractor, paths, per_image: int = 60,
                       max_total: int = 200_000, seed: int = 0) -> tuple[np.ndarray, float]:
    """Muestra de descriptores para entrenar el codebook. Devuelve (muestra, ms por imagen)."""
    rng = np.random.default_rng(seed)
    lotes, t_total, n = [], 0.0, 0
    for p in paths:
        t0 = time.perf_counter()
        d = extractor(p)
        t_total += time.perf_counter() - t0
        n += 1
        if len(d) > per_image:
            d = d[rng.choice(len(d), per_image, replace=False)]
        lotes.append(d)
        if sum(len(x) for x in lotes) >= max_total:
            break
    X = np.concatenate(lotes) if lotes else np.zeros((0, SIFT_DIM), dtype=np.float32)
    return X, (1000.0 * t_total / n if n else 0.0)
