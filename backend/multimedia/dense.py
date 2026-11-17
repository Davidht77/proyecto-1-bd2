"""Embeddings densos preentrenados (paradigma moderno, enunciado 5.3.1).

    ImageEmbedder  ResNet-18 (ImageNet) sin la capa de clasificacion: el vector
                   de 512-D que sale del average pooling. torchvision, CPU.
    TextEmbedder   SBERT multilingue `paraphrase-multilingual-MiniLM-L12-v2`
                   (384-D), cubre espanol e ingles para el corpus de Wikipedia.

Los modelos se cargan de forma perezosa (la primera llamada) para que importar
el modulo no cueste nada y los tests que no los usan no los descarguen. Los
vectores se devuelven normalizados a norma 1: con la metrica COSINE del IVF la
norma no importa, y asi `<->` y `<=>` ordenan igual.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterable

import numpy as np

IMAGE_DIM = 512
TEXT_DIM = 384
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


class ImageEmbedder:
    dim = IMAGE_DIM

    def __init__(self, batch: int = 32):
        self.batch = batch
        self._model = None
        self._tf = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        torch.set_num_threads(max(1, torch.get_num_threads()))
        pesos = ResNet18_Weights.IMAGENET1K_V1
        m = resnet18(weights=pesos)
        m.fc = torch.nn.Identity()  # salida: 512-D del avgpool
        m.eval()
        self._model = m
        self._tf = pesos.transforms()

    @staticmethod
    def _open(item):
        from PIL import Image

        if isinstance(item, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(item))).convert("RGB")
        return Image.open(item).convert("RGB")

    def embed(self, items: Iterable, progress=None) -> np.ndarray:
        """(N, 512) float32 normalizado. `items` son rutas o bytes de imagen."""
        import torch

        self._load()
        items = list(items)
        out = []
        with torch.inference_mode():
            for i in range(0, len(items), self.batch):
                lote = torch.stack([self._tf(self._open(it)) for it in items[i : i + self.batch]])
                out.append(self._model(lote).numpy())
                if progress:
                    progress(min(i + self.batch, len(items)), len(items))
        return _l2(np.concatenate(out)) if out else np.zeros((0, IMAGE_DIM), dtype=np.float32)

    def embed_one(self, item) -> np.ndarray:
        return self.embed([item])[0]

    def time_one(self, item) -> tuple[np.ndarray, float]:
        """(vector, ms) de una sola imagen, para medir el costo de extraccion."""
        self._load()
        t0 = time.perf_counter()
        v = self.embed_one(item)
        return v, (time.perf_counter() - t0) * 1000.0


class TextEmbedder:
    dim = TEXT_DIM

    def __init__(self, model_name: str = TEXT_MODEL, batch: int = 64):
        self.model_name = model_name
        self.batch = batch
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")

    def embed(self, texts: Iterable[str], progress=None) -> np.ndarray:
        self._load()
        texts = list(texts)
        out = []
        for i in range(0, len(texts), self.batch):
            out.append(self._model.encode(texts[i : i + self.batch], convert_to_numpy=True,
                                          normalize_embeddings=True, show_progress_bar=False))
            if progress:
                progress(min(i + self.batch, len(texts)), len(texts))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, TEXT_DIM), dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
