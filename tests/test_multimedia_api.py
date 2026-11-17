"""Tienda multimedia (BoVW vs. embeddings) y sus endpoints REST.

Se arma un Caltech-101 en miniatura con imagenes sinteticas de dos "categorias"
(rectangulos y circulos) y se reemplaza ResNet-18 por un embedder de
histograma de color, para que el test corra sin modelos ni red.
"""

import importlib
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.multimedia import datasets
from backend.multimedia.service import MultimediaStore


class FakeImageEmbedder:
    """Histograma de color 3x8 bins normalizado: distingue las dos categorias sinteticas."""

    dim = 24

    @staticmethod
    def _img(item):
        if isinstance(item, (bytes, bytearray)):
            return cv2.imdecode(np.frombuffer(bytes(item), np.uint8), cv2.IMREAD_COLOR)
        return cv2.imread(str(item), cv2.IMREAD_COLOR)

    def embed(self, items, progress=None):
        out = []
        for it in items:
            img = self._img(it)
            h = np.concatenate([np.histogram(img[..., c], bins=8, range=(0, 256))[0] for c in range(3)]).astype(np.float32)
            out.append(h / (np.linalg.norm(h) or 1.0))
        return np.asarray(out, dtype=np.float32)

    def embed_one(self, item):
        return self.embed([item])[0]

    def time_one(self, item):
        return self.embed_one(item), 0.5


def _imagen(rng, forma):
    img = np.full((160, 160, 3), 255, np.uint8)
    for _ in range(14):
        x, y = (int(v) for v in rng.integers(10, 130, 2))
        if forma == "rect":
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))),
                          (int(rng.integers(150, 255)), 40, 40), -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 16)), (40, 40, int(rng.integers(150, 255))), -1)
    return img


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets"
    base = datasets.categories_dir(root)
    rng = np.random.default_rng(3)
    for forma in ("rect", "circ"):
        (base / forma).mkdir(parents=True)
        for i in range(8):
            cv2.imwrite(str(base / forma / f"image_{i:04d}.jpg"), _imagen(rng, forma))
    (base / "BACKGROUND_Google").mkdir()
    cv2.imwrite(str(base / "BACKGROUND_Google" / "x.jpg"), _imagen(rng, "rect"))
    return root


def test_list_images_excluye_el_fondo_y_ordena(dataset):
    items = datasets.list_images(dataset)
    assert len(items) == 16 and {i.category for i in items} == {"rect", "circ"}
    assert items[0].relpath == "circ/image_0000.jpg"
    assert datasets.list_images(dataset, max_per_category=3).__len__() == 6


@pytest.fixture
def store(tmp_path, dataset):
    cat = Catalog(str(tmp_path / "data"), page_size=4096)
    ex = Executor(cat)
    s = MultimediaStore(cat, ex, dataset)
    s._embedder = FakeImageEmbedder()
    yield s
    s._tfidf = None
    cat.close_all()


def test_build_crea_tablas_indices_y_codebook(store):
    info = store.build(k_words=16, lists=4)
    assert info["n_images"] == 16 and info["k_words"] == 16
    st = store.status()
    assert st["ready"] and st["n_images"] == 16 and st["k_words"] == 16
    assert st["dense_index"]["kind"] == "IVF" and st["bovw"]["n_docs"] == 16
    assert store.catalog.exists("codebook_bovw") and store.codebook().centers.shape == (16, 128)
    assert [i["kind"] for i in store.catalog.indexes_of("imagenes")] == ["IVF"]


def test_busqueda_por_imagen_devuelve_ambos_paradigmas_con_io(store):
    store.build(k_words=16, lists=4)
    ruta = store.image_path(0)
    r = store.search_image(str(ruta), k=5, probes=2)
    for clave in ("bovw", "dense"):
        assert len(r[clave]["hits"]) == 5
        m = r[clave]["metrics"]
        assert m["disk_reads"] == m["index_reads"] + m["table_reads"] >= 1
        assert r[clave]["extract_ms"] >= 0
    assert r["bovw"]["hits"][0]["id"] == 0 and r["bovw"]["metrics"]["access"] == "InvertedIndex"
    assert r["dense"]["hits"][0]["id"] == 0 and r["dense"]["metrics"]["access"] == "IVFScan"
    e = store.search_dense(store._embedder.embed_one(str(ruta)), k=3, exact=True)
    assert e["metrics"]["access"] == "KnnSeqScan" and e["hits"][0]["id"] == 0


# -------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path, dataset, monkeypatch):
    monkeypatch.setenv("BD2_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BD2_DATASETS_DIR", str(dataset))
    import backend.api.main as main

    importlib.reload(main)
    main.multimedia._embedder = FakeImageEmbedder()
    with TestClient(main.app) as c:
        yield c
    main.multimedia._tfidf = None
    main.catalog.close_all()


def test_api_estado_inicial_y_errores_sin_tienda(client):
    st = client.get("/api/multimedia/status").json()
    assert st["ready"] is False and st["dataset_ready"] is True
    assert client.get("/api/multimedia/samples").status_code == 409
    assert client.post("/api/multimedia/search", data={"k": "3"}).status_code == 409
    assert client.get("/api/health").json()["implemented"]["multimedia"] is False
    assert client.get("/api/rag/status").json()["ready"] is False
    assert client.post("/api/rag/ask", json={"question": "x"}).status_code == 409


def test_api_build_sincrono_y_busqueda_multipart(client):
    import backend.api.main as main

    main.multimedia.build(k_words=16, lists=4)  # sin hilo, para el test
    assert client.get("/api/multimedia/status").json()["ready"] is True
    muestras = client.get("/api/multimedia/samples?n=4&seed=1").json()["samples"]
    assert len(muestras) == 4 and {"id", "categoria", "archivo"} <= set(muestras[0])

    img = client.get(f"/api/multimedia/image/{muestras[0]['id']}")
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")

    # consulta por id: la propia imagen se excluye del ranking
    # probes = todas las celdas: con 16 imagenes en 4 celdas, sondear menos
    # dejaria al IVF sin candidatos suficientes para llenar el top-k
    r = client.post("/api/multimedia/search", data={"image_id": str(muestras[0]["id"]), "k": "5", "probes": "4"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"]["kind"] == "indexed" and body["k"] == 5
    assert all(h["id"] != muestras[0]["id"] for h in body["bovw"]["hits"] + body["dense"]["hits"])
    assert len(body["dense"]["hits"]) == 5

    # consulta subiendo un archivo
    archivo = io.BytesIO(img.content)
    r = client.post("/api/multimedia/search", data={"k": "3"}, files={"file": ("q.jpg", archivo, "image/jpeg")})
    assert r.status_code == 200 and r.json()["query"]["kind"] == "upload"
    assert r.json()["bovw"]["hits"][0]["id"] == muestras[0]["id"]

    r = client.post("/api/multimedia/search", data={"k": "3"}, files={"file": ("q.jpg", io.BytesIO(b"no es imagen"), "image/jpeg")})
    assert r.status_code == 400
    assert client.get("/api/multimedia/image/99999").status_code == 404
