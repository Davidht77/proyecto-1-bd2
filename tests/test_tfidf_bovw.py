"""TF-IDF + coseno sobre el indice SPIMI y Bag of Visual Words (modulo multimedia)."""

import math
from collections import Counter

import cv2
import numpy as np
import pytest

from backend.multimedia import bovw, tfidf
from backend.text.preprocess import preprocess, preprocess_query

DOCS = [
    (1, 0, "Neural networks learn graph representations"),
    (1, 1, "Graph networks for data"),
    (1, 2, "Las redes neuronales aprenden de los datos"),
    (2, 0, "Data data model"),
    (2, 1, "Model of the graph"),
    (2, 2, "Redes de grafos y modelos"),
]


def coseno_en_memoria(query, k=3):
    """Oraculo: TF-IDF logaritmico + coseno con todo en RAM."""
    N = len(DOCS)
    tfs = [Counter(preprocess(t)) for _, _, t in DOCS]
    df = Counter(t for d in tfs for t in d)
    w = lambda tf, t: (1 + math.log10(tf)) * math.log10(N / df[t])
    normas = [math.sqrt(sum(w(tf, t) ** 2 for t, tf in d.items())) for d in tfs]
    q = Counter(t for t in preprocess_query(query) if t in df)
    wq = {t: w(tf, t) for t, tf in q.items()}
    qn = math.sqrt(sum(v * v for v in wq.values())) or 1
    sc = {}
    for i, d in enumerate(tfs):
        s = sum(wq[t] * w(d[t], t) for t in wq if t in d)
        if s:
            sc[(DOCS[i][0], DOCS[i][1])] = s / (qn * normas[i])
    return sorted(sc.items(), key=lambda e: -e[1])[:k]


def test_tfidf_coincide_con_el_oraculo_en_memoria(tmp_path):
    base = str(tmp_path / "t__idx")
    st = tfidf.build(DOCS, base)
    assert st["n_docs"] == 6
    with tfidf.TfidfCosine(base) as ix:
        for q in ("graph networks", "datos", "model"):
            got = ix.search(q, k=3)
            esp = coseno_en_memoria(q)
            assert [r for r, _ in got] == [r for r, _ in esp], q
            assert all(abs(a - b) < 1e-9 for (_, a), (_, b) in zip(got, esp))


def test_tfidf_guarda_normas_en_disco_y_las_relee(tmp_path):
    base = str(tmp_path / "t__idx")
    tfidf.build(DOCS, base)
    assert (tmp_path / "t__idx.norms.dat").exists()
    with tfidf.TfidfCosine(base) as ix:
        assert set(ix.norms) == {(p, s) for p, s, _ in DOCS}
        assert all(v > 0 for v in ix.norms.values())
        assert ix.stats()["norm_pages"] == 1


def test_tfidf_solo_lee_las_listas_de_la_consulta(tmp_path):
    base = str(tmp_path / "t__idx")
    tfidf.build(DOCS, base)
    with tfidf.TfidfCosine(base) as ix:
        ix.counter.reset()
        assert ix.search("palabrainexistente", k=3) == []
        sin = ix.counter.disk_reads  # solo el descenso por el B+ del vocabulario
        ix.counter.reset()
        ix.search("graph", k=3)
        assert ix.counter.disk_reads >= sin  # vocab + al menos la pagina de la lista


def test_tokenizador_alternativo_evita_el_stemming(tmp_path):
    docs = [(1, 0, ["vw0001", "vw0002", "vw0002"]), (1, 1, ["vw0003"])]
    base = str(tmp_path / "v__idx")
    tfidf.build(docs, base, tokenize=bovw.identity_tokens)
    with tfidf.TfidfCosine(base, tokenize_query=bovw.identity_tokens) as ix:
        assert ix.vocab.search("vw0002") is not None
        top = ix.search(None, k=2, tokens=["vw0002"])
        assert top and top[0][0] == (1, 0)


# ------------------------------------------------------------------- BoVW


def _imagen(rng, forma):
    img = np.full((160, 160, 3), 255, np.uint8)
    for _ in range(12):
        x, y = (int(v) for v in rng.integers(10, 130, 2))
        color = tuple(int(v) for v in rng.integers(0, 200, 3))
        if forma == "rect":
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 30)), y + int(rng.integers(8, 30))), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(5, 16)), color, -1)
    return img


@pytest.fixture
def imagenes(tmp_path):
    rng = np.random.default_rng(1)
    rutas = []
    for forma in ("rect", "circ"):
        for i in range(5):
            p = tmp_path / f"{forma}{i}.png"
            cv2.imwrite(str(p), _imagen(rng, forma))
            rutas.append((forma, str(p)))
    return rutas


def test_sift_devuelve_descriptores_de_128(imagenes):
    sift = bovw.SiftExtractor()
    d = sift(imagenes[0][1])
    assert d.ndim == 2 and d.shape[1] == 128 and len(d) > 0
    assert sift(open(imagenes[0][1], "rb").read()).shape == d.shape  # tambien acepta bytes


def test_codebook_cuantiza_y_tokeniza(imagenes):
    sift = bovw.SiftExtractor()
    muestra, ms = bovw.sample_descriptors(sift, [p for _, p in imagenes], per_image=40)
    cb = bovw.Codebook.fit(muestra, 8)
    assert cb.k == 8 and ms > 0
    toks = cb.tokens(sift(imagenes[0][1]))
    assert toks and all(t.startswith("vw") and len(t) == 6 for t in toks)
    h = cb.histogram(sift(imagenes[0][1]))
    assert h.shape == (8,) and h.sum() == len(toks)


def test_bovw_recupera_la_misma_imagen_primero(imagenes, tmp_path):
    sift = bovw.SiftExtractor()
    descs = [sift(p) for _, p in imagenes]
    cb = bovw.Codebook.fit(np.concatenate(descs), 16)
    docs = [(1, i, cb.tokens(d)) for i, d in enumerate(descs)]
    base = str(tmp_path / "img__bovw")
    tfidf.build(docs, base, tokenize=bovw.identity_tokens)
    with tfidf.TfidfCosine(base, tokenize_query=bovw.identity_tokens) as ix:
        for i in (0, 7):
            top = ix.search(None, k=3, tokens=cb.tokens(descs[i]))
            assert top[0][0] == (1, i)
            assert abs(top[0][1] - 1.0) < 1e-6
