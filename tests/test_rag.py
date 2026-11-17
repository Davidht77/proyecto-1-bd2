"""Aplicacion RAG: cliente LLM, corpus en el motor y pipeline lexico vs semantico.

El embedder SBERT se sustituye por uno determinista en RAM (hash de palabras)
para que los tests no descarguen modelos ni dependan de la red.
"""

import csv
import hashlib

import numpy as np
import pytest

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.rag import llm as llm_mod
from backend.rag.corpus import Corpus
from backend.rag.pipeline import RagService, build_prompt


class FakeTextEmbedder:
    """Bolsa de palabras proyectada por hashing a 32-D y normalizada."""

    dim = 32

    def embed(self, texts, progress=None):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 0, n, 1.0)

    def embed_one(self, text):
        return self.embed([text])[0]


@pytest.fixture
def corpus_csv(tmp_path):
    filas = [
        ("es", 1, "Agujero negro", "Un agujero negro es una region del espacio con una gravedad tan intensa que nada escapa."),
        ("es", 2, "Andorra", "Andorra es un microestado soberano de Europa situado entre Espana y Francia."),
        ("es", 3, "Fotosintesis", "La fotosintesis convierte la energia de la luz en energia quimica en las plantas."),
        ("en", 4, "Black hole", "A black hole is a region of spacetime where gravity is so strong that nothing can escape."),
        ("en", 5, "Photosynthesis", "Photosynthesis is the process plants use to convert light energy into chemical energy."),
        ("en", 6, "Ada Lovelace", "Ada Lovelace was an English mathematician known for her work on the Analytical Engine."),
    ] * 4  # repetido para que el IVF tenga mas de una celda
    path = tmp_path / "wiki.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "id", "title", "text"])
        for i, (lang, _, t, x) in enumerate(filas):
            w.writerow([lang, i, t, x])
    return str(path)


@pytest.fixture
def rag(tmp_path, corpus_csv):
    cat = Catalog(str(tmp_path / "data"), page_size=4096)
    ex = Executor(cat)
    corp = Corpus(cat, ex, embedder=FakeTextEmbedder())
    corp.build(corpus_csv)
    svc = RagService(cat, ex, corpus=corp, llm=llm_mod.LLMClient(provider="extractive"))
    yield svc
    svc.invalidate()
    cat.close_all()


# ------------------------------------------------------------------ corpus


def test_build_crea_tabla_ivf_e_indice_invertido(rag):
    st = rag.status()
    assert st["ready"] and st["n_docs"] == 24
    assert st["dense_index"]["kind"] == "IVF" and st["dense_index"]["metric"] == "COSINE"
    assert st["inverted"]["n_docs"] == 24 and st["inverted"]["postings"] > 0
    assert {"postings.inv", "vocab.dic", "norms.dat"} <= set(st["inverted_bytes"])
    # el catalogo no ve el vocabulario como un B+ de la tabla
    assert all(i["kind"] in ("IVF",) for i in rag.catalog.indexes_of("documentos"))


def test_documento_cabe_en_char_1000_y_titulo_en_80(rag):
    sf = rag.catalog.open_table("documentos")
    for row in sf.scan():
        assert len(row[2].encode()) <= 80 and len(row[3].encode()) <= 1000


# ------------------------------------------------------------- recuperacion


def test_lexico_tfidf_y_bm25_encuentran_el_articulo(rag):
    for scorer in ("tfidf", "bm25"):
        r = rag.retrieve_lexical("agujero negro gravedad", k=3, scorer=scorer)
        assert r["hits"][0]["titulo"] == "Agujero negro", scorer
        m = r["metrics"]
        assert m["access"] == "InvertedIndex" and m["index_reads"] > 0 and m["table_reads"] >= 1
        assert m["disk_reads"] == m["index_reads"] + m["table_reads"]


def test_semantico_usa_ivf_y_el_exacto_knn_seq_scan(rag):
    r = rag.retrieve_semantic("black hole gravity escape", k=3, probes=5)
    assert r["metrics"]["access"] == "IVFScan" and r["metrics"]["probes"] >= 1
    assert r["hits"][0]["titulo"] == "Black hole"
    e = rag.retrieve_semantic("black hole gravity escape", k=3, exact=True)
    assert e["metrics"]["access"] == "KnnSeqScan"
    assert {h["id"] for h in e["hits"]} and e["hits"][0]["titulo"] == "Black hole"


def test_ask_arma_prompt_con_citas_y_devuelve_ambas_rutas(rag):
    r = rag.ask("¿Qué es un agujero negro?", k=3, mode="both")
    assert r["lexical"]["hits"] and r["semantic"]["hits"]
    assert len(r["context"]) <= 3 and r["context"][0]["n"] == 1
    assert "[1]" in r["prompt"] and "Pregunta:" in r["prompt"]
    assert r["answer"]["provider"] == "extractive" and "[1]" in r["answer"]["text"]
    ids = [c["id"] for c in r["context"]]
    assert len(ids) == len(set(ids))


def test_ask_sin_generar_no_llama_al_llm(rag):
    r = rag.ask("fotosintesis", k=2, mode="lexical", generate=False)
    assert r["answer"] is None and [c["titulo"] for c in r["context"]][0].lower().startswith("fotos")


def test_build_prompt_numera_los_fragmentos():
    p = build_prompt("q", [{"lang": "es", "titulo": "T", "texto": "x"}, {"lang": "en", "titulo": "U", "texto": "y"}])
    assert "[1] (es) T: x" in p and "[2] (en) U: y" in p


# ------------------------------------------------------------------- LLM


def test_detect_provider_por_variables_de_entorno():
    assert llm_mod.detect_provider({}) == "extractive"
    assert llm_mod.detect_provider({"ANTHROPIC_API_KEY": "k"}) == "anthropic"
    assert llm_mod.detect_provider({"GEMINI_API_KEY": "k"}) == "gemini"
    assert llm_mod.detect_provider({"LLM_PROVIDER": "ollama", "ANTHROPIC_API_KEY": "k"}) == "ollama"
    c = llm_mod.LLMClient(env={"GEMINI_API_KEY": "k"})
    assert c.provider == "gemini" and c.model == "gemini-2.5-flash"
    assert llm_mod.LLMClient(env={"LLM_PROVIDER": "openai", "LLM_MODEL": "x"}).model == "x"


def test_extractivo_responde_con_los_fragmentos():
    g = llm_mod.LLMClient(provider="extractive").generate("s", "u", fragments=["uno", "dos"])
    assert g.provider == "extractive" and "[1] uno" in g.text and "[2] dos" in g.text and g.ms >= 0


def test_openai_compatible_arma_la_peticion(monkeypatch):
    capturado = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"hola [1]"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}'

    def fake_urlopen(req, timeout=0):
        capturado["url"] = req.full_url
        capturado["auth"] = req.get_header("Authorization")
        capturado["body"] = req.data
        return Resp()

    import json

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_mod.json, "load", lambda r: json.loads(r.read()))
    c = llm_mod.LLMClient(env={"GEMINI_API_KEY": "clave"})
    g = c.generate("sys", "user")
    assert g.text == "hola [1]" and g.input_tokens == 5 and g.output_tokens == 2
    assert capturado["url"].startswith("https://generativelanguage.googleapis.com") and capturado["auth"] == "Bearer clave"
    assert json.loads(capturado["body"])["model"] == "gemini-2.5-flash"
