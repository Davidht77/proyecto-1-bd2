"""Pipeline RAG (enunciado 5.5, opcion A).

    pregunta
      ├─ recuperacion lexica      indice invertido en disco: TF-IDF + coseno (o BM25)
      └─ recuperacion semantica   SBERT(pregunta) -> SQL `ORDER BY emb <=> '[...]' LIMIT k`
                                  resuelto por el planificador (IVFScan / KnnSeqScan)
      -> prompt con los k fragmentos numerados -> LLM -> respuesta con citas [n]

Cada ruta devuelve sus fragmentos con score y el desglose de I/O (bloques del
indice y de la tabla) y latencia, para mostrarlos lado a lado en el cliente.
"""

from __future__ import annotations

import time

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.index.hash import RID as HashRID
from backend.multimedia.dense import TextEmbedder
from backend.multimedia.tfidf import TfidfCosine
from backend.rag import corpus as corpus_mod
from backend.rag.llm import LLMClient
from backend.text.ranking import bm25

SYSTEM = (
    "Eres un asistente que responde preguntas usando exclusivamente los fragmentos de "
    "Wikipedia que se te entregan. Responde en el idioma de la pregunta, de forma concreta "
    "y breve. Cita las fuentes con su numero entre corchetes, por ejemplo [2], despues de "
    "cada afirmacion que se apoye en ellas. Si los fragmentos no alcanzan para responder, "
    "dilo explicitamente en vez de inventar."
)


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def build_prompt(question: str, hits: list[dict]) -> str:
    partes = [f"[{i}] ({h['lang']}) {h['titulo']}: {h['texto']}" for i, h in enumerate(hits, 1)]
    return ("Fragmentos recuperados:\n\n" + "\n\n".join(partes)
            + f"\n\nPregunta: {question}\n\nRespuesta (con citas [n]):")


class RagService:
    def __init__(self, catalog: Catalog, executor: Executor, corpus: corpus_mod.Corpus | None = None,
                 llm: LLMClient | None = None, embedder: TextEmbedder | None = None):
        self.catalog = catalog
        self.executor = executor
        self.corpus = corpus or corpus_mod.Corpus(catalog, executor, embedder=embedder)
        self.embedder = self.corpus.embedder
        self.llm = llm or LLMClient()
        self._tfidf: TfidfCosine | None = None

    # ------------------------------------------------------------- estado

    def status(self) -> dict:
        st = self.corpus.status()
        st["llm"] = self.llm.describe()
        if st["ready"]:
            st["inverted"] = self.tfidf().stats()
            st["inverted_bytes"] = self.corpus.index_size_bytes()
        return st

    def tfidf(self) -> TfidfCosine:
        if self._tfidf is None:
            self._tfidf = TfidfCosine(self.corpus.text_base, page_size=self.catalog.page_size,
                                      cache_size=self.catalog.cache_size)
        return self._tfidf

    def invalidate(self) -> None:
        if self._tfidf is not None:
            self._tfidf.close()
            self._tfidf = None

    # --------------------------------------------------------- recuperacion

    def _row(self, page: int, slot: int) -> dict | None:
        row = self.catalog.row_at(corpus_mod.TABLE, HashRID(page, slot, 0))
        if row is None:
            return None
        return {"id": row[0], "lang": row[1], "titulo": row[2], "texto": row[3]}

    def retrieve_lexical(self, question: str, k: int = 5, scorer: str = "tfidf") -> dict:
        idx = self.tfidf()
        sf = self.catalog.open_table(corpus_mod.TABLE)
        idx.counter.reset()
        sf.counter.reset()
        t0 = time.perf_counter()
        if scorer == "bm25":
            top = bm25(question, idx.vocab, idx.post, k=k)
        else:
            top = idx.search(question, k=k)
        index_reads = idx.counter.disk_reads
        hits = []
        for (page, slot), score in top:
            fila = self._row(page, slot)
            if fila:
                hits.append({**fila, "score": round(float(score), 6)})
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "route": "lexical", "scorer": scorer, "hits": hits,
            "metrics": {"access": "InvertedIndex", "exec_ms": round(ms, 3),
                        "disk_reads": index_reads + sf.counter.disk_reads,
                        "index_reads": index_reads, "table_reads": sf.counter.disk_reads},
        }

    def retrieve_semantic(self, question: str, k: int = 5, probes: int | None = 5, exact: bool = False) -> dict:
        t0 = time.perf_counter()
        q = self.embedder.embed_one(question)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        ex = self.executor
        if probes is not None:
            ex.execute(f"SET ivf.probes = {int(probes)}")
        op = "<->" if exact else "<=>"  # con L2 no hay IVF aplicable: k-NN exacto (KnnSeqScan)
        r = ex.execute(f"SELECT id, lang, titulo, texto, emb {op} '{_vec(q)}' AS dist "
                       f"FROM {corpus_mod.TABLE} ORDER BY dist ASC LIMIT {int(k)}")
        io = r.plan.detail.get("io", {})
        return {
            "route": "semantic", "hits": [
                {"id": row[0], "lang": row[1], "titulo": row[2], "texto": row[3], "score": row[4]} for row in r.rows],
            "metrics": {"access": r.plan.access, "exec_ms": round(r.exec_ms, 3), "embed_ms": round(embed_ms, 2),
                        "disk_reads": r.disk_reads, "index_reads": io.get("index_reads", 0),
                        "table_reads": io.get("table_reads", 0),
                        **{kk: r.plan.detail[kk] for kk in ("probes", "n_lists") if kk in r.plan.detail}},
        }

    # ------------------------------------------------------------------ ask

    def ask(self, question: str, k: int = 5, mode: str = "semantic", probes: int = 5,
            scorer: str = "tfidf", generate: bool = True) -> dict:
        """mode: 'semantic' | 'lexical' | 'both' (fusion por rango de ambas listas)."""
        lex = self.retrieve_lexical(question, k, scorer)
        sem = self.retrieve_semantic(question, k, probes)
        if mode == "lexical":
            contexto = lex["hits"]
        elif mode == "both":
            contexto, vistos = [], set()
            for a, b in zip(sem["hits"] + [None] * k, lex["hits"] + [None] * k):
                for h in (a, b):
                    if h and h["id"] not in vistos and len(contexto) < k:
                        vistos.add(h["id"])
                        contexto.append(h)
        else:
            contexto = sem["hits"]
        prompt = build_prompt(question, contexto)
        respuesta = None
        if generate:
            gen = self.llm.generate(SYSTEM, prompt, fragments=[f"{h['titulo']}: {h['texto']}" for h in contexto])
            respuesta = gen.as_dict()
        return {
            "question": question, "k": k, "mode": mode,
            "lexical": lex, "semantic": sem,
            "context": [{"n": i + 1, **h} for i, h in enumerate(contexto)],
            "prompt": prompt, "answer": respuesta,
        }
