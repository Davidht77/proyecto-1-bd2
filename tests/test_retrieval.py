"""Validacion de la recuperacion sobre la muestra de Wikipedia (500 es + 500 en).

Consulta = titulo de un articulo; acierto = ese articulo sale en el top-10
(busqueda de item conocido: no hace falta etiquetar relevancia a mano).
`bm25_oraculo` recorre los 1000 documentos en RAM: es la respuesta de
referencia que el indice en disco (R2-05) tiene que reproducir.
"""

import csv
import math
from collections import Counter
from pathlib import Path

from backend.text.preprocess import preprocess, preprocess_query

with open(Path(__file__).parent / "fixtures" / "wiki_muestra.csv", encoding="utf-8") as f:
    ROWS = list(csv.DictReader(f))
DOCS = [Counter(preprocess(r["text"])) for r in ROWS]
LENS = [sum(d.values()) for d in DOCS]
AVGDL = sum(LENS) / len(LENS)
DF = Counter(t for d in DOCS for t in d)


def bm25_oraculo(query: str, k: int = 10, k1: float = 1.2, b: float = 0.75) -> list[int]:
    scores = Counter()
    for t in preprocess_query(query):
        if t not in DF:
            continue
        idf = math.log(1 + (len(DOCS) - DF[t] + 0.5) / (DF[t] + 0.5))
        for i, d in enumerate(DOCS):
            if t in d:
                scores[i] += idf * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * LENS[i] / AVGDL))
    return [i for i, _ in scores.most_common(k)]


def acierto_top10(lang: str) -> float:
    idx = [i for i, r in enumerate(ROWS) if r["lang"] == lang]
    return sum(i in bm25_oraculo(ROWS[i]["title"]) for i in idx) / len(idx)


def test_muestra_tiene_500_por_idioma_y_cabe_en_char_1000():
    assert Counter(r["lang"] for r in ROWS) == {"es": 500, "en": 500}
    assert max(len(r["text"].encode("utf-8")) for r in ROWS) <= 1000


def test_titulo_en_espanol_encuentra_su_articulo():
    assert acierto_top10("es") >= 0.9  # medido: 0.99


def test_titulo_en_ingles_encuentra_su_articulo():
    assert acierto_top10("en") >= 0.9  # medido: 0.94


def test_consulta_libre_pone_primero_al_articulo_exacto():
    assert ROWS[bm25_oraculo("agujero negro")[0]]["title"] == "Agujero negro"


def test_consulta_en_espanol_encuentra_documentos_en_ingles():
    assert any(ROWS[i]["lang"] == "en" for i in bm25_oraculo("economía", k=3))
