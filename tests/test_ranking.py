import math

import pytest

from backend.text.ranking import bm25
from backend.text.spimi import build_index, open_index
from tests.test_retrieval import ROWS, bm25_oraculo

TRES = [(1, 0, "cat cat dog"), (1, 1, "dog bird"), (1, 2, "fish")]


@pytest.fixture
def tres(tmp_path):
    base = str(tmp_path / "t__idx")
    build_index(TRES, base)
    vocab, post = open_index(base)
    with vocab, post:
        yield vocab, post


def test_score_coincide_con_el_calculado_a_mano(tres):
    # N=3, df(cat)=1, largos 3,2,1 -> avgdl=2; doc (1,0): tf=2, dl=3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    a_mano = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2))
    [(rid, score)] = bm25("cat", *tres)
    assert rid == (1, 0) and score == pytest.approx(a_mano)


def test_documento_con_el_termino_raro_sale_primero(tres):
    # "dog" esta en 2 docs; "bird" solo en (1,1)
    assert [rid for rid, _ in bm25("dog bird", *tres)] == [(1, 1), (1, 0)]


def test_palabra_que_no_esta_da_lista_vacia(tres):
    assert bm25("unicorn", *tres) == []


def test_paginacion_devuelve_la_continuacion(tres):
    todo = bm25("dog bird cat", *tres, k=3)
    assert bm25("dog bird cat", *tres, k=1) + bm25("dog bird cat", *tres, k=2, offset=1) == todo


@pytest.fixture(scope="module")
def wiki(tmp_path_factory):
    base = str(tmp_path_factory.mktemp("wiki") / "articulos__idx")
    build_index(((i // 100, i % 100, r["text"]) for i, r in enumerate(ROWS)), base, budget=2000)
    vocab, post = open_index(base)
    with vocab, post:
        yield ROWS, bm25_oraculo, vocab, post


def test_mismo_top10_que_el_oraculo_en_la_muestra_de_wikipedia(wiki):
    rows, oraculo, vocab, post = wiki
    consultas = [r["title"] for r in rows[::10]] + ["agujero negro", "economía", "redes neuronales"]
    for q in consultas:
        esperado = [(i // 100, i % 100) for i in oraculo(q)]
        assert [rid for rid, _ in bm25(q, vocab, post)] == esperado, q
