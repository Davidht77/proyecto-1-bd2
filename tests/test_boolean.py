"""Mezclas booleanas O(n+m) y parser de consultas MATCH (enunciado 5.2.3)."""

import random

import pytest

from backend.text import boolean
from backend.text.boolean import And, BooleanQueryError, Not, Or, Term


@pytest.mark.parametrize("seed", range(20))
def test_mezclas_coinciden_con_operaciones_de_conjuntos(seed):
    rng = random.Random(seed)
    a = sorted(rng.sample(range(200), rng.randint(0, 60)))
    b = sorted(rng.sample(range(200), rng.randint(0, 60)))
    assert boolean.intersect(a, b) == sorted(set(a) & set(b))
    assert boolean.union(a, b) == sorted(set(a) | set(b))
    assert boolean.difference(a, b) == sorted(set(a) - set(b))


def test_mezclas_funcionan_con_rids_como_tuplas():
    a = [(1, 0), (1, 3), (2, 1)]
    b = [(1, 3), (2, 0), (2, 1)]
    assert boolean.intersect(a, b) == [(1, 3), (2, 1)]
    assert boolean.difference(a, b) == [(1, 0)]


def test_parser_precedencia_y_and_implicito():
    assert boolean.parse("a AND b OR c") == Or((And((Term("a"), Term("b"))), Term("c")))
    assert boolean.parse("a b AND NOT c") == And((Term("a"), Term("b"), Not(Term("c"))))
    assert boolean.parse("(a OR b) AND NOT c") == And((Or((Term("a"), Term("b"))), Not(Term("c"))))


def test_es_booleana_solo_con_operadores_en_mayusculas():
    assert boolean.is_boolean("aprendizaje AND profundo")
    assert boolean.is_boolean("(redes)")
    assert not boolean.is_boolean("redes neuronales and more")


@pytest.mark.parametrize("q", ["a AND", "(a OR b", "a OR OR b", "a )", ""])
def test_parser_rechaza_consultas_mal_formadas(q):
    with pytest.raises(BooleanQueryError):
        boolean.parse(q)


def test_evaluacion_con_listas():
    listas = {"a": [1, 2, 3, 5], "b": [2, 3, 4], "c": [3], "stop": None}
    fetch = listas.get
    ev = lambda q: boolean.evaluate(boolean.parse(q), fetch)
    assert ev("a AND b") == [2, 3]
    assert ev("a AND b AND NOT c") == [2]
    assert ev("a OR b") == [1, 2, 3, 4, 5]
    assert ev("a AND stop") == [1, 2, 3, 5]      # la stopword no filtra
    assert ev("(a OR b) AND NOT c") == [1, 2, 4, 5]


def test_not_sin_termino_positivo_falla():
    with pytest.raises(BooleanQueryError, match="positivo"):
        boolean.evaluate(boolean.parse("NOT a"), {"a": [1]}.get)
    with pytest.raises(BooleanQueryError, match="positivo"):
        boolean.evaluate(boolean.parse("a OR NOT b"), {"a": [1], "b": [2]}.get)


def test_palabras_positivas_excluyen_las_negadas():
    arbol = boolean.parse("(redes OR neuronas) AND NOT biologia")
    assert boolean.positive_words(arbol) == ["redes", "neuronas"]
    assert boolean.all_words(arbol) == ["redes", "neuronas", "biologia"]
