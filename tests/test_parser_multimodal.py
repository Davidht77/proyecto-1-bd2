"""Gramatica SQL multimodal extendida (enunciado 5.4)."""

import pytest

from backend.engine.errors import SQLSyntaxError
from backend.engine.parser import (
    CreateIndex,
    DistanceExpr,
    ScoreExpr,
    SetOption,
    SpatialRadius,
    TextMatch,
    parse,
)
from backend.storage.schema import ColumnType


def test_create_table_con_vector():
    st = parse("CREATE TABLE m (id INT PRIMARY KEY, cat CHAR(10), emb VECTOR(128)) USING HEAP")
    col = st.columns[2]
    assert col.type is ColumnType.VECTOR and col.length == 128 and col.size == 512


def test_vector_no_puede_ser_pk():
    with pytest.raises(SQLSyntaxError, match="PRIMARY KEY"):
        parse("CREATE TABLE m (emb VECTOR(4) PRIMARY KEY, id INT)")
    with pytest.raises(SQLSyntaxError, match="PRIMARY KEY"):
        parse("CREATE TABLE m (emb VECTOR(4), id INT)")


def test_create_index_ivf_con_opciones():
    st = parse("CREATE INDEX idx_vec ON multimedia (embedding) USING IVF WITH (lists = 100, metric = 'COSINE')")
    assert isinstance(st, CreateIndex)
    assert st.kind == "IVF"
    assert st.options == {"lists": 100, "metric": "COSINE"}


@pytest.mark.parametrize("kind", ["RTREE", "INVERTED", "HNSW"])
def test_create_index_tipos_multimodales_parsean(kind):
    assert parse(f"CREATE INDEX i ON t (c) USING {kind}").kind == kind


def test_set_opcion_con_punto():
    st = parse("SET ivf.probes = 5;")
    assert isinstance(st, SetOption)
    assert (st.name, st.value) == ("ivf.probes", 5)


def test_consulta_vectorial_hibrida_del_enunciado():
    st = parse("""
        SELECT id, categoria, precio, embedding <=> '[0.12, -0.34, 0.56]' AS dist_coseno
        FROM multimedia
        WHERE categoria = 'footwear' AND precio <= 250.0
        ORDER BY dist_coseno ASC LIMIT 5;
    """)
    dist = st.columns[3]
    assert isinstance(dist, DistanceExpr)
    assert dist.metric == "COSINE" and dist.alias == "dist_coseno"
    assert dist.vector == pytest.approx((0.12, -0.34, 0.56))
    assert st.distance is dist
    assert [f.column for f in st.filters] == ["categoria", "precio"]
    assert st.filters[0].is_equality and st.filters[1].hi == 250.0
    assert st.order_by.key == "dist_coseno" and not st.order_by.descending
    assert st.limit == 5


def test_order_by_con_expresion_l2():
    st = parse("SELECT id FROM m ORDER BY emb <-> '[1, 2e-1, -3]' LIMIT 3")
    assert st.distance.metric == "L2"
    assert st.distance.vector == pytest.approx((1.0, 0.2, -3.0))


def test_dos_predicados_sobre_la_misma_columna_se_intersectan():
    st = parse("SELECT * FROM t WHERE precio >= 10 AND cat = 'a' AND precio < 50")
    assert len(st.filters) == 2
    precio = st.filters[0]
    assert (precio.lo, precio.hi, precio.hi_inclusive) == (10, 50, False)


def test_match_y_cosine_score():
    st = parse("""
        SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos
        WHERE cuerpo MATCH 'aprendizaje AND profundo AND NOT basico'
        ORDER BY relevancia DESC LIMIT 10
    """)
    assert isinstance(st.columns[2], ScoreExpr) and st.columns[2].alias == "relevancia"
    assert st.special == [TextMatch("cuerpo", "aprendizaje AND profundo AND NOT basico")]
    assert st.order_by.descending


def test_within_radius():
    st = parse("SELECT id FROM locales WHERE ubicacion WITHIN_RADIUS((-77.0282, -12.1354), 2.5)")
    assert st.special == [SpatialRadius("ubicacion", -77.0282, -12.1354, 2.5)]


def test_vector_mal_formado_falla():
    with pytest.raises(SQLSyntaxError, match="vector"):
        parse("SELECT id FROM m ORDER BY emb <-> '1, 2' LIMIT 1")
    with pytest.raises(SQLSyntaxError, match="vector entre comillas"):
        parse("SELECT id FROM m ORDER BY emb <-> 3 LIMIT 1")


def test_dos_distancias_falla():
    with pytest.raises(SQLSyntaxError, match="una expresion de distancia"):
        parse("SELECT a <-> '[1]', b <=> '[1]' FROM t")
