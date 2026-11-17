"""Tests del parser para el sublenguaje espacial."""

import pytest

from backend.engine.errors import SQLSyntaxError
from backend.engine.parser import CreateIndex, CreateTable, Insert, Select, SpatialPredicate, parse
from backend.storage.schema import ColumnType


def test_create_table_con_point():
    st = parse("CREATE TABLE locales (id INT PRIMARY KEY, ubicacion POINT)")
    assert isinstance(st, CreateTable)
    assert st.columns[1].type is ColumnType.POINT
    assert st.columns[1].size == 16


def test_insert_con_literal_point():
    st = parse("INSERT INTO locales VALUES (1, POINT(-77.0282, -12.1354))")
    assert isinstance(st, Insert)
    assert st.values[1] == pytest.approx((-77.0282, -12.1354))


def test_within_radius():
    st = parse(
        "SELECT * FROM locales WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 2.5)"
    )
    assert isinstance(st, Select)
    assert st.where is None
    assert isinstance(st.spatial, SpatialPredicate)
    assert st.spatial.column == "ubicacion"
    assert st.spatial.kind == "radius"
    assert st.spatial.center == pytest.approx((-77.0282, -12.1354))
    assert st.spatial.radius_km == pytest.approx(2.5)


def test_knn():
    st = parse("SELECT id FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 10)")
    assert st.spatial.kind == "knn"
    assert st.spatial.k == 10


def test_create_index_rtree():
    st = parse("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")
    assert isinstance(st, CreateIndex)
    assert (st.name, st.table, st.column, st.kind) == (
        "idx_geo",
        "locales",
        "ubicacion",
        "RTREE",
    )


def test_knn_con_k_invalido_falla():
    with pytest.raises(SQLSyntaxError, match="k entero positivo"):
        parse("SELECT * FROM locales WHERE ubicacion KNN ((-77.0, -12.0), 0)")


def test_within_radius_con_radio_negativo_falla():
    with pytest.raises(SQLSyntaxError, match="radio"):
        parse("SELECT * FROM locales WHERE ubicacion WITHIN_RADIUS ((-77.0, -12.0), -1)")


def test_point_mal_formado_falla():
    with pytest.raises(SQLSyntaxError):
        parse("INSERT INTO locales VALUES (1, POINT(-77.0))")
