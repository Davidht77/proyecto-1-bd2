"""Tests de integracion del modulo espacial con el motor."""

import random

import pytest

from backend.engine.catalog import Catalog
from backend.engine.errors import ColumnNotFound
from backend.engine.executor import Executor
from backend.index import geo


@pytest.fixture
def db(tmp_path):
    cat = Catalog(str(tmp_path / "data"), page_size=1024)
    yield Executor(cat)
    cat.close_all()


def _puntos(n=80, seed=3):
    rnd = random.Random(seed)
    return [
        (round(-77.0282 + rnd.uniform(-0.15, 0.15), 6),
         round(-12.1354 + rnd.uniform(-0.12, 0.12), 6))
        for _ in range(n)
    ]


def _crear(db, puntos):
    db.execute(
        "CREATE TABLE locales (id INT PRIMARY KEY, nombre CHAR(20), "
        "ubicacion POINT, categoria CHAR(10)) USING HEAP"
    )
    for i, (lon, lat) in enumerate(puntos):
        db.execute(
            f"INSERT INTO locales VALUES ({i}, 'local{i}', "
            f"POINT({lon}, {lat}), 'x')"
        )


def _ids(rows):
    return sorted(row[0] for row in rows)


def test_consulta_espacial_sin_indice_es_scan(db):
    pts = _puntos()
    _crear(db, pts)
    r = db.execute(
        "SELECT id, ubicacion FROM locales "
        "WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 3.0)"
    )
    assert r.plan.access == "SpatialSeqScan"
    center = (-77.0282, -12.1354)
    want = sorted(
        i for i, (lon, lat) in enumerate(pts)
        if geo.haversine(lon, lat, center[0], center[1]) <= 3.0
    )
    assert _ids(r.rows) == want
    assert r.spatial["query"]["kind"] == "radius"


def test_rtree_sobre_sequential_sobrevive_a_splits_de_overflow(db):
    """La insercion ordenada y los splits mueven slots: el RID debe reindexarse."""
    pts = _puntos(120, seed=9)
    db.execute(
        "CREATE TABLE locales (id INT PRIMARY KEY, nombre CHAR(20), "
        "ubicacion POINT, categoria CHAR(10)) USING SEQUENTIAL"
    )
    for i, (lon, lat) in enumerate(pts):
        db.execute(
            f"INSERT INTO locales VALUES ({i}, 'local{i}', "
            f"POINT({lon}, {lat}), 'x')"
        )
        if i == 5:
            db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")

    r = db.execute(
        "SELECT id, ubicacion FROM locales "
        "WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 1000)"
    )
    assert r.plan.access == "RTreeRangeScan"
    assert _ids(r.rows) == list(range(len(pts)))


def test_indice_rtree_habilita_range_scan(db):
    pts = _puntos()
    _crear(db, pts)
    db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")
    idx = next(i for i in db.catalog.indexes_of("locales") if i["name"] == "idx_geo")
    assert idx["kind"] == "RTREE" and idx["column"] == "ubicacion"

    r = db.execute(
        "SELECT id, ubicacion FROM locales "
        "WHERE ubicacion WITHIN_RADIUS ((-77.0282, -12.1354), 3.0)"
    )
    assert r.plan.access == "RTreeRangeScan"
    center = (-77.0282, -12.1354)
    want = sorted(
        i for i, (lon, lat) in enumerate(pts)
        if geo.haversine(lon, lat, center[0], center[1]) <= 3.0
    )
    assert _ids(r.rows) == want
    assert r.index_reads > 0
    assert r.spatial["index_name"] == "idx_geo"
    assert r.spatial["visited_mbrs"]


def test_knn_rtree_ordena_por_distancia(db):
    pts = _puntos()
    _crear(db, pts)
    db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")
    r = db.execute(
        "SELECT id FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 7)"
    )
    assert r.plan.access == "RTreeKNN"
    center = (-77.0282, -12.1354)
    want = [
        i for _d, i in sorted(
            (geo.haversine(lon, lat, center[0], center[1]), i)
            for i, (lon, lat) in enumerate(pts)
        )[:7]
    ]
    assert [row[0] for row in r.rows] == want
    assert r.row_count == 7


def test_knn_sin_indice_es_scan(db):
    pts = _puntos(40)
    _crear(db, pts)
    r = db.execute(
        "SELECT id FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 5)"
    )
    assert r.plan.access == "SpatialSeqScan"
    center = (-77.0282, -12.1354)
    want = [
        i for _d, i in sorted(
            (geo.haversine(lon, lat, center[0], center[1]), i)
            for i, (lon, lat) in enumerate(pts)
        )[:5]
    ]
    assert [row[0] for row in r.rows] == want


def test_insert_mantiene_el_indice(db):
    pts = _puntos(40)
    _crear(db, pts)
    db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")
    nuevo = (-77.0281, -12.1353)
    db.execute(
        f"INSERT INTO locales VALUES (999, 'nuevo', POINT({nuevo[0]}, {nuevo[1]}), 'x')"
    )
    r = db.execute(
        "SELECT id FROM locales WHERE ubicacion KNN ((-77.0281, -12.1353), 3)"
    )
    assert 999 in [row[0] for row in r.rows]
    assert r.plan.access == "RTreeKNN"


def test_delete_reconstruye_el_indice(db):
    pts = _puntos(60)
    _crear(db, pts)
    db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")

    borrar = pts[0]
    db.execute(
        "DELETE FROM locales WHERE ubicacion "
        f"WITHIN_RADIUS (({borrar[0]}, {borrar[1]}), 0.01)"
    )
    r = db.execute(
        "SELECT id, ubicacion FROM locales WHERE ubicacion "
        "WITHIN_RADIUS ((-77.0282, -12.1354), 100.0)"
    )
    ids = _ids(r.rows)
    assert 0 not in ids
    assert len(ids) == len(pts) - 1
    assert r.plan.access == "RTreeRangeScan"


def test_indice_rtree_sobre_columna_no_point_falla(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT) USING HEAP")
    with pytest.raises(ColumnNotFound, match="POINT"):
        db.execute("CREATE INDEX i ON t (v) USING RTREE")


def test_el_indice_espacial_persiste_al_reabrir(db, tmp_path):
    pts = _puntos(50)
    _crear(db, pts)
    db.execute("CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE")
    db.catalog.close_all()

    otro = Catalog(str(tmp_path / "data"), page_size=1024)
    ex = Executor(otro)
    r = ex.execute(
        "SELECT id FROM locales WHERE ubicacion KNN ((-77.0282, -12.1354), 3)"
    )
    assert r.plan.access == "RTreeKNN"
    assert r.row_count == 3
    otro.close_all()
