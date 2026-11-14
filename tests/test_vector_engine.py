"""Busqueda vectorial ANN y planificador hibrido (enunciado 5.3 y 5.4)."""

import numpy as np
import pytest

from backend.engine.catalog import Catalog
from backend.engine.errors import NotImplementedFeature, SQLRuntimeError
from backend.engine.executor import Executor
from backend.vector.distance import distances

D = 8
N = 600
CATS = ["footwear", "shirts", "bags", "watches"]


def _datos(seed=7):
    rng = np.random.default_rng(seed)
    centros = rng.normal(size=(12, D)) * 4
    X = centros[rng.integers(12, size=N)] + rng.normal(size=(N, D))
    return X.astype(np.float32)


def _vec(v) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


@pytest.fixture
def db(tmp_path):
    cat = Catalog(str(tmp_path / "data"), page_size=4096)
    ex = Executor(cat)
    ex.execute(f"CREATE TABLE prod (id INT PRIMARY KEY, cat CHAR(10), precio FLOAT, emb VECTOR({D})) USING HEAP")
    X = _datos()
    sf = cat.open_table("prod")
    sf.bulk_load((i, CATS[i % 4], float(i % 500), tuple(map(float, X[i]))) for i in range(N))
    sf.flush()
    ex.X = X
    yield ex
    cat.close_all()


def _exacto(X, q, metric, k, mask=None):
    d = distances(q, X, metric)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    return [int(i) for i in np.argsort(d, kind="stable")[:k] if np.isfinite(d[i])]


def _ids(r):
    return [row[0] for row in r.rows]


# ----------------------------------------------------------------- exacto


@pytest.mark.parametrize("op,metric", [("<->", "L2"), ("<=>", "COSINE")])
def test_knn_secuencial_es_exacto(db, op, metric):
    q = db.X[3] + 0.05
    r = db.execute(f"SELECT id, emb {op} '{_vec(q)}' AS d FROM prod ORDER BY d LIMIT 10")
    assert r.plan.access == "KnnSeqScan"
    assert _ids(r) == _exacto(db.X, q, metric, 10)
    dists = [row[1] for row in r.rows]
    assert dists == sorted(dists)


def test_vector_con_dimension_equivocada_falla(db):
    with pytest.raises(SQLRuntimeError, match="dimension"):
        db.execute("SELECT id FROM prod ORDER BY emb <-> '[1, 2]' LIMIT 3")


def test_sin_limit_calcula_todas_las_distancias(db):
    r = db.execute(f"SELECT id, emb <-> '{_vec(db.X[0])}' AS d FROM prod WHERE cat = 'bags'")
    assert r.row_count == N // 4
    assert all(row[0] % 4 == 2 for row in r.rows)


# -------------------------------------------------------------------- IVF


def test_create_index_ivf_y_catalogo(db):
    r = db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16, metric = 'L2')")
    assert "16 celdas" in r.message
    info = next(i for i in db.catalog.indexes_of("prod") if i["kind"] == "IVF")
    assert (info["column"], info["metric"], info["n_lists"], info["n_vectors"]) == ("emb", "L2", 16, N)


def test_ivf_solo_sobre_vector(db):
    with pytest.raises(SQLRuntimeError, match="VECTOR"):
        db.execute("CREATE INDEX iv ON prod (precio) USING IVF")
    with pytest.raises(SQLRuntimeError, match="IVF"):
        db.execute("CREATE INDEX h ON prod (emb) USING HASH")


def test_ivf_con_todas_las_celdas_es_exacto_y_con_pocas_lee_menos(db):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    q = db.X[42] + 0.1
    sql = f"SELECT id FROM prod ORDER BY emb <-> '{_vec(q)}' LIMIT 10"

    db.execute("SET ivf.probes = 16")
    todas = db.execute(sql)
    assert todas.plan.access == "IVFScan"
    assert _ids(todas) == _exacto(db.X, q, "L2", 10)

    db.execute("SET ivf.probes = 1")
    una = db.execute(sql)
    assert una.plan.detail["io"]["index_reads"] < todas.plan.detail["io"]["index_reads"]
    assert una.plan.detail["lists_scanned"] == 1


def test_ivf_metrica_distinta_no_se_usa(db):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (metric = 'COSINE')")
    r = db.execute(f"SELECT id FROM prod ORDER BY emb <-> '{_vec(db.X[0])}' LIMIT 3")
    assert r.plan.access == "KnnSeqScan"
    r = db.execute(f"SELECT id FROM prod ORDER BY emb <=> '{_vec(db.X[0])}' LIMIT 3")
    assert r.plan.access == "IVFScan"


def test_ivf_persiste_al_reabrir(db, tmp_path):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 8)")
    db.catalog.close_all()
    otro = Executor(Catalog(str(tmp_path / "data"), page_size=4096))
    otro.execute("SET ivf.probes = 8")
    r = otro.execute(f"SELECT id FROM prod ORDER BY emb <-> '{_vec(db.X[5])}' LIMIT 1")
    assert r.plan.access == "IVFScan" and _ids(r) == [5]
    otro.catalog.close_all()


def test_insert_y_delete_mantienen_el_ivf(db):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 8)")
    db.execute("SET ivf.probes = 8")
    nuevo = np.full(D, 50.0, dtype=np.float32)
    db.execute(f"INSERT INTO prod VALUES (9999, 'bags', 10.0, '{_vec(nuevo)}')")
    r = db.execute(f"SELECT id FROM prod ORDER BY emb <-> '{_vec(nuevo)}' LIMIT 1")
    assert _ids(r) == [9999]

    db.execute("DELETE FROM prod WHERE id = 9999")
    r = db.execute(f"SELECT id FROM prod ORDER BY emb <-> '{_vec(nuevo)}' LIMIT 1")
    assert _ids(r) != [9999]
    assert db.catalog.open_index("prod", "iv").n_vectors == N


def test_delete_en_heap_reubica_la_fila_movida(db):
    """move-the-last: la ultima fila hereda el RID de la borrada y el IVF la sigue."""
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 8)")
    db.execute("SET ivf.probes = 8")
    db.execute("DELETE FROM prod WHERE id = 0")
    ultima = N - 1
    r = db.execute(f"SELECT id FROM prod ORDER BY emb <-> '{_vec(db.X[ultima])}' LIMIT 1")
    assert _ids(r) == [ultima]
    assert "stale_rids" not in r.plan.detail


# --------------------------------------------------------- planificador hibrido


def test_post_filter_sin_indice_relacional(db):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    db.execute("SET ivf.probes = 16")
    q = db.X[10]
    r = db.execute(f"SELECT id, cat FROM prod WHERE cat = 'watches' ORDER BY emb <-> '{_vec(q)}' LIMIT 5")
    assert r.plan.access == "PostFilter"
    assert all(row[1] == "watches" for row in r.rows)
    mask = np.array([i % 4 == 3 for i in range(N)])
    assert _ids(r) == _exacto(db.X, q, "L2", 5, mask)


def test_post_filter_amplia_el_sondeo_si_no_alcanza(db):
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    db.execute("SET ivf.probes = 1")
    # precio < 3 deja 6 filas dispersas: una sola celda no alcanza para 5.
    r = db.execute(f"SELECT id FROM prod WHERE precio < 3 ORDER BY emb <-> '{_vec(db.X[0])}' LIMIT 5")
    assert r.plan.access == "PostFilter"
    assert r.row_count == 5
    assert r.plan.detail.get("expansions", 0) >= 1


def test_pre_filter_sin_ivf_usa_el_btree(db):
    db.execute("CREATE INDEX ip ON prod (precio) USING BTREE")
    q = db.X[1]
    r = db.execute(f"SELECT id FROM prod WHERE precio <= 20 AND cat = 'footwear' ORDER BY emb <-> '{_vec(q)}' LIMIT 4")
    assert r.plan.access == "PreFilter"
    assert r.plan.detail["prefilter_access"] == "IndexRangeScan"
    mask = np.array([(i % 500) <= 20 and i % 4 == 0 for i in range(N)])
    assert _ids(r) == _exacto(db.X, q, "L2", 4, mask)
    assert r.plan.detail["io"]["index_reads"] > 0


@pytest.mark.parametrize("forzada,acceso", [("PRE", "PreFilter"), ("POST", "PostFilter")])
def test_estrategia_forzada(db, forzada, acceso):
    db.execute("CREATE INDEX ip ON prod (precio) USING BTREE")
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    db.execute("SET ivf.probes = 16")
    db.execute(f"SET hybrid.strategy = '{forzada}'")
    q = db.X[2]
    r = db.execute(f"SELECT id FROM prod WHERE precio <= 100 ORDER BY emb <-> '{_vec(q)}' LIMIT 5")
    assert r.plan.access == acceso
    mask = np.array([(i % 500) <= 100 for i in range(N)])
    assert _ids(r) == _exacto(db.X, q, "L2", 5, mask)


def test_set_valida_opciones(db):
    with pytest.raises(SQLRuntimeError):
        db.execute("SET ivf.probes = 0")
    with pytest.raises(SQLRuntimeError):
        db.execute("SET hybrid.strategy = 'MAYBE'")
    with pytest.raises(SQLRuntimeError, match="desconocida"):
        db.execute("SET foo.bar = 1")


# ------------------------------------------- AUTO: modelo de costos PRE vs POST


def test_auto_elige_pre_con_filtro_muy_selectivo(db):
    db.execute("CREATE INDEX ip ON prod (precio) USING BTREE")
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    r = db.execute(f"SELECT id FROM prod WHERE precio = 7 ORDER BY emb <-> '{_vec(db.X[0])}' LIMIT 3")
    assert r.plan.access == "PreFilter"
    assert "estimate" in r.plan.detail and r.plan.detail["estimate"]["est_matches"] == 2


def test_auto_elige_post_con_filtro_poco_selectivo(db):
    db.execute("CREATE INDEX ip ON prod (precio) USING BTREE")
    db.execute("CREATE INDEX iv ON prod (emb) USING IVF WITH (lists = 16)")
    r = db.execute(f"SELECT id FROM prod WHERE precio >= 0 ORDER BY emb <-> '{_vec(db.X[0])}' LIMIT 3")
    assert r.plan.access == "PostFilter"


# ------------------------------------------------- SQL relacional extendido


def test_where_multicolumna_con_filtro_residual_y_order_by(db):
    db.execute("CREATE INDEX ip ON prod (precio) USING BTREE")
    r = db.execute("SELECT id, precio FROM prod WHERE cat = 'shirts' AND precio < 30 ORDER BY precio DESC LIMIT 3")
    assert r.plan.access == "IndexRangeScan"
    assert r.plan.detail["residual_filters"] == ["cat"]
    assert [row[1] for row in r.rows] == [29.0, 29.0, 25.0]  # ids 29 y 529


def test_match_y_within_radius_se_reportan_como_pendientes(db):
    with pytest.raises(NotImplementedFeature, match="textual"):
        db.execute("SELECT id FROM prod WHERE cat MATCH 'zapatillas'")
    with pytest.raises(NotImplementedFeature, match="espacial"):
        db.execute("SELECT id FROM prod WHERE cat WITHIN_RADIUS((1, 2), 3)")
