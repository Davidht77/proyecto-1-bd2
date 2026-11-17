"""Indice invertido conectado al SQL: CREATE INDEX ... USING INVERTED y MATCH
(enunciado 5.2 y 5.4). El Full Text Scan es la referencia: con indice o sin
indice la respuesta tiene que ser la misma; lo que cambia es el I/O."""

import csv
from pathlib import Path

import numpy as np
import pytest

from backend.engine.catalog import Catalog
from backend.engine.errors import SQLRuntimeError
from backend.engine.executor import Executor

N_DOCS = 300
with open(Path(__file__).parent / "fixtures" / "wiki_muestra.csv", encoding="utf-8") as f:
    _todas = list(csv.DictReader(f))
# mitad en espanol y mitad en ingles
FILAS = [r for r in _todas if r["lang"] == "es"][: N_DOCS // 2] + [r for r in _todas if r["lang"] == "en"][: N_DOCS // 2]
D = 4

CONSULTAS = [
    "historia AND ciudad",
    "historia AND NOT guerra",
    "(guerra OR batalla) AND NOT mundial",
    "music rock",
    "río montaña",
]


def _vec(i):
    rng = np.random.default_rng(i)
    return "[" + ", ".join(f"{x:.4f}" for x in rng.normal(size=D)) + "]"


def _cargar(ex, engine="HEAP"):
    ex.execute(f"CREATE TABLE docs (id INT PRIMARY KEY, titulo CHAR(80), texto CHAR(1000), "
               f"anio INT, emb VECTOR({D})) USING {engine}")
    sf = ex.catalog.open_table("docs")
    sf.bulk_load((i, r["title"].encode()[:80].decode("utf-8", "ignore"), r["text"], 1990 + i % 30,
                  tuple(np.random.default_rng(i).normal(size=D).round(4).tolist()))
                 for i, r in enumerate(FILAS))
    sf.flush()


@pytest.fixture
def db(tmp_path):
    cat = Catalog(str(tmp_path / "data"))
    ex = Executor(cat)
    _cargar(ex)
    yield ex
    cat.close_all()


@pytest.fixture
def db_idx(db):
    db.execute("CREATE INDEX itx ON docs (texto) USING INVERTED")
    return db


def _ids(r):
    return [row[0] for row in r.rows]


# ------------------------------------------------------------ DDL y catalogo


def test_create_index_inverted_y_catalogo(db):
    r = db.execute("CREATE INDEX itx ON docs (texto) USING INVERTED WITH (scoring = 'BM25')")
    assert f"{N_DOCS} documentos" in r.message
    info = [i for i in db.catalog.indexes_of("docs") if not i.get("implicit")]
    assert info == [{"name": "itx", "column": "texto", "kind": "INVERTED", "implicit": False,
                     "scoring": "BM25", "stale": False}]  # el vocabulario B+ no aparece como indice


def test_inverted_valida_columna_y_opciones(db):
    with pytest.raises(SQLRuntimeError, match="CHAR"):
        db.execute("CREATE INDEX i ON docs (anio) USING INVERTED")
    with pytest.raises(SQLRuntimeError, match="scoring"):
        db.execute("CREATE INDEX i ON docs (texto) USING INVERTED WITH (scoring = 'PAGERANK')")
    with pytest.raises(SQLRuntimeError, match="desconocidas"):
        db.execute("CREATE INDEX i ON docs (texto) USING INVERTED WITH (lists = 3)")


def test_drop_table_borra_los_archivos_del_invertido(db_idx, tmp_path):
    db_idx.execute("DROP TABLE docs")
    assert not list((tmp_path / "data").glob("docs__itx*"))


# ------------------------------------------- misma respuesta con y sin indice


@pytest.mark.parametrize("consulta", CONSULTAS)
@pytest.mark.parametrize("funcion", ["COSINE_SCORE", "BM25_SCORE"])
def test_indice_y_full_scan_coinciden(db, consulta, funcion):
    sql = (f"SELECT id, {funcion}() AS s FROM docs WHERE texto MATCH '{consulta}' "
           f"ORDER BY s DESC LIMIT 8")
    sin = db.execute(sql)
    assert sin.plan.access == "TextSeqScan"
    db.execute("CREATE INDEX itx ON docs (texto) USING INVERTED")
    con = db.execute(sql)
    assert con.plan.access == "InvertedIndexScan"
    assert [r[1] for r in con.rows] == pytest.approx([r[1] for r in sin.rows], rel=1e-5)
    assert set(_ids(con)) == set(_ids(sin))
    assert con.row_count > 0


def test_el_indice_lee_mucho_menos_que_el_full_scan(db_idx):
    sql = "SELECT id FROM docs WHERE texto MATCH 'historia AND ciudad'"
    con = db_idx.execute(sql)
    io = con.plan.detail["io"]
    assert io["table_reads"] == con.plan.detail["text"]["rows_fetched"] == con.row_count
    db_idx.catalog.drop_index("docs", "itx")
    sin = db_idx.execute(sql)
    assert sin.plan.detail["io"]["table_reads"] == db_idx.catalog.open_table("docs").page_count
    assert con.disk_reads < sin.disk_reads


def test_booleana_respeta_and_not(db_idx):
    r = db_idx.execute("SELECT id, texto FROM docs WHERE texto MATCH 'football AND NOT women'")
    assert r.plan.detail["text"]["mode"] == "boolean"
    for _, texto in r.rows:
        t = texto.lower()
        assert "football" in t and "women" not in t


def test_ranking_sin_operadores_es_or(db_idx):
    a = db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'football'").row_count
    b = db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'guerra'").row_count
    ab = db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'football guerra'")
    assert ab.plan.detail["text"]["mode"] == "ranked"
    assert max(a, b) <= ab.row_count <= a + b


# ------------------------------------------------- hibrido texto + relacional


def test_filtros_relacionales_y_corte_por_limit(db_idx):
    r = db_idx.execute("SELECT id, anio, COSINE_SCORE() AS s FROM docs "
                       "WHERE texto MATCH 'historia ciudad' AND anio >= 2010 ORDER BY s DESC LIMIT 3")
    assert r.row_count == 3 and all(row[1] >= 2010 for row in r.rows)
    assert [row[2] for row in r.rows] == sorted((row[2] for row in r.rows), reverse=True)
    det = r.plan.detail
    assert det["residual_filters"] == ["anio"]
    # Filter-and-refine: se para al juntar 3, sin leer todas las candidatas.
    assert det["text"]["rows_fetched"] < det["text"]["matches"]


def test_order_by_columna(db_idx):
    r = db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'historia' ORDER BY id DESC LIMIT 5")
    assert _ids(r) == sorted(_ids(r), reverse=True)


def test_match_mas_distancia_vectorial(db_idx):
    q = "[0.1, -0.2, 0.3, 0.05]"
    r = db_idx.execute(f"SELECT id, emb <-> '{q}' AS d FROM docs WHERE texto MATCH 'historia' "
                       f"ORDER BY d LIMIT 4")
    assert r.plan.access == "TextPreFilter"
    candidatos = _ids(db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'historia'"))
    X = {i: np.random.default_rng(i).normal(size=D).round(4) for i in candidatos}
    qv = np.array([0.1, -0.2, 0.3, 0.05])
    esperado = sorted(candidatos, key=lambda i: np.linalg.norm(X[i] - qv))[:4]
    assert _ids(r) == esperado


# ------------------------------------------------------------- mantenimiento


def test_insert_y_delete_reconstruyen_el_indice(db_idx):
    db_idx.execute(f"INSERT INTO docs VALUES (5000, 'Nuevo', 'zorblax palabra inventada', 2020, '{_vec(1)}')")
    assert db_idx.catalog.indexes_of("docs")[0]["stale"]
    r = db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'zorblax'")
    assert _ids(r) == [5000]
    assert "rebuilt" in r.plan.detail["text"]
    db_idx.execute("DELETE FROM docs WHERE id = 5000")
    assert db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'zorblax'").row_count == 0
    # move-the-last movio la ultima fila: tiene que seguir encontrandose.
    ultima = N_DOCS - 1
    titulo = FILAS[ultima]["text"].split()[0]
    assert ultima in _ids(db_idx.execute(f"SELECT id FROM docs WHERE texto MATCH '{titulo}'"))


def test_sequential_file_con_overflow(tmp_path):
    cat = Catalog(str(tmp_path / "seq"))
    ex = Executor(cat)
    ex.execute("CREATE TABLE s (id INT PRIMARY KEY, texto CHAR(40)) USING SEQUENTIAL")
    for i in range(0, 60, 2):
        ex.execute(f"INSERT INTO s VALUES ({i}, 'base comun {i}')")
    for i in range(1, 60, 2):  # caen en overflow entre los anteriores
        ex.execute(f"INSERT INTO s VALUES ({i}, 'impar especial')")
    ex.execute("CREATE INDEX it ON s (texto) USING INVERTED")
    r = ex.execute("SELECT id FROM s WHERE texto MATCH 'especial' ORDER BY id ASC")
    assert _ids(r) == list(range(1, 60, 2))
    cat.close_all()


# ------------------------------------------------------------------- errores


def test_errores_de_match(db_idx):
    with pytest.raises(SQLRuntimeError, match="texto"):
        db_idx.execute("SELECT id FROM docs WHERE anio MATCH 'x'")
    with pytest.raises(SQLRuntimeError, match="positivo"):
        db_idx.execute("SELECT id FROM docs WHERE texto MATCH 'NOT historia'")
    with pytest.raises(SQLRuntimeError, match="una sola"):
        db_idx.execute("SELECT COSINE_SCORE(), BM25_SCORE() FROM docs WHERE texto MATCH 'historia'")
