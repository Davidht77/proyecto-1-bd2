"""Tests de la API REST.

Cada test arranca con un directorio de datos limpio, recargando el modulo de la
app con BD2_DATA_DIR apuntando a tmp_path. Es menos elegante que inyectar el
catalogo, pero mantiene el modulo simple: la app se configura por entorno, que
es como se despliega.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BD2_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BD2_PAGE_SIZE", "1024")
    import backend.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c
    main.catalog.close_all()


def sql(client, texto, **kw):
    return client.post("/api/query", json={"sql": texto, **kw})


def crear(client, n=0):
    sql(client, "CREATE TABLE emp (id INT PRIMARY KEY, nombre CHAR(20), salario FLOAT)")
    for k in range(n):
        sql(client, f"INSERT INTO emp VALUES ({k}, 'emp{k}', {1000.0 + k})")


# ------------------------------------------------------------------ health


def test_health_declara_que_falta(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    impl = r.json()["implemented"]
    assert impl["storage"] and impl["sequential_file"] and impl["heap_file"] and impl["hash"]
    assert not impl["btree"]


# ------------------------------------------------------------------- query


def test_query_select_devuelve_filas_plan_y_metricas(client):
    crear(client, n=20)
    r = sql(client, "SELECT * FROM emp WHERE id = 7")
    assert r.status_code == 200
    body = r.json()
    assert body["columns"] == ["id", "nombre", "salario"]
    assert body["rows"] == [[7, "emp7", 1007.0]]
    assert body["plan"]["access"] == "BinarySearch"
    assert body["metrics"]["disk_reads"] > 0
    assert body["metrics"]["total_ms"] >= 0


def test_error_de_sintaxis_es_400(client):
    r = sql(client, "SELEC * FROM emp")
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "sintaxis"


def test_tabla_inexistente_es_404(client):
    r = sql(client, "SELECT * FROM nada")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "tabla_no_encontrada"


def test_tabla_duplicada_es_409(client):
    crear(client)
    r = sql(client, "CREATE TABLE emp (id INT)")
    assert r.status_code == 409


def test_funcionalidad_pendiente_es_501(client):
    crear(client)
    r = sql(client, "CREATE INDEX i ON emp (id) USING BTREE")
    assert r.status_code == 501
    assert r.json()["detail"]["error"] == "no_implementado"
    assert "árbol B+" in r.json()["detail"]["detail"]


def test_heap_funciona_y_se_lista(client):
    assert sql(client, "CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP").status_code == 200
    for k in range(10):
        sql(client, f"INSERT INTO libre VALUES ({k}, 'v{k}')")
    t = next(t for t in client.get("/api/tables").json()["tables"] if t["name"] == "libre")
    assert t["engine"] == "HEAP"
    assert t["n_records"] == 10
    assert t["indexes"] == []  # sin orden fisico no hay indice implicito
    body = sql(client, "SELECT * FROM libre WHERE id = 7").json()
    assert body["plan"]["access"] == "SeqScan"
    assert body["rows"] == [[7, "v7"]]


def test_max_rows_se_respeta(client):
    crear(client, n=40)
    body = sql(client, "SELECT * FROM emp", max_rows=5).json()
    assert len(body["rows"]) == 5
    assert body["row_count"] == 40
    assert body["truncated"]


# ------------------------------------------------------------------ tables


def test_listado_de_tablas(client):
    crear(client, n=3)
    body = client.get("/api/tables").json()
    assert len(body["tables"]) == 1
    t = body["tables"][0]
    assert t["name"] == "emp"
    assert t["engine"] == "SEQUENTIAL"
    assert t["n_records"] == 3
    assert [c["name"] for c in t["columns"]] == ["id", "nombre", "salario"]
    assert t["columns"][0]["primary_key"] is True
    assert t["indexes"][0]["kind"] == "SEQUENTIAL"


def test_describe_tabla(client):
    crear(client, n=2)
    body = client.get("/api/tables/emp").json()
    assert body["record_size"] == 32
    assert body["size_bytes"] > 0


def test_describe_tabla_inexistente_es_404(client):
    assert client.get("/api/tables/nada").status_code == 404


# ------------------------------------------------------------- reorganize


def test_reorganize_reporta_antes_y_despues(client):
    crear(client, n=200)
    body = client.post("/api/tables/emp/reorganize").json()
    assert body["table"] == "emp"
    assert "pages" in body["before"] and "pages" in body["after"]
    assert body["after"]["overflow_records"] == 0
    assert body["stats"]["records"] == 200
    assert body["stats"]["disk_writes"] > 0


def test_reorganize_conserva_los_datos(client):
    crear(client, n=100)
    antes = sql(client, "SELECT * FROM emp", max_rows=500).json()["rows"]
    client.post("/api/tables/emp/reorganize")
    assert sql(client, "SELECT * FROM emp", max_rows=500).json()["rows"] == antes


# ----------------------------------------------------------------- pages


def test_inspeccion_de_paginas(client):
    crear(client, n=100)
    body = client.get("/api/tables/emp/pages").json()
    assert body["page_size"] == 1024
    assert body["total_pages"] >= 1
    assert body["capacity_per_page"] > 0
    p0 = body["pages"][0]
    assert p0["logical_index"] == 0
    assert p0["low_key"] is None  # centinela -inf de la primera pagina
    assert p0["page_type"] == "DATA"
    assert 0 < p0["fill_pct"] <= 100
    assert isinstance(p0["overflow_chain"], list)


def test_la_inspeccion_reporta_el_overflow_global(client):
    """La franja solo dibuja una ventana de paginas, pero el contador de
    overflow es del archivo entero: si no, la leyenda mentiria cuando el
    overflow esta fuera de la ventana."""
    crear(client, n=400)
    body = client.get("/api/tables/emp/pages?limit=3").json()
    assert len(body["pages"]) == 3
    assert body["total_pages"] > 3
    assert body["overflow_records"] == client.get("/api/tables/emp").json()["n_overflow"]
    assert body["limit"] == 3


def test_paginacion_de_la_inspeccion(client):
    crear(client, n=200)
    total = client.get("/api/tables/emp/pages").json()["total_pages"]
    if total > 1:
        body = client.get("/api/tables/emp/pages?offset=1&limit=1").json()
        assert body["pages"][0]["logical_index"] == 1


# ------------------------------------------------------------------ seed


def test_seed_bulk(client):
    sql(client, "CREATE TABLE big (id INT PRIMARY KEY, v CHAR(10))")
    body = client.post("/api/tables/big/seed", json={"n": 5000, "mode": "bulk"}).json()
    assert body["inserted"] == 5000
    assert body["state"]["n_records"] == 5000
    assert body["metrics"]["disk_reads"] == 0  # bulk_load no lee
    assert sql(client, "SELECT * FROM big WHERE id = 4999").json()["row_count"] == 1


def test_seed_bulk_sobre_tabla_no_vacia_es_409(client):
    crear(client, n=1)
    r = client.post("/api/tables/emp/seed", json={"n": 10, "mode": "bulk"})
    assert r.status_code == 409


def test_seed_insert_si_usa_la_ruta_normal(client):
    sql(client, "CREATE TABLE big (id INT PRIMARY KEY, v CHAR(10))")
    body = client.post("/api/tables/big/seed", json={"n": 300, "mode": "insert"}).json()
    assert body["state"]["n_records"] == 300
    assert body["metrics"]["disk_reads"] > 0  # insert si hace busqueda binaria


# ------------------------------------------------ endpoints pendientes (501)


def test_benchmarks_declara_lo_que_falta(client):
    r = client.post("/api/benchmarks/run")
    assert r.status_code == 501
    assert set(r.json()["detail"]["missing"]) == {"btree", "hash"}


def test_reorganizar_un_heap_es_409(client):
    """Reorganizar es una operacion del Sequential File; en un Heap no aplica."""
    sql(client, "CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    r = client.post("/api/tables/libre/reorganize")
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "no_aplica"


def test_inspeccion_de_paginas_de_un_heap(client):
    sql(client, "CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    for k in range(60):
        sql(client, f"INSERT INTO libre VALUES ({k}, 'v{k}')")
    body = client.get("/api/tables/libre/pages").json()
    assert body["engine"] == "HEAP"
    assert body["total_pages"] >= 1
    assert body["overflow_records"] == 0
    assert all(p["low_key"] is None for p in body["pages"])
    assert all(p["overflow_chain"] == [] for p in body["pages"])


def test_inspeccion_de_indice_declara_lo_que_falta(client):
    crear(client)
    r = client.get("/api/tables/emp/index/cualquiera")
    assert r.status_code == 501
    assert "btree" in r.json()["detail"]["missing"]
