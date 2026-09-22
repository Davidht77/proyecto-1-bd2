import pytest

from backend.engine.catalog import Catalog
from backend.engine.errors import (
    ColumnNotFound,
    SQLRuntimeError,
    TableAlreadyExists,
    TableNotFound,
)
from backend.engine.executor import Executor


@pytest.fixture
def db(tmp_path):
    cat = Catalog(str(tmp_path / "data"), page_size=1024)
    yield Executor(cat)
    cat.close_all()


def crear(db, n=0):
    db.execute(
        "CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(20), salario FLOAT) "
        "USING SEQUENTIAL"
    )
    for k in range(n):
        db.execute(f"INSERT INTO empleados VALUES ({k}, 'emp{k}', {1000.0 + k})")


# ------------------------------------------------------------------- DDL


def test_create_table_y_listado(db):
    crear(db)
    tablas = db.catalog.list_tables()
    assert [t.name for t in tablas] == ["empleados"]
    t = tablas[0]
    assert t.engine == "SEQUENTIAL"
    assert t.record_size == 4 + 20 + 8
    assert [c.name for c in t.schema.columns] == ["id", "nombre", "salario"]


def test_el_catalogo_se_reconstruye_del_disco(db, tmp_path):
    """No hay archivo de catalogo: el esquema vive en la pagina 0 de cada tabla."""
    crear(db, n=5)
    db.catalog.close_all()

    otro = Catalog(str(tmp_path / "data"), page_size=1024)
    info = otro.describe("empleados")
    assert info.n_records == 5
    assert [c.name for c in info.schema.columns] == ["id", "nombre", "salario"]
    otro.close_all()


def test_create_table_duplicada_falla(db):
    crear(db)
    with pytest.raises(TableAlreadyExists):
        crear(db)


def test_drop_table(db):
    crear(db, n=3)
    db.execute("DROP TABLE empleados")
    assert db.catalog.table_names() == []
    with pytest.raises(TableNotFound):
        db.execute("SELECT * FROM empleados")


def test_create_table_heap(db):
    db.execute("CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    info = db.catalog.describe("libre")
    assert info.engine == "HEAP"
    assert info.n_overflow == 0  # el Heap File no tiene area de overflow


def test_el_catalogo_reabre_una_tabla_heap(db, tmp_path):
    """El archivo heap tambien debe ser autodescriptivo: su esquema vive en la
    pagina 0, como el del Sequential File."""
    db.execute("CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    for k in range(5):
        db.execute(f"INSERT INTO libre VALUES ({k}, 'v{k}')")
    db.catalog.close_all()

    otro = Catalog(str(tmp_path / "data"), page_size=1024)
    info = otro.describe("libre")
    assert info.n_records == 5
    assert [c.name for c in info.schema.columns] == ["id", "v"]
    otro.close_all()


def test_el_heap_no_tiene_indice_implicito(db):
    db.execute("CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    db.execute("CREATE TABLE ord (id INT PRIMARY KEY, v CHAR(10)) USING SEQUENTIAL")
    assert db.catalog.indexes_of("libre") == []
    assert db.catalog.indexes_of("ord")[0]["kind"] == "SEQUENTIAL"


def test_la_igualdad_sobre_un_heap_sigue_siendo_full_scan(db):
    """Sin orden fisico no hay busqueda binaria posible: es el argumento
    central del Experimento 2."""
    db.execute("CREATE TABLE libre (id INT PRIMARY KEY, v CHAR(10)) USING HEAP")
    for k in range(300):
        db.execute(f"INSERT INTO libre VALUES ({k}, 'v{k}')")
    r = db.execute("SELECT * FROM libre WHERE id = 250")
    assert r.plan.access == "SeqScan"
    assert "Heap File" in r.plan.reason
    assert r.rows == [[250, "v250"]]


def test_el_heap_lee_mas_bloques_que_el_sequential_en_igualdad(db):
    for motor in ("HEAP", "SEQUENTIAL"):
        db.execute(f"CREATE TABLE t{motor} (id INT PRIMARY KEY, v CHAR(10)) USING {motor}")
        for k in range(400):
            db.execute(f"INSERT INTO t{motor} VALUES ({k}, 'v{k}')")
    heap = db.execute("SELECT * FROM tHEAP WHERE id = 399")
    seq = db.execute("SELECT * FROM tSEQUENTIAL WHERE id = 399")
    assert heap.plan.access == "SeqScan"
    assert seq.plan.access == "BinarySearch"
    assert heap.disk_reads > seq.disk_reads


def test_create_index_hash_habilita_index_scan(db):
    crear(db, n=10)
    db.execute("CREATE INDEX i ON empleados (id) USING HASH")
    r = db.execute("SELECT * FROM empleados WHERE id = 5")
    assert r.plan.access == "IndexScan"
    assert r.rows[0][0] == 5


def test_create_index_btree_habilita_index_scan(db):
    crear(db, n=10)
    db.execute("CREATE INDEX i ON empleados (id) USING BTREE")
    r = db.execute("SELECT * FROM empleados WHERE id = 5")
    assert r.plan.access == "IndexScan"
    assert r.rows[0][0] == 5


def test_create_index_btree_habilita_index_range_scan(db):
    crear(db, n=10)
    db.execute("CREATE INDEX i ON empleados (id) USING BTREE")
    r = db.execute("SELECT * FROM empleados WHERE id BETWEEN 3 AND 6")
    assert r.plan.access == "IndexRangeScan"
    assert [row[0] for row in r.rows] == [3, 4, 5, 6]


def test_create_index_sobre_tabla_inexistente(db):
    with pytest.raises(TableNotFound):
        db.execute("CREATE INDEX i ON nada (id) USING BTREE")


# ------------------------------------------------------------------ INSERT


def test_insert_y_select_puntual(db):
    crear(db, n=10)
    r = db.execute("SELECT * FROM empleados WHERE id = 7")
    assert r.rows == [[7, "emp7", 1007.0]]
    assert r.row_count == 1


def test_insert_con_aridad_incorrecta(db):
    crear(db)
    with pytest.raises(SQLRuntimeError, match="3 columnas"):
        db.execute("INSERT INTO empleados VALUES (1, 'x')")


def test_insert_reporta_escrituras(db):
    crear(db)
    r = db.execute("INSERT INTO empleados VALUES (1, 'x', 2.0)")
    assert r.disk_writes > 0
    assert "1 registro insertado" in r.message


# ------------------------------------------------- planificador (enunciado 3.4)


def test_igualdad_sobre_la_pk_usa_busqueda_binaria(db):
    crear(db, n=200)
    r = db.execute("SELECT * FROM empleados WHERE id = 150")
    assert r.plan.access == "BinarySearch"
    assert "clave primaria" in r.plan.reason


def test_rango_sobre_la_pk_usa_binary_range_scan(db):
    crear(db, n=200)
    r = db.execute("SELECT * FROM empleados WHERE id >= 50 AND id <= 60")
    assert r.plan.access == "BinaryRangeScan"
    assert [row[0] for row in r.rows] == list(range(50, 61))


def test_filtro_sobre_columna_no_clave_usa_full_scan(db):
    crear(db, n=50)
    r = db.execute("SELECT * FROM empleados WHERE salario = 1010.0")
    assert r.plan.access == "SeqScan"
    assert "no es la clave primaria" in r.plan.reason
    assert r.rows == [[10, "emp10", 1010.0]]


def test_select_sin_where_usa_full_scan(db):
    crear(db, n=20)
    r = db.execute("SELECT * FROM empleados")
    assert r.plan.access == "SeqScan"
    assert r.row_count == 20


def test_la_busqueda_binaria_lee_menos_bloques_que_el_full_scan(db):
    """Es la comparativa central del Experimento 2, verificada en test."""
    crear(db, n=400)
    binaria = db.execute("SELECT * FROM empleados WHERE id = 399")
    full = db.execute("SELECT * FROM empleados WHERE salario = 1399.0")
    assert binaria.plan.access == "BinarySearch"
    assert full.plan.access == "SeqScan"
    assert binaria.disk_reads < full.disk_reads


def test_el_plan_estima_lecturas(db):
    crear(db, n=300)
    r = db.execute("SELECT * FROM empleados WHERE id = 1")
    assert r.plan.estimated_reads is not None
    assert r.disk_reads <= r.plan.estimated_reads + 1


# ------------------------------------------------------------------ SELECT


def test_proyeccion_de_columnas(db):
    crear(db, n=5)
    r = db.execute("SELECT nombre, id FROM empleados WHERE id = 3")
    assert r.columns == ["nombre", "id"]
    assert r.rows == [["emp3", 3]]


def test_columna_inexistente_falla(db):
    crear(db, n=1)
    with pytest.raises(ColumnNotFound, match="Columnas disponibles: id, nombre, salario"):
        db.execute("SELECT nada FROM empleados")


def test_where_sobre_columna_inexistente_falla(db):
    crear(db, n=1)
    with pytest.raises(ColumnNotFound):
        db.execute("SELECT * FROM empleados WHERE nada = 1")


def test_rango_exclusivo(db):
    crear(db, n=20)
    r = db.execute("SELECT * FROM empleados WHERE id > 5 AND id < 9")
    assert [row[0] for row in r.rows] == [6, 7, 8]


def test_between(db):
    crear(db, n=20)
    r = db.execute("SELECT * FROM empleados WHERE id BETWEEN 5 AND 8")
    assert [row[0] for row in r.rows] == [5, 6, 7, 8]


def test_limit(db):
    crear(db, n=50)
    r = db.execute("SELECT * FROM empleados LIMIT 5")
    assert r.row_count == 5


def test_max_rows_trunca_pero_reporta_el_total(db):
    crear(db, n=60)
    r = db.execute("SELECT * FROM empleados", max_rows=10)
    assert len(r.rows) == 10
    assert r.row_count == 60
    assert r.truncated


def test_select_sin_resultados(db):
    crear(db, n=10)
    r = db.execute("SELECT * FROM empleados WHERE id = 9999")
    assert r.rows == [] and r.row_count == 0


# ------------------------------------------------------------------ DELETE


def test_delete_puntual(db):
    crear(db, n=10)
    r = db.execute("DELETE FROM empleados WHERE id = 5")
    assert "1 registro" in r.message
    assert db.execute("SELECT * FROM empleados WHERE id = 5").row_count == 0
    assert db.execute("SELECT * FROM empleados").row_count == 9


def test_delete_por_rango(db):
    crear(db, n=20)
    db.execute("DELETE FROM empleados WHERE id >= 5 AND id <= 9")
    assert db.execute("SELECT * FROM empleados").row_count == 15


def test_delete_sin_where_se_rechaza(db):
    crear(db, n=5)
    with pytest.raises(SQLRuntimeError, match="DROP TABLE"):
        db.execute("DELETE FROM empleados")
    assert db.execute("SELECT * FROM empleados").row_count == 5


# ------------------------------------------------------------- telemetria


def test_toda_consulta_reporta_metricas(db):
    crear(db, n=30)
    r = db.execute("SELECT * FROM empleados WHERE id = 1")
    d = r.as_dict()
    assert set(d["metrics"]) == {"disk_reads", "disk_writes", "parse_ms", "exec_ms", "total_ms"}
    assert d["metrics"]["parse_ms"] >= 0
    assert d["metrics"]["exec_ms"] >= 0
    assert d["plan"]["access"] == "BinarySearch"


def test_la_persistencia_sobrevive_a_reabrir(db, tmp_path):
    crear(db, n=25)
    db.catalog.close_all()
    otro = Executor(Catalog(str(tmp_path / "data"), page_size=1024))
    assert otro.execute("SELECT * FROM empleados").row_count == 25
    assert otro.execute("SELECT * FROM empleados WHERE id = 24").rows == [[24, "emp24", 1024.0]]
    otro.catalog.close_all()
