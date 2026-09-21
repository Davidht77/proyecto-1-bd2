import pytest

from backend.engine.errors import SQLSyntaxError
from backend.engine.parser import (
    CreateIndex,
    CreateTable,
    Delete,
    DropTable,
    Insert,
    Select,
    parse,
)
from backend.storage.schema import ColumnType


# ----------------------------------------------------------- CREATE TABLE


def test_create_table_con_todos_los_tipos():
    st = parse(
        "CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30), "
        "dept CHAR(20), salario FLOAT) USING SEQUENTIAL"
    )
    assert isinstance(st, CreateTable)
    assert st.table == "empleados"
    assert st.engine == "SEQUENTIAL"
    assert st.pk_index == 0
    assert [c.name for c in st.columns] == ["id", "nombre", "dept", "salario"]
    assert st.columns[0].type is ColumnType.INT
    assert st.columns[1].type is ColumnType.CHAR and st.columns[1].length == 30
    assert st.columns[3].type is ColumnType.FLOAT


def test_create_table_sin_using_usa_sequential():
    assert parse("CREATE TABLE t (id INT)").engine == "SEQUENTIAL"


def test_create_table_con_heap():
    assert parse("CREATE TABLE t (id INT) USING HEAP").engine == "HEAP"


def test_primary_key_en_columna_no_inicial():
    st = parse("CREATE TABLE t (nombre CHAR(10), id INT PRIMARY KEY)")
    assert st.pk_index == 1


def test_sin_primary_key_usa_la_primera_columna():
    assert parse("CREATE TABLE t (a INT, b INT)").pk_index == 0


def test_dos_primary_key_falla():
    with pytest.raises(SQLSyntaxError, match="una PRIMARY KEY"):
        parse("CREATE TABLE t (a INT PRIMARY KEY, b INT PRIMARY KEY)")


def test_tipo_no_soportado_falla():
    with pytest.raises(SQLSyntaxError, match="tipo no soportado"):
        parse("CREATE TABLE t (a VARCHAR)")


def test_char_sin_longitud_falla():
    with pytest.raises(SQLSyntaxError):
        parse("CREATE TABLE t (a CHAR)")


def test_motor_invalido_falla():
    with pytest.raises(SQLSyntaxError, match="motor no soportado"):
        parse("CREATE TABLE t (a INT) USING BTREE")


# ------------------------------------------------------------------ INSERT


def test_insert_mezcla_tipos():
    st = parse("INSERT INTO empleados VALUES (101, 'Ada Lovelace', 'Analytics', 5200.0)")
    assert isinstance(st, Insert)
    assert st.table == "empleados"
    assert st.values == [101, "Ada Lovelace", "Analytics", 5200.0]


def test_insert_con_negativos():
    assert parse("INSERT INTO t VALUES (-5, -2.5)").values == [-5, -2.5]


def test_comilla_escapada_en_string():
    assert parse("INSERT INTO t VALUES ('O''Brien')").values == ["O'Brien"]


# ------------------------------------------------------------------ SELECT


def test_select_estrella_sin_where():
    st = parse("SELECT * FROM empleados")
    assert isinstance(st, Select)
    assert st.columns == []
    assert st.where is None


def test_select_con_proyeccion():
    assert parse("SELECT id, nombre FROM t").columns == ["id", "nombre"]


def test_select_igualdad():
    w = parse("SELECT * FROM empleados WHERE id = 101").where
    assert w.column == "id"
    assert w.is_equality and w.lo == 101 and w.hi == 101


def test_select_rango_con_and():
    w = parse("SELECT * FROM empleados WHERE id >= 100 AND id <= 500").where
    assert (w.lo, w.hi) == (100, 500)
    assert w.lo_inclusive and w.hi_inclusive
    assert not w.is_equality


def test_select_rango_exclusivo():
    w = parse("SELECT * FROM t WHERE id > 10 AND id < 20").where
    assert (w.lo, w.hi) == (10, 20)
    assert not w.lo_inclusive and not w.hi_inclusive


def test_select_between():
    w = parse("SELECT * FROM t WHERE id BETWEEN 5 AND 9").where
    assert (w.lo, w.hi) == (5, 9)
    assert w.lo_inclusive and w.hi_inclusive


def test_select_solo_cota_inferior():
    w = parse("SELECT * FROM t WHERE id >= 100").where
    assert w.lo == 100 and w.hi is None


def test_select_solo_cota_superior():
    w = parse("SELECT * FROM t WHERE id < 100").where
    assert w.lo is None and w.hi == 100 and not w.hi_inclusive


def test_where_sobre_dos_columnas_falla():
    with pytest.raises(SQLSyntaxError, match="una columna"):
        parse("SELECT * FROM t WHERE a > 1 AND b < 2")


def test_select_con_limit():
    assert parse("SELECT * FROM t LIMIT 10").limit == 10


# ------------------------------------------------------------------ DELETE


def test_delete_con_where():
    st = parse("DELETE FROM empleados WHERE id = 7")
    assert isinstance(st, Delete)
    assert st.where.is_equality and st.where.lo == 7


def test_delete_sin_where_parsea():
    """El parser lo acepta; el ejecutor es quien lo rechaza."""
    assert parse("DELETE FROM t").where is None


# ------------------------------------------------------- CREATE INDEX / DROP


def test_create_index():
    st = parse("CREATE INDEX idx_emp_id ON empleados (id) USING BTREE")
    assert isinstance(st, CreateIndex)
    assert (st.name, st.table, st.column, st.kind) == ("idx_emp_id", "empleados", "id", "BTREE")


def test_create_index_hash():
    assert parse("CREATE INDEX i ON t (a) USING HASH").kind == "HASH"


def test_indice_no_soportado_falla():
    with pytest.raises(SQLSyntaxError, match="tipo de indice"):
        parse("CREATE INDEX i ON t (a) USING RTREE")


def test_drop_table():
    assert isinstance(parse("DROP TABLE t"), DropTable)


# ------------------------------------------------------------- robustez


def test_insensible_a_mayusculas():
    st = parse("select * from Empleados where Id = 1")
    assert st.table == "Empleados"
    assert st.where.column == "Id"


def test_ignora_comentarios_y_espacios():
    st = parse("""
        -- consulta de prueba
        SELECT *
        FROM   t
        WHERE  id = 1;   -- fin
    """)
    assert st.where.lo == 1


def test_punto_y_coma_opcional():
    assert parse("SELECT * FROM t;").table == "t"
    assert parse("SELECT * FROM t").table == "t"


def test_dos_sentencias_falla_con_mensaje_util():
    with pytest.raises(SQLSyntaxError, match="una sentencia a la vez"):
        parse("SELECT * FROM t; SELECT * FROM u")


def test_consulta_vacia_falla():
    with pytest.raises(SQLSyntaxError, match="vacia"):
        parse("   ")


def test_sentencia_incompleta_falla():
    with pytest.raises(SQLSyntaxError):
        parse("SELECT * FROM")


def test_caracter_invalido_falla():
    with pytest.raises(SQLSyntaxError, match="caracter inesperado"):
        parse("SELECT * FROM t WHERE id # 1")


def test_sentencia_desconocida_falla():
    with pytest.raises(SQLSyntaxError):
        parse("UPDATE t SET a = 1")
