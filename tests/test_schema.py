import pytest

from backend.storage.schema import Column, ColumnType, Schema


def empleados() -> Schema:
    return Schema(
        [
            Column("id", ColumnType.INT),
            Column("nombre", ColumnType.CHAR, 30),
            Column("dept", ColumnType.CHAR, 20),
        ],
        pk_index=0,
    )


def test_record_size_es_la_suma_de_anchos():
    assert empleados().record_size == 54


def test_pack_unpack_round_trip():
    s = empleados()
    valores = (101, "Ada Lovelace", "Analytics")
    assert s.unpack(s.pack(valores)) == valores


def test_char_se_rellena_y_se_recorta():
    s = empleados()
    raw = s.pack((1, "ab", "cd"))
    assert len(raw) == 54
    assert s.unpack(raw) == (1, "ab", "cd")


def test_float_round_trip():
    s = Schema([Column("id", ColumnType.INT), Column("salario", ColumnType.FLOAT)])
    assert s.unpack(s.pack((7, 5200.5))) == (7, 5200.5)


def test_key_from_bytes_no_decodifica_el_registro_completo():
    s = empleados()
    raw = s.pack((999, "x", "y"))
    assert s.key_from_bytes(raw) == 999


def test_key_from_bytes_con_pk_no_inicial():
    s = Schema(
        [Column("nombre", ColumnType.CHAR, 10), Column("id", ColumnType.INT)],
        pk_index=1,
    )
    assert s.key_from_bytes(s.pack(("ada", 42))) == 42


def test_char_demasiado_largo_falla():
    with pytest.raises(ValueError):
        empleados().pack((1, "x" * 40, "y"))


def test_numero_de_valores_incorrecto_falla():
    with pytest.raises(ValueError):
        empleados().pack((1, "x"))


def test_key_of_usa_el_pk_index():
    s = Schema(
        [Column("nombre", ColumnType.CHAR, 10), Column("id", ColumnType.INT)],
        pk_index=1,
    )
    assert s.key_of(("ada", 42)) == 42


def test_pk_column_expone_el_tipo():
    assert empleados().pk_column.type is ColumnType.INT
