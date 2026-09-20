import pytest

from backend.storage.page import NULL_PAGE, PageType
from backend.storage.pager import InvalidFileError, Pager
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def abrir(tmp_path, nombre="t.dat", **kw):
    return Pager(str(tmp_path / nombre), record_size=S.record_size, schema=S, **kw)


def test_archivo_nuevo_tiene_solo_la_cabecera(tmp_path):
    with abrir(tmp_path) as pg:
        assert pg.page_count == 1


def test_allocate_read_write_round_trip(tmp_path):
    with abrir(tmp_path) as pg:
        pid = pg.allocate_page(PageType.DATA)
        p = pg.read_page(pid)
        p.write_record(0, S.pack((42, "hola")))
        p.record_count = 1
        p.set_occupied(0, True)
        pg.write_page(p)
        assert S.unpack(pg.read_page(pid).read_record(0)) == (42, "hola")


def test_allocate_asigna_el_tipo(tmp_path):
    with abrir(tmp_path) as pg:
        pid = pg.allocate_page(PageType.OVERFLOW)
        assert pg.read_page(pid).page_type is PageType.OVERFLOW


def test_el_contador_cuenta_exactamente(tmp_path):
    with abrir(tmp_path) as pg:
        pid = pg.allocate_page(PageType.DATA)
        pg.counter.reset()
        pg.read_page(pid)
        pg.read_page(pid)
        pg.write_page(pg.read_page(pid))
        assert pg.counter.disk_reads == 3
        assert pg.counter.disk_writes == 1


def test_el_contador_registra_bytes(tmp_path):
    with abrir(tmp_path, page_size=2048) as pg:
        pid = pg.allocate_page(PageType.DATA)
        pg.counter.reset()
        pg.read_page(pid)
        assert pg.counter.bytes_read == 2048


def test_persistencia_tras_cerrar_y_reabrir(tmp_path):
    with abrir(tmp_path) as pg:
        pid = pg.allocate_page(PageType.DATA)
        p = pg.read_page(pid)
        p.record_count = 7
        pg.write_page(p)
        pg.user_a = 12345
    with abrir(tmp_path) as pg:
        assert pg.page_count == 2
        assert pg.read_page(pid).record_count == 7
        assert pg.user_a == 12345


def test_user_b_tambien_persiste(tmp_path):
    with abrir(tmp_path) as pg:
        pg.user_b = 777
    with abrir(tmp_path) as pg:
        assert pg.user_b == 777


def test_free_page_reusa_el_id(tmp_path):
    with abrir(tmp_path) as pg:
        a = pg.allocate_page(PageType.DATA)
        pg.free_page(a)
        assert pg.allocate_page(PageType.DATA) == a
        assert pg.page_count == 2  # no crecio el archivo


def test_free_list_es_lifo_y_persiste(tmp_path):
    with abrir(tmp_path) as pg:
        a, b = pg.allocate_page(PageType.DATA), pg.allocate_page(PageType.DATA)
        pg.free_page(a)
        pg.free_page(b)
    with abrir(tmp_path) as pg:
        assert pg.allocate_page(PageType.DATA) == b
        assert pg.allocate_page(PageType.DATA) == a


def test_magic_invalido_falla(tmp_path):
    f = tmp_path / "malo.dat"
    f.write_bytes(b"X" * 4096)
    with pytest.raises(InvalidFileError):
        Pager(str(f), record_size=S.record_size, schema=S)


def test_page_size_discrepante_falla(tmp_path):
    with abrir(tmp_path, page_size=4096):
        pass
    with pytest.raises(InvalidFileError):
        abrir(tmp_path, page_size=8192)


def test_record_size_discrepante_falla(tmp_path):
    with abrir(tmp_path):
        pass
    with pytest.raises(InvalidFileError):
        Pager(str(tmp_path / "t.dat"), record_size=99, schema=S)


def test_leer_pagina_inexistente_falla(tmp_path):
    with abrir(tmp_path) as pg:
        with pytest.raises(IndexError):
            pg.read_page(99)


def test_page_id_se_valida_al_leer(tmp_path):
    """La pagina lleva su propio id en disco: detecta errores de offset."""
    with abrir(tmp_path) as pg:
        pid = pg.allocate_page(PageType.DATA)
        assert pg.read_page(pid).page_id == pid
        assert pg.read_page(pid).stored_page_id == pid


def test_la_pagina_0_es_la_cabecera_y_no_se_puede_liberar(tmp_path):
    with abrir(tmp_path) as pg:
        with pytest.raises(ValueError):
            pg.free_page(0)


@pytest.mark.parametrize("B", [1024, 2048, 4096, 8192])
def test_funciona_con_todos_los_tamanos_de_bloque(tmp_path, B):
    with abrir(tmp_path, nombre=f"b{B}.dat", page_size=B) as pg:
        pid = pg.allocate_page(PageType.DATA)
        p = pg.read_page(pid)
        p.write_record(0, S.pack((1, "x")))
        pg.write_page(p)
        assert S.unpack(pg.read_page(pid).read_record(0)) == (1, "x")
        import os

        assert os.path.getsize(str(tmp_path / f"b{B}.dat")) == 2 * B


def test_el_esquema_se_recupera_del_archivo(tmp_path):
    """El binario es autodescriptivo: util para la demo con un visor hexadecimal."""
    with abrir(tmp_path):
        pass
    with Pager(str(tmp_path / "t.dat"), record_size=S.record_size) as pg:
        leido = pg.schema
        assert [c.name for c in leido.columns] == ["id", "v"]
        assert leido.record_size == S.record_size
        assert leido.pk_index == 0
