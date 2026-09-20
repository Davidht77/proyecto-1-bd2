import pytest

from backend.storage.page import (
    HEADER_SIZE,
    NULL_PAGE,
    Page,
    PageType,
    bitmap_size,
    page_capacity,
)


# Calculados a mano con C = (8*(B-32)) // (8*54+1) = (8*(B-32)) // 433
@pytest.mark.parametrize(
    "B,R,esperado",
    [(1024, 54, 18), (2048, 54, 37), (4096, 54, 75), (8192, 54, 150)],
)
def test_capacidad_por_tamano_de_bloque(B, R, esperado):
    assert page_capacity(B, R) == esperado


@pytest.mark.parametrize("B", [1024, 2048, 4096, 8192])
def test_la_capacidad_siempre_cabe_en_la_pagina(B):
    R = 54
    C = page_capacity(B, R)
    assert HEADER_SIZE + bitmap_size(C) + C * R <= B
    # y una mas no cabria
    assert HEADER_SIZE + bitmap_size(C + 1) + (C + 1) * R > B


def test_la_cabecera_mide_exactamente_32_bytes():
    assert HEADER_SIZE == 32


def test_round_trip_de_cabecera():
    p = Page(7, 4096, 54)
    p.page_type = PageType.OVERFLOW
    p.record_count = 3
    p.next_page_id = 9
    p.prev_page_id = NULL_PAGE
    p.aux_page_id = 12
    q = Page(7, 4096, 54, data=p.to_bytes())
    assert (q.page_type, q.record_count) == (PageType.OVERFLOW, 3)
    assert (q.next_page_id, q.prev_page_id, q.aux_page_id) == (9, NULL_PAGE, 12)
    assert q.page_id == 7


def test_punteros_nuevos_son_nulos():
    p = Page(1, 4096, 54)
    assert p.next_page_id == p.prev_page_id == p.aux_page_id == NULL_PAGE
    assert p.record_count == 0


def test_bitmap_marca_slots_independientes():
    p = Page(1, 4096, 54)
    p.set_occupied(0, True)
    p.set_occupied(9, True)
    assert p.is_occupied(0) and p.is_occupied(9)
    assert not p.is_occupied(1) and not p.is_occupied(8)
    p.set_occupied(0, False)
    assert not p.is_occupied(0) and p.is_occupied(9)


def test_first_free_slot_y_pagina_llena():
    p = Page(1, 1024, 54)
    for s in range(p.capacity):
        assert p.first_free_slot() == s
        p.set_occupied(s, True)
        p.record_count += 1
    assert p.first_free_slot() is None
    assert p.is_full()


def test_registros_en_slots_distintos_no_se_pisan():
    p = Page(1, 4096, 54)
    p.write_record(0, b"A" * 54)
    p.write_record(3, b"B" * 54)
    assert p.read_record(0) == b"A" * 54
    assert p.read_record(3) == b"B" * 54


def test_slot_fuera_de_rango_falla():
    p = Page(1, 1024, 54)
    with pytest.raises(IndexError):
        p.read_record(p.capacity)
    with pytest.raises(IndexError):
        p.write_record(p.capacity, b"x" * 54)


def test_registro_de_tamano_incorrecto_falla():
    p = Page(1, 1024, 54)
    with pytest.raises(ValueError):
        p.write_record(0, b"corto")


def test_to_bytes_devuelve_exactamente_page_size():
    for B in (1024, 2048, 4096, 8192):
        assert len(Page(1, B, 54).to_bytes()) == B
