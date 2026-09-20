import math

import pytest

from backend.storage.page import Page
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def rec(k):
    return S.pack((k, f"v{k}"))


def test_insert_sorted_mantiene_orden_con_entrada_desordenada():
    p = Page(1, 4096, S.record_size)
    for k in [50, 10, 30, 20, 40]:
        p.insert_sorted(rec(k), k, S)
    assert p.keys(S) == [10, 20, 30, 40, 50]
    assert p.record_count == 5
    assert all(p.is_occupied(i) for i in range(5))


def test_binary_search_encuentra_y_no_encuentra():
    p = Page(1, 4096, S.record_size)
    for k in [10, 20, 30]:
        p.insert_sorted(rec(k), k, S)
    assert p.binary_search(20, S) == (True, 1)
    assert p.binary_search(15, S) == (False, 1)  # posicion de insercion
    assert p.binary_search(5, S) == (False, 0)
    assert p.binary_search(99, S) == (False, 3)


def test_binary_search_en_pagina_vacia():
    assert Page(1, 4096, S.record_size).binary_search(1, S) == (False, 0)


def test_binary_search_con_un_solo_registro():
    p = Page(1, 4096, S.record_size)
    p.insert_sorted(rec(10), 10, S)
    assert p.binary_search(10, S) == (True, 0)
    assert p.binary_search(9, S) == (False, 0)
    assert p.binary_search(11, S) == (False, 1)


def test_delete_dense_compacta_a_la_izquierda():
    p = Page(1, 4096, S.record_size)
    for k in [10, 20, 30]:
        p.insert_sorted(rec(k), k, S)
    p.delete_dense(1)
    assert p.keys(S) == [10, 30]
    assert p.record_count == 2
    assert not p.is_occupied(2)


def test_delete_dense_del_ultimo_y_del_primero():
    p = Page(1, 4096, S.record_size)
    for k in [10, 20, 30]:
        p.insert_sorted(rec(k), k, S)
    p.delete_dense(2)
    assert p.keys(S) == [10, 20]
    p.delete_dense(0)
    assert p.keys(S) == [20]


def test_insertar_en_pagina_llena_falla():
    p = Page(1, 1024, S.record_size)
    for k in range(p.capacity):
        p.insert_sorted(rec(k), k, S)
    with pytest.raises(OverflowError):
        p.insert_sorted(rec(9999), 9999, S)


def test_insertar_en_medio_de_pagina_casi_llena():
    p = Page(1, 1024, S.record_size)
    for k in range(0, (p.capacity - 1) * 10, 10):
        p.insert_sorted(rec(k), k, S)
    p.insert_sorted(rec(55), 55, S)
    assert p.keys(S) == sorted(p.keys(S))
    assert 55 in p.keys(S)


def test_records_y_first_key():
    p = Page(1, 4096, S.record_size)
    for k in [30, 10, 20]:
        p.insert_sorted(rec(k), k, S)
    assert p.first_key(S) == 10
    assert [S.unpack(r) for r in p.records()] == [(10, "v10"), (20, "v20"), (30, "v30")]


def test_first_key_de_pagina_vacia_es_none():
    assert Page(1, 4096, S.record_size).first_key(S) is None


def test_low_key_round_trip_int():
    p = Page(1, 4096, S.record_size)
    p.set_low_key(1234, S)
    assert Page(1, 4096, S.record_size, data=p.to_bytes()).get_low_key(S) == 1234


def test_low_key_centinela_es_menos_infinito():
    p = Page(1, 4096, S.record_size)
    assert p.get_low_key(S) == -math.inf  # por defecto
    p.set_low_key(5, S)
    p.set_low_key(None, S)
    assert p.get_low_key(S) == -math.inf


def test_low_key_cero_no_se_confunde_con_el_centinela():
    """El bug clasico: usar 0 o -1 como valor magico en vez de una bandera."""
    p = Page(1, 4096, S.record_size)
    p.set_low_key(0, S)
    assert p.get_low_key(S) == 0
    assert p.get_low_key(S) != -math.inf


def test_low_key_negativo():
    p = Page(1, 4096, S.record_size)
    p.set_low_key(-99999, S)
    assert p.get_low_key(S) == -99999


def test_low_key_round_trip_float():
    sf = Schema([Column("k", ColumnType.FLOAT), Column("v", ColumnType.INT)])
    p = Page(1, 4096, sf.record_size)
    p.set_low_key(-2.5, sf)
    assert p.get_low_key(sf) == -2.5


def test_low_key_no_pisa_los_demas_campos_de_la_cabecera():
    p = Page(3, 4096, S.record_size)
    p.next_page_id = 11
    p.aux_page_id = 22
    p.record_count = 4
    p.set_low_key(777, S)
    q = Page(3, 4096, S.record_size, data=p.to_bytes())
    assert (q.next_page_id, q.aux_page_id, q.record_count) == (11, 22, 4)
    assert q.get_low_key(S) == 777
