"""Convierte las afirmaciones teoricas del informe en tests ejecutables.

Si el codigo deja de cumplir la cota de I/O que el informe afirma, estos tests
fallan. Es la diferencia entre "creemos que cuesta log2(P)" y "lo medimos".
"""

import math

from backend.organization.sequential_file import SequentialFile
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def rec(k):
    return (k, f"v{k}")


def test_la_busqueda_sin_overflow_esta_acotada_por_log2_P(tmp_path):
    sf = SequentialFile(str(tmp_path / "m"), S, page_size=1024, cache_size=0)
    sf.bulk_load(rec(k) for k in range(20_000))
    P = sf.page_count
    esperado = math.ceil(math.log2(P))
    for k in [0, 137, 9_999, 19_999]:
        sf.counter.reset()
        assert sf.search(k) == rec(k)
        assert sf.counter.disk_reads <= esperado, f"clave {k}: {sf.counter.disk_reads} > {esperado}"
    sf.close()


def test_la_busqueda_fallida_tambien_esta_acotada(tmp_path):
    sf = SequentialFile(str(tmp_path / "mf"), S, page_size=1024, cache_size=0)
    sf.bulk_load(rec(k) for k in range(0, 40_000, 2))  # solo pares
    esperado = math.ceil(math.log2(sf.page_count))
    for k in [1, 9_999, 39_999]:
        sf.counter.reset()
        assert sf.search(k) is None
        assert sf.counter.disk_reads <= esperado
    sf.close()


def test_el_overflow_degrada_el_costo_y_la_reorganizacion_lo_recupera(tmp_path):
    sf = SequentialFile(
        str(tmp_path / "d"), S, page_size=1024, fill_factor=1.0, cache_size=0
    )
    sf.bulk_load(rec(k) for k in range(0, 20_000, 10))
    sf.counter.reset()
    sf.search(1000)
    limpio = sf.counter.disk_reads

    for k in range(1001, 1010):
        sf.insert(rec(k))
    sf.counter.reset()
    assert sf.search(1009) == rec(1009)
    degradado = sf.counter.disk_reads
    assert degradado > limpio, "el overflow debe costar lecturas extra"

    sf.reorganize()
    sf.counter.reset()
    assert sf.search(1009) == rec(1009)
    assert sf.counter.disk_reads <= limpio + 1, "la reorganizacion debe recuperar el costo"
    sf.close()


def test_el_buffer_pool_reduce_las_lecturas(tmp_path):
    def lecturas(cache):
        sf = SequentialFile(
            str(tmp_path / f"b{cache}"), S, page_size=1024, cache_size=cache
        )
        sf.bulk_load(rec(k) for k in range(20_000))
        sf.counter.reset()
        for k in range(0, 2000, 7):
            sf.search(k)
        n = sf.counter.disk_reads
        sf.close()
        return n

    con_cache, sin_cache = lecturas(64), lecturas(0)
    assert con_cache < sin_cache


def test_el_scan_completo_lee_cada_pagina_una_sola_vez(tmp_path):
    """Full Table Scan: coste P lecturas, la base del Experimento 2."""
    sf = SequentialFile(str(tmp_path / "s"), S, page_size=1024, cache_size=0)
    sf.bulk_load(rec(k) for k in range(20_000))
    P = sf.page_count
    sf.counter.reset()
    assert sum(1 for _ in sf.scan()) == 20_000
    assert sf.counter.disk_reads == P
    sf.close()


def test_el_fan_out_crece_con_el_tamano_de_bloque(tmp_path):
    """Datos crudos del Experimento 4: sensibilidad al tamano de bloque."""
    from backend.storage.page import page_capacity

    capacidades = [page_capacity(B, S.record_size) for B in (1024, 2048, 4096, 8192)]
    assert capacidades == sorted(capacidades)
    # Duplicar el bloque casi duplica la capacidad: el overhead fijo de la
    # cabecera de 32B se amortiza cada vez mejor.
    for chico, grande in zip(capacidades, capacidades[1:]):
        # Es algo MAS del doble: C(2B)/C(B) = (2B-32)/(B-32) > 2, porque la
        # cabecera fija de 32B se amortiza mejor en el bloque grande.
        assert 2.0 <= grande / chico < 2.1


def test_menos_paginas_con_bloques_mayores(tmp_path):
    anteriores = None
    for B in (1024, 2048, 4096, 8192):
        sf = SequentialFile(str(tmp_path / f"fb{B}"), S, page_size=B, cache_size=0)
        sf.bulk_load(rec(k) for k in range(20_000))
        if anteriores is not None:
            assert sf.page_count < anteriores
        anteriores = sf.page_count
        sf.close()
