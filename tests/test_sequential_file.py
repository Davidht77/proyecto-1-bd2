import math
import random

import pytest

from backend.structures.sequential_file import (
    DuplicateKeyError,
    SequentialFile,
    UnsupportedKeyType,
)
from backend.storage.page import NULL_PAGE
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def rec(k):
    return (k, f"v{k}")


def abrir(tmp_path, nombre="emp", **kw):
    return SequentialFile(str(tmp_path / nombre), S, **kw)


# --------------------------------------------------------------- construccion


def test_pk_char_no_soportada(tmp_path):
    bad = Schema([Column("cod", ColumnType.CHAR, 10), Column("n", ColumnType.INT)])
    with pytest.raises(UnsupportedKeyType):
        SequentialFile(str(tmp_path / "x"), bad)


def test_archivo_vacio(tmp_path):
    with abrir(tmp_path) as sf:
        assert sf.n_records == 0
        assert sf.page_count == 0
        assert list(sf.scan()) == []


# ------------------------------------------------------------------ bulk_load


def test_bulk_load_ordena_y_el_scan_sale_ordenado(tmp_path):
    with abrir(tmp_path) as sf:
        sf.bulk_load(rec(k) for k in [30, 10, 50, 20, 40])
        assert list(sf.scan()) == [rec(k) for k in [10, 20, 30, 40, 50]]
        assert sf.n_records == 5


def test_bulk_load_respeta_el_fill_factor(tmp_path):
    with abrir(tmp_path, page_size=1024, fill_factor=0.75) as sf:
        sf.bulk_load(rec(k) for k in range(300))
        por_pagina = int(sf._capacity * 0.75)
        assert sf.page_count == math.ceil(300 / por_pagina)
        for p in range(sf.page_count):
            assert sf._read_main(p).record_count <= por_pagina


def test_low_key_de_la_primera_pagina_es_menos_infinito(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(300))
        assert sf._read_main(0).get_low_key(S) == -math.inf
        assert sf._read_main(1).get_low_key(S) > -math.inf


def test_bulk_load_persiste(tmp_path):
    with abrir(tmp_path) as sf:
        sf.bulk_load(rec(k) for k in range(100))
    with abrir(tmp_path) as sf:
        assert sf.n_records == 100
        assert list(sf.scan()) == [rec(k) for k in range(100)]


def test_bulk_load_no_hace_lecturas(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.counter.reset()
        sf.bulk_load(rec(k) for k in range(300))
        assert sf.counter.disk_reads == 0


def test_bulk_load_sobre_archivo_no_vacio_falla(tmp_path):
    with abrir(tmp_path) as sf:
        sf.bulk_load([rec(1)])
        with pytest.raises(ValueError):
            sf.bulk_load([rec(2)])


# --------------------------------------------------------------- localizacion


def test_locate_page_devuelve_la_ultima_con_low_key_menor_o_igual(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(300))
        assert sf._locate_page(-5) == 0  # por debajo de todo => pagina 0
        assert sf._locate_page(0) == 0
        ultima = sf.page_count - 1
        assert sf._locate_page(9999) == ultima  # por encima => ultima pagina
        for k in range(0, 300, 7):
            p = sf._locate_page(k)
            assert sf._read_main(p).get_low_key(S) <= k
            if p + 1 < sf.page_count:
                assert sf._read_main(p + 1).get_low_key(S) > k


# -------------------------------------------------------------------- busqueda


def test_search_encuentra_todo_lo_cargado(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(300))
        for k in range(300):
            assert sf.search(k) == rec(k)


def test_search_devuelve_none_si_no_existe(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 300, 2))  # solo pares
        assert sf.search(151) is None
        assert sf.search(-1) is None
        assert sf.search(9999) is None


def test_search_en_archivo_vacio(tmp_path):
    with abrir(tmp_path) as sf:
        assert sf.search(1) is None


# ------------------------------------------------------------------- insercion


def test_insert_en_archivo_vacio(tmp_path):
    with abrir(tmp_path) as sf:
        sf.insert(rec(1))
        assert sf.search(1) == rec(1)
        assert sf.n_records == 1


@pytest.mark.parametrize("orden", ["asc", "desc", "aleatorio"])
def test_insert_recupera_todo_en_cualquier_orden(tmp_path, orden):
    claves = list(range(500))
    if orden == "desc":
        claves.reverse()
    if orden == "aleatorio":
        random.Random(42).shuffle(claves)
    with abrir(tmp_path, nombre=orden, page_size=1024) as sf:
        for k in claves:
            sf.insert(rec(k))
        assert sf.n_records == 500
        for k in claves:
            assert sf.search(k) == rec(k)
        assert list(sf.scan()) == [rec(k) for k in range(500)]


def test_las_inserciones_van_a_overflow_cuando_la_pagina_se_llena(tmp_path):
    with abrir(tmp_path, page_size=1024, auto_reorganize=False) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))  # huecos de 10
        paginas_antes = sf.page_count
        assert sf.n_overflow == 0
        for k in range(1, 400):
            if k % 10:
                sf.insert(rec(k))
        assert sf.n_overflow > 0
        assert sf.page_count == paginas_antes  # el area principal no crece


def test_clave_duplicada_falla(tmp_path):
    with abrir(tmp_path) as sf:
        sf.insert(rec(1))
        with pytest.raises(DuplicateKeyError):
            sf.insert(rec(1))


def test_duplicado_detectado_en_overflow(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))
        sf.insert(rec(5))
        with pytest.raises(DuplicateKeyError):
            sf.insert(rec(5))


def test_la_cadena_de_overflow_queda_ordenada(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))
        pendientes = [k for k in range(1, 400) if k % 10]
        random.Random(7).shuffle(pendientes)
        for k in pendientes:
            sf.insert(rec(k))
        for p in range(sf.page_count):
            cadena = list(sf._chain_pages(sf._read_main(p).aux_page_id))
            claves = [k for pg in cadena for k in pg.keys(S)]
            assert claves == sorted(claves)


def test_las_inserciones_persisten(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 1000, 10))
        for k in [1, 2, 3, 501, 502]:
            sf.insert(rec(k))
        esperado = list(sf.scan())
    with abrir(tmp_path, page_size=1024) as sf:
        assert list(sf.scan()) == esperado
        assert sf.search(502) == rec(502)


# ----------------------------------------------------------------------- rango


def cargado(tmp_path, n=500, nombre="rango"):
    sf = abrir(tmp_path, nombre=nombre, page_size=1024)
    sf.bulk_load(rec(k) for k in range(n))
    return sf


def test_rango_basico(tmp_path):
    with cargado(tmp_path) as sf:
        assert list(sf.range_search(100, 110)) == [rec(k) for k in range(100, 111)]


def test_rango_con_limites_inclusivos(tmp_path):
    with cargado(tmp_path) as sf:
        assert list(sf.range_search(7, 7)) == [rec(7)]


def test_rango_vacio_y_fuera_de_rango(tmp_path):
    with cargado(tmp_path) as sf:
        assert list(sf.range_search(10, 5)) == []
        assert list(sf.range_search(9000, 9999)) == []
        assert list(sf.range_search(-100, -1)) == []


def test_rango_total_equivale_al_scan(tmp_path):
    with cargado(tmp_path) as sf:
        assert list(sf.range_search(-(10**9), 10**9)) == list(sf.scan())


def test_el_rango_incluye_registros_en_overflow(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))
        for k in [101, 102, 103]:
            sf.insert(rec(k))
        assert list(sf.range_search(100, 110)) == [
            rec(k) for k in [100, 101, 102, 103, 110]
        ]


def test_el_rango_cruza_varias_paginas(tmp_path):
    with cargado(tmp_path, n=500, nombre="cruza") as sf:
        assert list(sf.range_search(5, 495)) == [rec(k) for k in range(5, 496)]


# --------------------------------------------------------------------- borrado


def test_delete_del_area_principal(tmp_path):
    with cargado(tmp_path, nombre="del1") as sf:
        assert sf.delete(250) is True
        assert sf.search(250) is None
        assert sf.n_records == 499
        assert sf.delete(250) is False


def test_delete_de_overflow_y_liberacion_de_pagina(tmp_path):
    # fill_factor=1.0 deja las paginas principales llenas, asi que toda
    # insercion posterior va forzosamente a overflow.
    with abrir(tmp_path, page_size=1024, fill_factor=1.0) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))
        for k in [101, 102, 103]:
            sf.insert(rec(k))
        assert sf.n_overflow == 3
        p = sf._locate_page(101)
        for k in [101, 102, 103]:
            assert sf.delete(k) is True
        assert sf.n_overflow == 0
        assert sf._read_main(p).aux_page_id == NULL_PAGE


def test_borrar_el_primer_registro_no_pierde_su_overflow(tmp_path):
    """Regresion del bug de diseno: low_key no debe depender del contenido."""
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(0, 2000, 10))
        p = sf._locate_page(500)
        primera = sf._read_main(p).first_key(S)
        sf.insert(rec(primera + 1))
        assert sf.delete(primera) is True
        assert sf.search(primera + 1) == rec(primera + 1)


def test_vaciar_una_pagina_entera_no_rompe_la_busqueda(tmp_path):
    with abrir(tmp_path, page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(300))
        p = 2
        claves = sf._read_main(p).keys(S)
        for k in claves:
            assert sf.delete(k) is True
        assert sf._read_main(p).record_count == 0
        for k in claves:
            assert sf.search(k) is None
        assert sf.search(0) == rec(0)
        assert sf.search(299) == rec(299)


def test_delete_deja_el_scan_ordenado(tmp_path):
    with cargado(tmp_path, nombre="del2") as sf:
        for k in range(0, 500, 3):
            sf.delete(k)
        restantes = [k for k in range(500) if k % 3]
        assert list(sf.scan()) == [rec(k) for k in restantes]


def test_delete_de_clave_inexistente(tmp_path):
    with cargado(tmp_path, nombre="del3") as sf:
        assert sf.delete(99999) is False
        assert sf.delete(-1) is False
        assert sf.n_records == 500


# -------------------------------------------------------------- reorganizacion


def con_overflow(tmp_path, nombre="reorg"):
    sf = abrir(tmp_path, nombre=nombre, page_size=1024)
    sf.bulk_load(rec(k) for k in range(0, 4000, 10))
    for k in range(1, 800):
        if k % 10:
            sf.insert(rec(k))
    assert sf.n_overflow > 0
    return sf


def test_reorganize_conserva_todos_los_registros(tmp_path):
    with con_overflow(tmp_path, "r1") as sf:
        antes = list(sf.scan())
        sf.reorganize()
        assert list(sf.scan()) == antes
        assert sf.n_records == len(antes)


def test_reorganize_vacia_el_overflow(tmp_path):
    with con_overflow(tmp_path, "r2") as sf:
        sf.reorganize()
        assert sf.n_overflow == 0
        for p in range(sf.page_count):
            assert sf._read_main(p).aux_page_id == NULL_PAGE


def test_reorganize_respeta_el_fill_factor(tmp_path):
    with con_overflow(tmp_path, "r3") as sf:
        sf.reorganize()
        tope = int(sf._capacity * sf.fill_factor)
        for p in range(sf.page_count):
            assert sf._read_main(p).record_count <= tope


def test_reorganize_es_idempotente(tmp_path):
    with con_overflow(tmp_path, "r4") as sf:
        sf.reorganize()
        estado = (list(sf.scan()), sf.page_count)
        sf.reorganize()
        assert (list(sf.scan()), sf.page_count) == estado


def test_reorganize_deja_todo_buscable(tmp_path):
    with con_overflow(tmp_path, "r5") as sf:
        claves = [k for k, _ in sf.scan()]
        sf.reorganize()
        for k in claves:
            assert sf.search(k) == rec(k)


def test_reorganize_reporta_estadisticas(tmp_path):
    with con_overflow(tmp_path, "r6") as sf:
        st = sf.reorganize()
        assert st.records == sf.n_records
        assert st.pages_before > 0 and st.pages_after > 0
        assert st.disk_writes > 0 and st.elapsed_ms >= 0
        assert st.overflow_pages_freed > 0


def test_reorganize_persiste(tmp_path):
    with con_overflow(tmp_path, "r7") as sf:
        sf.reorganize()
        esperado = list(sf.scan())
    with abrir(tmp_path, nombre="r7", page_size=1024) as sf:
        assert list(sf.scan()) == esperado
        assert sf.n_overflow == 0


def test_reorganize_de_archivo_vacio_no_falla(tmp_path):
    with abrir(tmp_path, nombre="r8") as sf:
        assert sf.reorganize().records == 0


def test_se_puede_insertar_despues_de_reorganizar(tmp_path):
    with con_overflow(tmp_path, "r9") as sf:
        sf.reorganize()
        sf.insert(rec(99999))
        assert sf.search(99999) == rec(99999)
        assert list(sf.scan())[-1] == rec(99999)


# ------------------------------------------- reorganizacion automatica (cap K)


def test_el_cap_de_overflow_escala_con_log2_P(tmp_path):
    """K = (C/2) * ceil(log2(P)).

    El factor 1/2 sale del split 50/50: las paginas de overflow quedan entre el
    50% y el 100% ocupadas, asi que K registros ocupan hasta 2K/C paginas.
    """
    with abrir(tmp_path, nombre="k1", page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(20_000))
        assert sf.overflow_cap == (sf._capacity // 2) * math.ceil(math.log2(sf.page_count))

    with abrir(tmp_path, nombre="k2", page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(200))
        chico = sf.overflow_cap
    with abrir(tmp_path, nombre="k3", page_size=1024) as sf:
        sf.bulk_load(rec(k) for k in range(20_000))
        assert sf.overflow_cap > chico


def test_el_cap_es_valido_con_archivo_vacio_o_de_una_pagina(tmp_path):
    with abrir(tmp_path, nombre="k4") as sf:
        assert sf.overflow_cap >= 1
        sf.insert(rec(1))
        assert sf.overflow_cap >= 1


def test_no_reorganiza_por_debajo_del_cap(tmp_path):
    with abrir(tmp_path, nombre="k5", page_size=1024, fill_factor=1.0) as sf:
        sf.bulk_load(rec(k) for k in range(0, 20_000, 10))
        paginas = sf.page_count
        for k in range(1, 1 + sf.overflow_cap // 2):
            if k % 10:
                sf.insert(rec(k))
        assert sf.n_overflow > 0, "deberia haber overflow acumulado"
        assert sf.page_count == paginas, "no debio reorganizar todavia"


def test_reorganiza_al_superar_el_cap(tmp_path):
    with abrir(tmp_path, nombre="k6", page_size=1024, fill_factor=1.0) as sf:
        sf.bulk_load(rec(k) for k in range(0, 40_000, 10))
        cap_inicial = sf.overflow_cap
        insertadas, pico, hubo_reorg = [], 0, False
        k = 1
        while len(insertadas) < cap_inicial * 3:
            if k % 10:
                sf.insert(rec(k))
                insertadas.append(k)
                if sf.n_overflow < pico:
                    hubo_reorg = True
                pico = max(pico, sf.n_overflow)
            k += 1
        assert hubo_reorg, "debio dispararse al menos una reorganizacion automatica"
        assert pico <= cap_inicial + 1, f"el overflow llego a {pico}, cap {cap_inicial}"
        for j in insertadas:
            assert sf.search(j) == rec(j), f"la reorganizacion perdio {j}"


def test_sin_auto_reorganize_el_overflow_crece_sin_limite(tmp_path):
    """El Experimento 1 compara el Sequential File con y sin reorganizacion."""
    with abrir(
        tmp_path, nombre="k8", page_size=1024, fill_factor=1.0, auto_reorganize=False
    ) as sf:
        sf.bulk_load(rec(k) for k in range(0, 40_000, 10))
        cap = sf.overflow_cap
        k = 1
        while sf.n_overflow <= cap + 50:
            if k % 10:
                sf.insert(rec(k))
            k += 1
        assert sf.n_overflow > cap, "sin auto_reorganize nada debe frenar el crecimiento"


def test_la_busqueda_queda_acotada_bajo_insercion_monotona_sostenida(tmp_path):
    """El peor caso del Sequential File: claves siempre crecientes.

    Sin cap, todas las inserciones caen en la ultima pagina y su cadena crece
    sin limite. Con K = C*log2(P), la busqueda queda acotada por 2*log2(P).
    """
    with abrir(tmp_path, nombre="k7", page_size=1024, fill_factor=1.0) as sf:
        N = 20_000
        sf.bulk_load(rec(k) for k in range(N))
        for k in range(N, N + 4_000):
            sf.insert(rec(k))
        cota = 2 * math.ceil(math.log2(sf.page_count))
        for k in range(N + 3_900, N + 4_000):
            sf.counter.reset()
            assert sf.search(k) == rec(k)
            assert sf.counter.disk_reads <= cota, (
                f"clave {k}: {sf.counter.disk_reads} lecturas > cota {cota}"
            )
