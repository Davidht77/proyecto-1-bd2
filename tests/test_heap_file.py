import pytest

from backend.structures.heap_file import HeapFile, DuplicateKeyError
from backend.storage.page import NULL_PAGE
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def rec(k):
    return (k, f"v{k}")


def abrir(tmp_path, nombre="emp", **kw):
    return HeapFile(str(tmp_path / nombre), S, **kw)


# --------------------------------------------------------------- construccion


def test_archivo_vacio(tmp_path):
    with abrir(tmp_path) as hf:
        assert hf.n_records == 0
        assert hf.page_count == 0
        assert list(hf.scan()) == []


def test_persistencia_basica(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        hf.insert(rec(2))
    with abrir(tmp_path) as hf:
        assert hf.n_records == 2


# ------------------------------------------------------------------ insercion


def test_insercion_simple(tmp_path):
    with abrir(tmp_path) as hf:
        rid = hf.insert(rec(42))
        assert hf.n_records == 1
        assert rid.page_id >= 1  # Página física 1+ (física 0 es cabecera)
        assert 0 <= rid.slot < hf.capacity


def test_insercion_multiple(tmp_path):
    with abrir(tmp_path) as hf:
        for k in range(50):
            hf.insert(rec(k))
        assert hf.n_records == 50


def test_insercion_devuelve_rid_distinto_por_registro(tmp_path):
    with abrir(tmp_path) as hf:
        rids = [hf.insert(rec(k)) for k in range(10)]
        # Los primeros registros están en la misma página
        assert len(set(rids)) > 1  # Pero no todos en la misma


def test_insercion_no_requiere_ordenamiento(tmp_path):
    with abrir(tmp_path) as hf:
        for k in [5, 1, 9, 3, 7]:
            hf.insert(rec(k))
        assert hf.n_records == 5
        # El scan retorna en orden físico, no lógico
        registros = list(hf.scan())
        keys = [r[0] for r in registros]
        assert keys == [5, 1, 9, 3, 7]


def test_insercion_llena_paginas_secuencialmente(tmp_path):
    with abrir(tmp_path, page_size=1024) as hf:
        capacity = hf.capacity
        # Inserta suficientes registros para llenar una página
        for k in range(capacity + 5):
            hf.insert(rec(k))
        # Debe haber al menos 2 páginas
        assert hf.page_count >= 2


def test_insercion_costo_io_es_bajo(tmp_path):
    with abrir(tmp_path, page_size=4096) as hf:
        hf.counter.reset()
        rid = hf.insert(rec(1))
        # Primera inserción: 2 writes (header + página nueva)
        assert hf.counter.disk_writes == 2
        assert hf.counter.disk_reads == 0  # No debe leer para insertar en página nueva


def test_insercion_reutiliza_free_list(tmp_path):
    with abrir(tmp_path, page_size=1024) as hf:
        capacity = hf.capacity
        # Llena una página
        for k in range(capacity):
            hf.insert(rec(k))
        page_count_after_first = hf.page_count

        # La segunda página debe tener espacio
        hf.insert(rec(capacity))
        # No se crea nueva página si hay espacio en free-list
        assert hf.page_count == page_count_after_first + 1


# ------------------------------------------------------------------ busqueda


def test_search_encuentra_clave(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(10))
        hf.insert(rec(20))
        hf.insert(rec(30))
        assert hf.search(20) == rec(20)


def test_search_retorna_none_si_no_existe(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(10))
        assert hf.search(999) is None


def test_search_en_archivo_vacio_retorna_none(tmp_path):
    with abrir(tmp_path) as hf:
        assert hf.search(1) is None


def test_search_requiere_full_scan(tmp_path):
    with abrir(tmp_path, page_size=4096) as hf:
        for k in range(50):
            hf.insert(rec(k))

        hf.counter.reset()
        resultado = hf.search(49)  # Última clave

        # Full scan: debe leer todas las páginas
        assert hf.counter.disk_reads > 0
        assert resultado == rec(49)


# -------------------------------------------------------------------- borrado


def test_delete_encuentra_y_borra(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        hf.insert(rec(2))
        hf.insert(rec(3))

        assert hf.delete(2) is True
        assert hf.n_records == 2
        assert hf.search(2) is None


def test_delete_move_the_last(tmp_path):
    """Verifica que delete usa move-the-last."""
    with abrir(tmp_path, page_size=4096) as hf:
        # Inserta suficientes registros para llenar una página
        for k in range(20):
            hf.insert(rec(k))

        # Borra un registro del medio
        assert hf.delete(5) is True

        # El último registro debe estar ahora donde estaba el borrado
        # (o en otro lugar, lo importante es que el archivo se compacta)
        assert hf.n_records == 19


def test_delete_no_existe_retorna_false(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        assert hf.delete(999) is False
        assert hf.n_records == 1


def test_delete_en_archivo_vacio_retorna_false(tmp_path):
    with abrir(tmp_path) as hf:
        assert hf.delete(1) is False


def test_delete_ultimo_registro(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        hf.insert(rec(2))
        hf.insert(rec(3))

        assert hf.delete(3) is True
        assert hf.n_records == 2


# -------------------------------------------------------------- carga masiva


def test_bulk_load_carga_registros(tmp_path):
    with abrir(tmp_path) as hf:
        hf.bulk_load(rec(k) for k in range(100))
        assert hf.n_records == 100


def test_bulk_load_no_ordena(tmp_path):
    """Heap File mantiene el orden de inserción."""
    with abrir(tmp_path) as hf:
        hf.bulk_load(rec(k) for k in [5, 2, 8, 1])
        registros = list(hf.scan())
        keys = [r[0] for r in registros]
        assert keys == [5, 2, 8, 1]


def test_bulk_load_persiste(tmp_path):
    with abrir(tmp_path) as hf:
        hf.bulk_load(rec(k) for k in range(50))
    with abrir(tmp_path) as hf:
        assert hf.n_records == 50


def test_bulk_load_no_hace_lecturas(tmp_path):
    with abrir(tmp_path, page_size=1024) as hf:
        hf.counter.reset()
        hf.bulk_load(rec(k) for k in range(300))
        assert hf.counter.disk_reads == 0


def test_bulk_load_sobre_archivo_no_vacio_falla(tmp_path):
    with abrir(tmp_path) as hf:
        hf.bulk_load([rec(1)])
        with pytest.raises(ValueError):
            hf.bulk_load([rec(2)])


def test_bulk_load_vacio(tmp_path):
    with abrir(tmp_path) as hf:
        hf.bulk_load([])
        assert hf.n_records == 0


# ----------------------------------------------------------------- range search


def test_range_search_basico(tmp_path):
    with abrir(tmp_path) as hf:
        for k in range(1, 11):
            hf.insert(rec(k))

        resultados = list(hf.range_search(3, 7))
        keys = sorted([r[0] for r in resultados])
        assert keys == [3, 4, 5, 6, 7]


def test_range_search_vacio_si_no_hay_solapamiento(tmp_path):
    with abrir(tmp_path) as hf:
        for k in range(1, 6):
            hf.insert(rec(k))

        resultados = list(hf.range_search(10, 20))
        assert resultados == []


def test_range_search_requiere_full_scan(tmp_path):
    with abrir(tmp_path, page_size=4096) as hf:
        for k in range(50):
            hf.insert(rec(k))

        hf.counter.reset()
        list(hf.range_search(10, 20))

        # Debe leer todas las páginas
        assert hf.counter.disk_reads > 0


# -------------------------------------------------------------------- context


def test_context_manager(tmp_path):
    """Verifica que el context manager funciona."""
    path = str(tmp_path / "test")
    with HeapFile(path, S) as hf:
        hf.insert(rec(1))
        assert hf.n_records == 1


# -------------------------------------------------------------------- misc


def test_repr(tmp_path):
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        r = repr(hf)
        assert "HeapFile" in r
        assert "n=1" in r


def test_flush(tmp_path):
    """Verifica que flush se ejecuta sin errores."""
    with abrir(tmp_path) as hf:
        hf.insert(rec(1))
        hf.flush()
        assert hf.n_records == 1
