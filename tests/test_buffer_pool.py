from backend.storage.page import PageType
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def abrir(tmp_path, nombre="t.dat", **kw):
    return Pager(str(tmp_path / nombre), record_size=S.record_size, schema=S, **kw)


def test_sin_cache_cada_lectura_es_un_io(tmp_path):
    with abrir(tmp_path, cache_size=0) as pg:
        pid = pg.allocate_page(PageType.DATA)
        pg.counter.reset()
        for _ in range(5):
            pg.read_page(pid)
        assert pg.counter.disk_reads == 5


def test_con_cache_la_relectura_no_cuenta(tmp_path):
    with abrir(tmp_path, cache_size=4) as pg:
        pid = pg.allocate_page(PageType.DATA)
        pg.invalidate_cache()  # arranca desde cache fria
        pg.counter.reset()
        for _ in range(5):
            pg.read_page(pid)
        assert pg.counter.disk_reads == 1


def test_evicion_lru_expulsa_la_menos_usada(tmp_path):
    with abrir(tmp_path, cache_size=2) as pg:
        a = pg.allocate_page(PageType.DATA)
        b = pg.allocate_page(PageType.DATA)
        c = pg.allocate_page(PageType.DATA)
        pg.invalidate_cache()
        pg.read_page(a)
        pg.read_page(b)
        pg.read_page(a)  # 'a' pasa a ser la mas reciente
        pg.read_page(c)  # expulsa 'b'
        pg.counter.reset()
        pg.read_page(a)
        assert pg.counter.disk_reads == 0
        pg.read_page(b)
        assert pg.counter.disk_reads == 1


def test_write_back_difiere_la_escritura_hasta_flush(tmp_path):
    with abrir(tmp_path, cache_size=4) as pg:
        pid = pg.allocate_page(PageType.DATA)
        p = pg.read_page(pid)
        p.record_count = 3
        pg.counter.reset()
        pg.write_page(p)
        assert pg.counter.disk_writes == 0  # queda sucia en la cache
        pg.flush()
        assert pg.counter.disk_writes >= 1


def test_los_datos_sobreviven_al_cierre_con_cache(tmp_path):
    with abrir(tmp_path, cache_size=4) as pg:
        pid = pg.allocate_page(PageType.DATA)
        p = pg.read_page(pid)
        p.record_count = 9
        pg.write_page(p)
    with abrir(tmp_path, cache_size=0) as pg:
        assert pg.read_page(pid).record_count == 9


def test_una_pagina_sucia_expulsada_se_escribe(tmp_path):
    """Si el write-back perdiera la victima sucia, los datos se corromperian."""
    with abrir(tmp_path, cache_size=2) as pg:
        ids = [pg.allocate_page(PageType.DATA) for _ in range(6)]
        for i, pid in enumerate(ids):
            p = pg.read_page(pid)
            p.record_count = i + 1
            pg.write_page(p)
    with abrir(tmp_path, cache_size=0) as pg:
        for i, pid in enumerate(ids):
            assert pg.read_page(pid).record_count == i + 1


def test_el_pool_reporta_aciertos_y_fallos(tmp_path):
    with abrir(tmp_path, cache_size=4) as pg:
        pid = pg.allocate_page(PageType.DATA)
        pg.invalidate_cache()
        pg.read_page(pid)
        pg.read_page(pid)
        assert pg.pool.hits == 1
        assert pg.pool.misses == 1
