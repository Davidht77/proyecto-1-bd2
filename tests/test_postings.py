from backend.text.postings import PostingsFile

LISTA = [(p, p % 7, 1 + p % 3, 50 + p) for p in range(1000)]


def test_ida_y_vuelta(tmp_path):
    with PostingsFile(str(tmp_path / "x.spimi.post")) as pf:
        head = pf.write(LISTA)
        assert pf.read(head) == LISTA


def test_leer_cuesta_una_lectura_por_pagina(tmp_path):
    with PostingsFile(str(tmp_path / "x.spimi.post")) as pf:
        head = pf.write(LISTA)
        pf.pager.counter.reset()
        pf.read(head)
        assert pf.pager.counter.disk_reads == 4  # ceil(1000 / 252)


def test_varias_listas_en_un_archivo(tmp_path):
    with PostingsFile(str(tmp_path / "x.spimi.post")) as pf:
        a = pf.write(LISTA[:300])
        b = pf.write(LISTA[300:])
        assert pf.read(a) == LISTA[:300]
        assert pf.read(b) == LISTA[300:]


def test_persiste_al_reabrir(tmp_path):
    path = str(tmp_path / "x.spimi.post")
    with PostingsFile(path) as pf:
        head = pf.write(LISTA)
    with PostingsFile(path) as pf:
        assert pf.read(head) == LISTA
