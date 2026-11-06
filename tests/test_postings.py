from backend.text.postings import PostingsFile

LISTA = [(p, p % 7, 1 + p % 3, 50 + p) for p in range(1000)]


def test_ida_y_vuelta(tmp_path):
    with PostingsFile(str(tmp_path / "x.spimi.post")) as pf:
        start = pf.write(LISTA)
        assert pf.read(start, len(LISTA)) == LISTA


def test_leer_cuesta_una_lectura_por_pagina(tmp_path):
    path = str(tmp_path / "x.spimi.post")
    with PostingsFile(path) as pf:
        pf.write(LISTA)
    with PostingsFile(path) as pf:
        pf.pager.counter.reset()  # abrir lee la cabecera; aqui se mide solo la lista
        pf.read(0, len(LISTA))
        assert pf.pager.counter.disk_reads == 4  # ceil(1000 / 252)


def test_listas_chicas_van_seguidas_en_la_misma_pagina(tmp_path):
    with PostingsFile(str(tmp_path / "x.spimi.post")) as pf:
        starts = [pf.write(LISTA[i : i + 10]) for i in range(0, 50, 10)]
        assert starts == [0, 10, 20, 30, 40]
        assert pf.pager.page_count == 2  # cabecera + 1 pagina para las 5 listas
        assert [pf.read(s, 10) for s in starts] == [LISTA[i : i + 10] for i in range(0, 50, 10)]


def test_lista_que_cruza_el_borde_de_pagina_cuesta_2_lecturas(tmp_path):
    path = str(tmp_path / "x.spimi.post")
    with PostingsFile(path) as pf:
        pf.write(LISTA[:250])
        start = pf.write(LISTA[250:260])  # posiciones 250..259; la pagina 1 termina en 251
    with PostingsFile(path) as pf:
        pf.pager.counter.reset()
        assert pf.read(start, 10) == LISTA[250:260]
        assert pf.pager.counter.disk_reads == 2


def test_persiste_y_se_puede_seguir_escribiendo_al_reabrir(tmp_path):
    path = str(tmp_path / "x.spimi.post")
    with PostingsFile(path) as pf:
        a = pf.write(LISTA[:300])
    with PostingsFile(path) as pf:
        b = pf.write(LISTA[300:])
    with PostingsFile(path) as pf:
        assert (a, b, len(pf)) == (0, 300, 1000)
        assert pf.read(a, 300) + pf.read(b, 700) == LISTA
