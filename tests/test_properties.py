"""Pruebas de propiedades con datos aleatorios.

Los tests de casos concretos cubren lo que se me ocurrio; estos cubren lo que
no. Cada uno ejercita el ciclo completo carga -> insercion -> reorganizacion ->
borrado y comprueba los mismos invariantes en cada etapa.
"""

import random

from backend.organization.sequential_file import SequentialFile
from backend.storage.schema import Column, ColumnType, Schema

S = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 10)])


def rec(k):
    return (k, f"v{k}")


def test_5000_claves_aleatorias_sobreviven_a_todo(tmp_path):
    rng = random.Random(1234)
    claves = rng.sample(range(1_000_000), 5_000)
    sf = SequentialFile(str(tmp_path / "p"), S, page_size=1024, cache_size=0)
    sf.bulk_load(rec(k) for k in claves[:1000])
    for k in claves[1000:]:
        sf.insert(rec(k))

    def invariantes(etapa):
        vistos = [k for k, _ in sf.scan()]
        assert vistos == sorted(vistos), f"{etapa}: el scan debe salir ordenado"
        assert set(vistos) == set(claves), f"{etapa}: faltan o sobran registros"
        for k in rng.sample(claves, 200):
            assert sf.search(k) == rec(k), f"{etapa}: no encuentra {k}"

    invariantes("tras insertar")
    sf.reorganize()
    invariantes("tras reorganizar")

    borradas = set(rng.sample(claves, 1000))
    for k in borradas:
        assert sf.delete(k) is True
    restantes = sorted(set(claves) - borradas)
    assert [k for k, _ in sf.scan()] == restantes
    for k in list(borradas)[:100]:
        assert sf.search(k) is None
    sf.reorganize()
    assert [k for k, _ in sf.scan()] == restantes
    sf.close()


def test_los_rangos_aleatorios_coinciden_con_la_referencia_en_memoria(tmp_path):
    rng = random.Random(99)
    claves = sorted(rng.sample(range(100_000), 3_000))
    sf = SequentialFile(str(tmp_path / "r"), S, page_size=1024, cache_size=0)
    sf.bulk_load(rec(k) for k in claves[:1500])
    for k in claves[1500:]:
        sf.insert(rec(k))

    for _ in range(50):
        lo, hi = sorted(rng.sample(range(-10, 100_010), 2))
        esperado = [rec(k) for k in claves if lo <= k <= hi]
        assert list(sf.range_search(lo, hi)) == esperado, f"rango [{lo}, {hi}]"
    sf.close()


def test_intercalar_inserciones_borrados_y_busquedas(tmp_path):
    """Modelo de referencia: un set en memoria contra el archivo en disco."""
    rng = random.Random(7)
    sf = SequentialFile(str(tmp_path / "i"), S, page_size=1024, cache_size=0)
    presentes: set[int] = set()

    for paso in range(3_000):
        accion = rng.random()
        if accion < 0.55 or not presentes:
            k = rng.randrange(50_000)
            if k in presentes:
                continue
            sf.insert(rec(k))
            presentes.add(k)
        elif accion < 0.85:
            k = rng.choice(sorted(presentes))
            assert sf.delete(k) is True, f"paso {paso}: no pudo borrar {k}"
            presentes.discard(k)
        else:
            k = rng.randrange(50_000)
            esperado = rec(k) if k in presentes else None
            assert sf.search(k) == esperado, f"paso {paso}: clave {k}"

        if paso % 750 == 0:
            sf.reorganize()

    assert [k for k, _ in sf.scan()] == sorted(presentes)
    assert sf.n_records == len(presentes)
    sf.close()


def test_claves_negativas_y_extremas(tmp_path):
    claves = [-(2**31), -1, 0, 1, 2**31 - 1]
    SE = Schema([Column("id", ColumnType.INT), Column("v", ColumnType.CHAR, 16)])

    def rec_e(k):
        return (k, f"v{k}")

    sf = SequentialFile(str(tmp_path / "e"), SE, page_size=1024, cache_size=0)
    for k in claves:
        sf.insert(rec_e(k))
    for k in claves:
        assert sf.search(k) == rec_e(k)
    assert [k for k, _ in sf.scan()] == sorted(claves)
    sf.reorganize()
    assert [k for k, _ in sf.scan()] == sorted(claves)
    sf.close()


def test_pk_float(tmp_path):
    SF = Schema([Column("k", ColumnType.FLOAT), Column("v", ColumnType.CHAR, 8)])
    rng = random.Random(3)
    claves = sorted({round(rng.uniform(-1000, 1000), 4) for _ in range(2000)})
    sf = SequentialFile(str(tmp_path / "f"), SF, page_size=1024, cache_size=0)
    sf.bulk_load((k, "x") for k in claves)
    for k in rng.sample(claves, 200):
        assert sf.search(k) == (k, "x")
    assert [k for k, _ in sf.scan()] == claves
    sf.close()
