import pytest

from backend.text.wikipedia import MAX_TEXT_BYTES, clip


def test_clip_quita_saltos_de_linea():
    assert clip("Andorra es un país.\n\nLimita con España.") == "Andorra es un país. Limita con España."


def test_clip_no_pasa_de_1000_bytes_ni_corta_una_letra_a_la_mitad():
    out = clip("ñ" * 900)  # 1800 bytes en utf-8
    assert len(out.encode("utf-8")) <= MAX_TEXT_BYTES == 1000
    assert out == "ñ" * 500


def test_fetch_reintenta_si_la_red_falla(monkeypatch):
    import io
    import json
    import urllib.error

    import backend.text.wikipedia as wiki

    fallos = iter([urllib.error.URLError("caida"), urllib.error.URLError("caida")])
    cuerpo = {"rows": [{"row": {"id": "7", "title": "Andorra", "text": "Andorra es un país."}}]}

    def urlopen(url, timeout):
        err = next(fallos, None)
        if err:
            raise err
        return io.BytesIO(json.dumps(cuerpo).encode())

    monkeypatch.setattr(wiki.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(wiki.time, "sleep", lambda s: None)
    assert list(wiki.fetch("es", 1)) == [("es", "7", "Andorra", "Andorra es un país.")]


def test_from_parquet_saca_los_primeros_n_y_los_recorta(tmp_path):
    pa = pytest.importorskip("pyarrow")  # solo se instala para preparar el dataset (ver README)
    import pyarrow.parquet as pq

    from backend.text.wikipedia import from_parquet

    path = str(tmp_path / "x.parquet")
    pq.write_table(pa.table({"id": ["7", "10", "12"], "url": ["u"] * 3, "title": ["Andorra", "Argentina", "Arte"],
                             "text": ["Andorra\n\nes un país.", "ñ" * 900, "Arte."]}), path)
    out = list(from_parquet(path, "es", 2))
    assert out == [("es", "7", "Andorra", "Andorra es un país."), ("es", "10", "Argentina", "ñ" * 500)]
