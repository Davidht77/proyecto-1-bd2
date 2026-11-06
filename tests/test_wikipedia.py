from backend.text.wikipedia import MAX_TEXT_BYTES, clip


def test_clip_quita_saltos_de_linea():
    assert clip("Andorra es un país.\n\nLimita con España.") == "Andorra es un país. Limita con España."


def test_clip_no_pasa_de_1000_bytes_ni_corta_una_letra_a_la_mitad():
    out = clip("ñ" * 900)  # 1800 bytes en utf-8
    assert len(out.encode("utf-8")) <= MAX_TEXT_BYTES == 1000
    assert out == "ñ" * 500
