from backend.text.preprocess import MAX_TERM_BYTES, preprocess


def test_variantes_de_una_palabra_dan_la_misma_raiz():
    assert preprocess("the cat runs") == preprocess("Cats running!") == ["cat", "run"]


def test_quita_stopwords_y_signos():
    assert preprocess("The model, and the data.") == ["model", "data"]


def test_conserva_repetidos_para_contar_tf():
    assert preprocess("data data") == ["data", "data"]


def test_ningun_termino_pasa_de_24_bytes():
    terms = preprocess("pneumonoultramicroscopicsilicovolcanoconiosis Schrödingerschrödingerschrödinger")
    assert terms and all(len(t.encode("utf-8")) <= MAX_TERM_BYTES == 24 for t in terms)


def test_texto_sin_palabras_da_lista_vacia():
    assert preprocess("  ... 123 , ;") == []
