from backend.text.preprocess import MAX_TERM_BYTES, STOPWORDS, preprocess, preprocess_query


def test_documento_en_espanol_usa_el_stemmer_espanol():
    assert preprocess("Las redes neuronales aprenden de los datos") == ["red", "neuronal", "aprend", "dat"]


def test_documento_en_ingles_usa_el_stemmer_ingles():
    assert preprocess("The networks are learning from the data") == ["network", "learn", "data"]


def test_consulta_corta_encuentra_documentos_en_los_dos_idiomas():
    doc_es = set(preprocess("Las redes de computadoras son rápidas"))
    doc_en = set(preprocess("The networks of computers are fast"))
    assert set(preprocess_query("redes")) & doc_es
    assert set(preprocess_query("networks")) & doc_en


def test_stopwords_son_las_de_nltk_en_ingles_y_espanol():
    assert len(STOPWORDS) == 504  # 198 en + 313 es - 7 compartidas


def test_quita_stopwords_en_espanol():
    terms = preprocess("Los datos de la red están en el servidor")
    assert len(terms) == 3 and not {"los", "de", "la", "están", "en", "el"} & set(terms)


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
