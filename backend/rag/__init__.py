"""Aplicacion de IA (enunciado 5.5, opcion A): RAG sobre el corpus de Wikipedia.

    corpus.py    descarga reanudable del corpus y carga en el motor (tabla +
                 IVF sobre embeddings SBERT + indice invertido SPIMI)
    llm.py       cliente de modelo de lenguaje independiente del proveedor
    pipeline.py  recuperacion lexica vs. semantica, prompt y respuesta citada
"""
