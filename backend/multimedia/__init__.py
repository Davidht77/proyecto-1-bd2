"""Modulo multimedia (enunciado 5.3.1): dos formas de representar un objeto.

    Paradigma clasico   bovw.py    SIFT (128-D) -> K-Means (codebook de K palabras
                                   visuales) -> histograma por imagen -> indice
                                   invertido de la Parte 3 con TF-IDF + coseno
    Paradigma moderno   dense.py   ResNet-18 (512-D) para imagenes y SBERT (384-D)
                                   para texto -> columna VECTOR(D) -> IVF en disco

    tfidf.py     TF-IDF + similitud de coseno sobre los postings SPIMI (lo usan
                 BoVW y la ruta lexica del RAG)
    datasets.py  Caltech-101 en disco
    service.py   construye las tablas e indices en el motor y resuelve consultas
                 por los dos caminos reportando el I/O de cada uno
"""
