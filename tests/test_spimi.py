import csv
import os
from collections import Counter
from pathlib import Path

from backend.text.postings import PostingsFile
from backend.text.preprocess import preprocess
from backend.text.spimi import BLOCK, build_index, invert_blocks, open_index

DOCS = [
    (1, 0, "Neural networks learn graph representations"),
    (1, 1, "Graph networks for data"),
    (1, 2, "Las redes neuronales aprenden de los datos"),
    (2, 0, "Data data model"),
    (2, 1, "Model of the graph"),
    (2, 2, "Redes de grafos y modelos"),
]


def leer_bloque(path):
    with PostingsFile(path, schema=BLOCK) as pf:
        return pf.read(0, len(pf))


def test_con_tope_chico_salen_varios_bloques(tmp_path):
    paths = invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)[0]
    assert len(paths) >= 2 and all(os.path.exists(p) for p in paths)


def test_cada_bloque_esta_ordenado_y_no_pasa_el_tope(tmp_path):
    for path in invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)[0]:
        recs = leer_bloque(path)
        assert 0 < len(recs) <= 5
        assert [r[0] for r in recs] == sorted(r[0] for r in recs)


def test_posting_guarda_rid_tf_y_largo(tmp_path):
    recs = [r for p in invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)[0] for r in leer_bloque(p)]
    assert ("data", 2, 0, 2, 3) in recs and ("model", 2, 0, 1, 3) in recs


def test_entre_todos_los_bloques_no_se_pierde_ningun_posting(tmp_path):
    esperado = sorted((t, pg, sl, tf, len(terms))
                      for pg, sl, txt in DOCS
                      for terms in [preprocess(txt)]
                      for t, tf in Counter(terms).items())
    paths = invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)[0]
    assert sorted(r for p in paths for r in leer_bloque(p)) == esperado


def indice_en_memoria(docs):
    idx = {}
    for pg, sl, txt in docs:
        terms = preprocess(txt)
        for t, tf in Counter(terms).items():
            idx.setdefault(t, []).append((pg, sl, tf, len(terms)))
    return idx


def comparar_con_memoria(docs, base, budget):
    build_index(docs, base, budget=budget)
    vocab, post = open_index(base)
    with vocab, post:
        for term, lista in indice_en_memoria(docs).items():
            start, df = vocab.search(term)
            assert df == len(lista) and post.read(start, df) == lista, term


def test_merge_da_el_mismo_indice_que_un_diccionario_en_memoria(tmp_path):
    comparar_con_memoria(DOCS, str(tmp_path / "t__idx"), budget=5)


def test_merge_con_un_posting_por_bloque(tmp_path):
    comparar_con_memoria(DOCS, str(tmp_path / "t__idx"), budget=1)


def docs_wiki():
    with open(Path(__file__).parent / "fixtures" / "wiki_muestra.csv", encoding="utf-8") as f:
        return [(i // 100, i % 100, r["text"]) for i, r in enumerate(csv.DictReader(f))]


def test_merge_sobre_la_muestra_de_wikipedia(tmp_path):
    comparar_con_memoria(docs_wiki(), str(tmp_path / "wiki__idx"), budget=2000)


def test_merge_empaqueta_las_listas_sin_hojas_vacias(tmp_path):
    docs = docs_wiki()
    base = str(tmp_path / "wiki__idx")
    build_index(docs, base, budget=2000)
    total = sum(len(lista) for lista in indice_en_memoria(docs).values())
    vocab, post = open_index(base)
    with vocab, post:
        assert post.pager.page_count - 1 == -(-total // 252)  # ceil: 261 paginas, no 16 366


def test_merge_guarda_n_y_largo_total_para_bm25(tmp_path):
    base = str(tmp_path / "t__idx")
    build_index(DOCS, base, budget=5)
    vocab, post = open_index(base)
    with vocab, post:
        assert post.pager.user_a == len(DOCS)
        assert post.pager.user_b == sum(len(preprocess(t)) for *_, t in DOCS)


def test_merge_borra_los_bloques_temporales(tmp_path):
    build_index(DOCS, str(tmp_path / "t__idx"), budget=5)
    assert not list(tmp_path.glob("*.spimi.blk*"))
