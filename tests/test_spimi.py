import os
from collections import Counter

from backend.text.postings import PostingsFile
from backend.text.preprocess import preprocess
from backend.text.spimi import BLOCK, invert_blocks

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
        return pf.read(1)  # pagina 0 = cabecera


def test_con_tope_chico_salen_varios_bloques(tmp_path):
    paths = invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)
    assert len(paths) >= 2 and all(os.path.exists(p) for p in paths)


def test_cada_bloque_esta_ordenado_y_no_pasa_el_tope(tmp_path):
    for path in invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5):
        recs = leer_bloque(path)
        assert 0 < len(recs) <= 5
        assert [r[0] for r in recs] == sorted(r[0] for r in recs)


def test_posting_guarda_rid_tf_y_largo(tmp_path):
    recs = [r for p in invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5) for r in leer_bloque(p)]
    assert ("data", 2, 0, 2, 3) in recs and ("model", 2, 0, 1, 3) in recs


def test_entre_todos_los_bloques_no_se_pierde_ningun_posting(tmp_path):
    esperado = sorted((t, pg, sl, tf, len(terms))
                      for pg, sl, txt in DOCS
                      for terms in [preprocess(txt)]
                      for t, tf in Counter(terms).items())
    paths = invert_blocks(DOCS, str(tmp_path / "t__idx"), budget=5)
    assert sorted(r for p in paths for r in leer_bloque(p)) == esperado
