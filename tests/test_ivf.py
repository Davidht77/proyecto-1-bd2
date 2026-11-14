"""IVF en disco, K-Means y k-NN lineal (enunciado 5.3)."""

import numpy as np
import pytest

from backend.storage.schema import Column, ColumnType, Schema, parse_vector
from backend.vector.distance import distances
from backend.vector.ivf import IVFIndex, VRID
from backend.vector.kmeans import assign, kmeans
from backend.vector.knn import knn_linear, range_linear

D = 16


@pytest.fixture
def X():
    rng = np.random.default_rng(3)
    centros = rng.normal(size=(10, D)) * 5
    return (centros[rng.integers(10, size=1500)] + rng.normal(size=(1500, D))).astype(np.float32)


def _build(tmp_path, X, metric="L2", lists=None):
    idx = IVFIndex(str(tmp_path / "v"), Column("emb", ColumnType.VECTOR, D),
                   Column("id", ColumnType.INT), metric=metric)
    idx.build(((i, VRID(1 + i // 7, i % 7), X[i]) for i in range(len(X))), n_lists=lists)
    return idx


def test_schema_vector_ida_y_vuelta():
    s = Schema([Column("id", ColumnType.INT), Column("e", ColumnType.VECTOR, 3), Column("p", ColumnType.FLOAT)])
    assert s.record_size == 4 + 12 + 8
    assert s.unpack(s.pack((7, "[1, 2.5, -3]", 9.0))) == (7, (1.0, 2.5, -3.0), 9.0)
    with pytest.raises(ValueError, match="dimension"):
        s.pack((7, [1, 2], 9.0))


def test_parse_vector_formatos():
    assert parse_vector(" [1, -2e-1 ,3] ") == (1.0, -0.2, 3.0)
    assert parse_vector((1, 2)) == (1.0, 2.0)
    with pytest.raises(ValueError):
        parse_vector("1, 2")


def test_distancia_coseno_y_l2():
    q = np.array([1.0, 0.0])
    X = np.array([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    assert distances(q, X, "L2") == pytest.approx([1.0, np.sqrt(10), 1.0])
    assert distances(q, X, "COSINE") == pytest.approx([0.0, 1.0, 1.0])


def test_knn_linear_coincide_con_fuerza_bruta(X):
    q = X[0] + 0.3
    res = knn_linear(range(len(X)), lambda i: X[i], q, "L2", 7)
    assert [i for _, i in res] == list(np.argsort(np.linalg.norm(X - q, axis=1))[:7])
    r = res[-1][0]
    assert {i for _, i in range_linear(range(len(X)), lambda i: X[i], q, "L2", r)} == {i for _, i in res}


def test_kmeans_separa_clusters(X):
    C = kmeans(X, 10, seed=0)
    labels = assign(X, C)
    assert C.shape == (10, D)
    assert len(np.unique(labels)) == 10


def test_ivf_c_por_defecto_es_raiz_de_n(tmp_path, X):
    idx = _build(tmp_path, X)
    assert idx.n_lists == int(np.sqrt(len(X)))
    assert idx.n_vectors == len(X) == sum(idx.counts)
    idx.close()


@pytest.mark.parametrize("metric", ["L2", "COSINE"])
def test_recall_crece_con_probes_y_es_total_con_todas(tmp_path, X, metric):
    idx = _build(tmp_path, X, metric, lists=20)
    rng = np.random.default_rng(0)
    recalls, lecturas = {}, {}
    for p in (1, 20):
        rs, ls = [], []
        for qi in rng.integers(len(X), size=15):
            q = X[qi] + rng.normal(size=D).astype(np.float32) * 0.2
            gt = set(np.argsort(distances(q, X, metric), kind="stable")[:10].tolist())
            idx.counter.reset()
            got = {c.pk for c in idx.search(q, 10, p)}
            rs.append(len(gt & got) / 10)
            ls.append(idx.counter.disk_reads)
        recalls[p], lecturas[p] = np.mean(rs), np.mean(ls)
    assert recalls[20] == 1.0
    assert recalls[1] <= recalls[20]
    assert lecturas[1] < lecturas[20]
    idx.close()


def test_reabrir_desde_disco_sin_metadatos_externos(tmp_path, X):
    _build(tmp_path, X, "COSINE", lists=12).close()
    idx = IVFIndex(str(tmp_path / "v"))
    assert (idx.metric, idx.dim, idx.n_lists, idx.column.name) == ("COSINE", D, 12, "emb")
    assert idx.search(X[9], 1, 12)[0].pk == 9
    idx.close()


def test_delete_compacta_y_libera_paginas(tmp_path, X):
    idx = _build(tmp_path, X, lists=5)
    c = idx.nearest_cluster(X[0])
    miembros = [i for i in range(len(X)) if idx.nearest_cluster(X[i]) == c]
    for i in miembros:
        assert idx.delete(i, X[i])
    assert idx.counts[c] == 0 and idx.heads[c] == 0
    assert idx.n_vectors == len(X) - len(miembros)
    assert not idx.delete(miembros[0], X[miembros[0]])
    # Lo que queda sigue completo y encontrable.
    resto = next(i for i in range(len(X)) if i not in set(miembros))
    assert idx.search(X[resto], 1, 5)[0].pk == resto
    idx.close()


def test_insert_y_relocate(tmp_path, X):
    idx = _build(tmp_path, X, lists=8)
    v = np.full(D, 40.0, dtype=np.float32)
    idx.insert(5000, VRID(3, 1), v)
    assert idx.search(v, 1, 8)[0].pk == 5000
    assert idx.relocate(5000, v, VRID(9, 4))
    assert idx.search(v, 1, 8)[0].rid == VRID(9, 4, 0)
    idx.close()
