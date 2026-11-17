"""Tests del R-Tree: correctitud contra fuerza bruta y persistencia."""

import math
import random

import pytest

from backend.index.rtree import RTree, RTreeRID


def _random_points(n, seed=0):
    rnd = random.Random(seed)
    return [(rnd.random(), rnd.random()) for _ in range(n)]


def _rid(i):
    return RTreeRID(i // 100, i % 100, 0)


def test_window_contra_fuerza_bruta(tmp_path):
    pts = _random_points(1200)
    tree = RTree(str(tmp_path / "rt"), page_size=512)
    for i, p in enumerate(pts):
        tree.insert(p, _rid(i))

    window = (0.2, 0.3, 0.6, 0.8)
    got = set(tree.search_window(window))
    want = {
        _rid(i)
        for i, (x, y) in enumerate(pts)
        if window[0] <= x <= window[2] and window[1] <= y <= window[3]
    }
    assert got == want
    assert tree.height >= 2


def test_knn_contra_ordenar_distancias(tmp_path):
    pts = _random_points(800, seed=3)
    tree = RTree(str(tmp_path / "rt"), page_size=512)
    for i, p in enumerate(pts):
        tree.insert(p, _rid(i))

    center = (0.5, 0.5)
    k = 21
    got = tree.knn(center, k)
    want = [
        _rid(i)
        for _d, i in sorted((math.dist(center, p), i) for i, p in enumerate(pts))[:k]
    ]
    assert [tuple(r) for r in got] == [tuple(r) for r in want]


def test_bulk_load_equivale_a_insercion(tmp_path):
    pts = _random_points(1500, seed=5)
    normal = RTree(str(tmp_path / "normal"), page_size=512)
    bulk = RTree(str(tmp_path / "bulk"), page_size=512)
    for i, p in enumerate(pts):
        normal.insert(p, _rid(i))
    bulk.bulk_load((p, _rid(i)) for i, p in enumerate(pts))

    window = (0.1, 0.1, 0.9, 0.9)
    assert set(normal.search_window(window)) == set(bulk.search_window(window))
    center = (0.42, 0.57)
    assert normal.knn(center, 15) == bulk.knn(center, 15)

    normal.close()
    bulk.close()


def test_persistencia_al_reabrir(tmp_path):
    pts = _random_points(600, seed=7)
    tree = RTree(str(tmp_path / "rt"), page_size=512)
    for i, p in enumerate(pts):
        tree.insert(p, _rid(i))
    window = (0.3, 0.3, 0.7, 0.7)
    before = set(tree.search_window(window))
    before_knn = tree.knn((0.5, 0.5), 10)
    tree.flush()
    tree.close()

    reopened = RTree(str(tmp_path / "rt"), page_size=512)
    assert set(reopened.search_window(window)) == before
    assert reopened.knn((0.5, 0.5), 10) == before_knn
    reopened.close()


def test_delete_mantiene_consistencia(tmp_path):
    pts = _random_points(400, seed=9)
    tree = RTree(str(tmp_path / "rt"), page_size=512)
    for i, p in enumerate(pts):
        tree.insert(p, _rid(i))

    for i in range(0, 400, 3):
        assert tree.delete(pts[i], _rid(i))

    window = (0.0, 0.0, 1.0, 1.0)
    got = set(tree.search_window(window))
    want = {_rid(i) for i in range(400) if i % 3 != 0}
    assert got == want


def test_dimension_generica(tmp_path):
    rnd = random.Random(4)
    dim = 8
    pts = [tuple(rnd.random() for _ in range(dim)) for _ in range(300)]
    tree = RTree(str(tmp_path / "rt"), page_size=1024, dim=dim)
    tree.bulk_load((p, _rid(i)) for i, p in enumerate(pts))

    window = tuple(0.2 for _ in range(dim)) + tuple(0.8 for _ in range(dim))
    got = set(tree.search_window(window))
    want = {
        _rid(i)
        for i, p in enumerate(pts)
        if all(0.2 <= v <= 0.8 for v in p)
    }
    assert got == want
    tree.close()


def test_bulk_load_requiere_arbol_vacio(tmp_path):
    tree = RTree(str(tmp_path / "rt"), page_size=512)
    tree.insert((0.1, 0.1), _rid(0))
    with pytest.raises(ValueError):
        tree.bulk_load([((0.2, 0.2), _rid(1))])
    tree.close()
