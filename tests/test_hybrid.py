"""Modelo de costos PRE vs POST del planificador hibrido (enunciado 5.4).

Los casos replican la configuracion del Experimento 8 (N = 20k, D = 128,
141 listas, ~20 paginas por lista, probes = 5, k = 10) y comparan la decision
con lo que se midio en benchmarks/results/exp8_hibrido.csv.
"""

import pytest

from backend.engine.hybrid import HybridEstimate, choose_strategy, estimate_costs

N = 20_000


def _exp8(matches: int, capped: bool = False, k: int = 10) -> HybridEstimate:
    return HybridEstimate(est_matches=matches, capped=capped, n_rows=N, k=k, probes=5,
                          n_lists=141, avg_pages_per_list=20.3, rows_per_page=7)


@pytest.mark.parametrize("matches,capped,esperada", [
    (20, False, "PRE"),      # 0.1 %: medido PRE 23 vs POST 14127 bloques
    (200, False, "PRE"),     # 1 %:   medido PRE 204 vs POST 1333
    (1000, False, "POST"),   # 5 %:   medido PRE 1009 vs POST 333
    (2000, True, "POST"),    # >=10 % (conteo cortado): medido POST 165 / 136
])
def test_decision_coincide_con_el_experimento_8(matches, capped, esperada):
    estrategia, razon = choose_strategy(_exp8(matches, capped))
    assert estrategia == esperada
    assert "bloques" in razon


def test_costo_post_se_acerca_a_lo_medido():
    _, post = estimate_costs(_exp8(1000))      # 5 %: medido 333
    assert 250 <= post <= 450
    _, post = estimate_costs(_exp8(2000, True))  # filtro amplio: medido ~136-165
    assert post < 250


def test_sin_filas_que_cumplan_gana_pre():
    assert choose_strategy(_exp8(0))[0] == "PRE"


def test_menos_filas_que_k_obliga_a_post_a_recorrer_todo():
    pre, post = estimate_costs(_exp8(3, k=10))
    assert post >= N
    assert pre == 3


def test_empate_favorece_pre():
    est = HybridEstimate(est_matches=7, capped=False, n_rows=7, k=7, probes=1,
                         n_lists=1, avg_pages_per_list=0.0, rows_per_page=7)
    assert estimate_costs(est) == (7.0, 7.0)
    assert choose_strategy(est)[0] == "PRE"


def test_razon_marca_la_cota_cuando_el_conteo_se_corto():
    assert ">=2000" in choose_strategy(_exp8(2000, True))[1]
