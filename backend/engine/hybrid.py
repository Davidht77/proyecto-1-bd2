"""Decision del planificador hibrido (enunciado 5.4).

Consulta tipica:

    SELECT ... FROM multimedia
    WHERE categoria = 'footwear' AND precio <= 250.0
    ORDER BY embedding <=> '[...]' LIMIT 5

Hay dos formas de resolverla cuando existen a la vez un indice relacional
(B+ / Hash / clave del Sequential) y un indice IVF sobre el vector:

  PRE-filtering   El indice relacional entrega solo las filas que cumplen el
                  WHERE; sobre ellas se hace k-NN exacto. Costo ~ proporcional
                  a la cantidad de filas que pasan el filtro (cada una cuesta
                  hasta una lectura de pagina de la tabla). Recall = 100 %.

  POST-filtering  El IVF entrega candidatos en orden de distancia escaneando
                  `probes` listas; a cada candidato se le lee la fila y se le
                  aplica el WHERE hasta juntar k. Costo ~ paginas de las listas
                  sondeadas + una lectura de tabla por candidato revisado. Si
                  el filtro es muy selectivo, casi todos los candidatos se
                  descartan y hay que ampliar el sondeo.

Este modulo no hace I/O: recibe estimaciones ya calculadas y decide.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HybridEstimate:
    """Lo que el planificador sabe al momento de decidir.

    est_matches       filas que cumplen el predicado indexado, contadas recorriendo
                      solo el indice relacional (sin tocar la tabla). Si `capped`
                      es True el conteo se corto en ese valor: hay AL MENOS esas.
    n_rows            filas de la tabla.
    k                 LIMIT de la consulta.
    probes            listas que sondearia el IVF (SET ivf.probes).
    n_lists           celdas del IVF (C).
    avg_pages_per_list paginas de disco promedio por lista invertida.
    rows_per_page     registros por pagina de la tabla de datos.
    """

    est_matches: int
    capped: bool
    n_rows: int
    k: int
    probes: int
    n_lists: int
    avg_pages_per_list: float
    rows_per_page: int

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_costs(est: HybridEstimate) -> tuple[float, float]:
    """Bloques de disco que leeria cada estrategia: (costo_pre, costo_post).

    PRE   una lectura de tabla por fila que cumple el filtro: el indice
          relacional no es clustered, asi que filas vecinas en el indice caen
          en paginas distintas de la tabla.

    POST  con selectividad s, para juntar k aciertos hay que revisar ~k/s
          candidatos (una lectura de tabla cada uno, para evaluar el WHERE).
          Esos candidatos salen de las listas mas cercanas: cada lista aporta
          ~n_rows/n_lists vectores y cuesta avg_pages_per_list lecturas. Nunca
          se recorren menos de `probes` listas ni mas de `n_lists`. Si el
          filtro deja pasar menos de k filas, POST termina recorriendo todo.
    """
    n_rows = max(1, est.n_rows)
    n_lists = max(1, est.n_lists)
    costo_pre = float(est.est_matches)

    if est.est_matches == 0:
        candidatos = n_rows
    else:
        candidatos = min(n_rows, math.ceil(est.k * n_rows / est.est_matches))
    por_lista = n_rows / n_lists
    listas = min(n_lists, max(est.probes, math.ceil(candidatos / por_lista)))
    costo_post = listas * est.avg_pages_per_list + candidatos
    return costo_pre, costo_post


def choose_strategy(est: HybridEstimate) -> tuple[str, str]:
    """Devuelve ("PRE" | "POST", razon legible para el panel del plan).

    Minimiza bloques leidos, la metrica que reporta el DiskCounter. Ante un
    empate gana PRE: cuesta lo mismo y su resultado es exacto, mientras que
    POST hereda el recall aproximado del IVF.
    """
    pre, post = estimate_costs(est)
    sel = est.est_matches / max(1, est.n_rows)
    cota = ">=" if est.capped else "~"
    datos = (f"{cota}{est.est_matches} filas cumplen el filtro ({sel:.1%}); "
             f"PRE {cota}{pre:,.0f} bloques vs POST ~{post:,.0f}")
    if pre <= post:
        return "PRE", f"{datos}: conviene filtrar primero (y es exacto)"
    return "POST", f"{datos}: conviene recorrer el IVF y filtrar despues"
