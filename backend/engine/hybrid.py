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


def choose_strategy(est: HybridEstimate) -> tuple[str, str]:
    """Devuelve ("PRE" | "POST", razon legible para el panel del plan).

    TODO(contribucion): implementar el modelo de costos que elige la estrategia.
    """
    raise NotImplementedError("choose_strategy: falta el modelo de costos PRE vs POST")
