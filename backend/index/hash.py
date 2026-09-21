"""Hashing Dinamico en disco -- PENDIENTE.

Responsable: (asignar en el equipo)

Requisitos del enunciado (3.3.2): Extendible Hashing o Linear Hashing.

Lo que la capa de almacenamiento ya te resuelve:

  - pager.user_a como profundidad global; page.aux_page_id como profundidad
    local de cada bucket.
  - PageType.HASH ya existe en el enum.
  - pager.free_page(id) devuelve buckets vacios a la free-list tras una fusion.
  - page.next_page_id sirve para encadenar buckets de overflow.

Para conectarlo al motor basta con:
  1. Registrar el indice en Catalog.indexes_of() con kind="HASH".
  2. Implementar la rama IndexScan de Executor._rows_for().
     El planificador ya prefiere HASH sobre BTREE para igualdad.
"""

from backend.engine.errors import NotImplementedFeature


class ExtendibleHash:
    def __init__(self, *args, **kwargs):
        raise NotImplementedFeature("El Hashing Dinamico todavia no esta implementado.")
