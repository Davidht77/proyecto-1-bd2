"""Arbol B+ multinivel en disco -- PENDIENTE.

Responsable: (asignar en el equipo)

Requisitos del enunciado (3.3.1):
  - Nodos internos: <P0, K1, P1, ..., Km, Pm> ordenados.
  - Nodos hoja: pares <Key, RID> + enlaces next_leaf_id / prev_leaf_id.
  - Insercion con split recursivo al 50% y creacion de nueva raiz.
  - Busqueda puntual en h = O(log_M N) transferencias.
  - Busqueda por rango descendiendo a la primera hoja y siguiendo next_leaf.

Lo que la capa de almacenamiento ya te resuelve:

  - page.next_page_id / page.prev_page_id de la cabecera SON los
    next_leaf_id / prev_leaf_id que pide el enunciado. No inventes otro
    mecanismo.
  - pager.user_a es un uint32 persistente en la cabecera del archivo: usalo
    como root_page_id. pager.user_b queda libre para la altura del arbol.
  - PageType.BT_INTERNAL y PageType.BT_LEAF ya existen en el enum.
  - page_capacity(page_size, entry_size) te da el fan-out M directamente, que
    es el dato que necesita la tabla del Experimento 4.

Para conectarlo al motor basta con:
  1. Registrar el indice en Catalog.indexes_of() con kind="BTREE".
  2. Implementar la rama IndexScan / IndexRangeScan de Executor._rows_for().
     El planificador (Executor._plan) ya las elige cuando encuentra el indice.
"""

from backend.engine.errors import NotImplementedFeature


class BPlusTree:
    def __init__(self, *args, **kwargs):
        raise NotImplementedFeature("El Arbol B+ todavia no esta implementado.")
