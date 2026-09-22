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

from __future__ import annotations
 
from typing import Iterator, NamedTuple
 
from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE, Page, PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema
 
class RID(NamedTuple):
    page_id: int
    slot: int
 
class BPlusTree:
    def __init__(
        self,
        base_path: str,
        key_column: Column,
        page_size: int = 4096,
        cache_size: int = 0,
        counter: DiskCounter | None = None,
    ):
        self.base_path = base_path
        self.key_column = key_column
        self.page_size = page_size
        self.counter = counter if counter is not None else DiskCounter()
 
        # Mismo "shape" para hoja e interno: (key, f2:INT, f3:INT).
        self.node_schema = Schema(
            [key_column, Column("f2", ColumnType.INT), Column("f3", ColumnType.INT)],
            pk_index=0,
        )
        self._capacity = page_capacity(page_size, self.node_schema.record_size)
        if self._capacity < 3:
            raise ValueError(
                "la pagina es demasiado chica para el fan-out minimo de un "
                f"B+ Tree sobre {key_column.name} (capacidad calculada: {self._capacity})"
            )
 
        # Se pasa node_schema para que el archivo sea autodescriptivo (misma
        # filosofia que SequentialFile/HeapFile): Catalog puede reabrir un
        # indice leyendo su columna 0 con peek_header(), sin guardar el tipo
        # de la clave en ningun metadato externo.
        self.pager = Pager(
            f"{base_path}.btree.dat",
            record_size=self.node_schema.record_size,
            schema=self.node_schema,
            page_size=page_size,
            cache_size=cache_size,
            counter=self.counter,
        )
 
    # --------------------------------------------------- metadatos persistentes
 
    @property
    def root_page_id(self) -> int:
        return self.pager.user_a
 
    @root_page_id.setter
    def root_page_id(self, value: int) -> None:
        self.pager.user_a = value
 
    @property
    def height(self) -> int:
        return self.pager.user_b
 
    @height.setter
    def height(self, value: int) -> None:
        self.pager.user_b = value
 
    @property
    def is_empty(self) -> bool:
        return self.root_page_id == 0
 
    @property
    def fanout(self) -> int:
        return self._capacity + 1
 
    # ------------------------------------------------------------- empaquetado
 
    def _leaf_payload(self, key, rid: RID) -> bytes:
        return self.node_schema.pack((key, rid.page_id, rid.slot))
 
    def _internal_payload(self, key, child_page_id: int) -> bytes:
        return self.node_schema.pack((key, child_page_id, 0))
 
    def _leaf_entry(self, raw: bytes) -> tuple:
        key, page_id, slot = self.node_schema.unpack(raw)
        return key, RID(page_id, slot)
 
    def _internal_entry(self, raw: bytes) -> tuple:
        key, child_page_id, _ = self.node_schema.unpack(raw)
        return key, child_page_id
 
    # ------------------------------------------------------------------- API
 
    def search(self, key) -> RID | None:
        if self.is_empty:
            return None
        page = self._descend_to_leaf(key)
        found, pos = page.binary_search(key, self.node_schema)
        if not found:
            return None
        _, rid = self._leaf_entry(page.read_record(pos))
        return rid
 
    def range_search(self, lo, hi) -> Iterator[RID]:
        if self.is_empty:
            return
        page = self._descend_to_leaf(lo)
        while page is not None:
            for slot in range(page.record_count):
                k, rid = self._leaf_entry(page.read_record(slot))
                if k > hi:
                    return
                if k >= lo:
                    yield rid
            page = (
                self.pager.read_page(page.next_page_id)
                if page.next_page_id != NULL_PAGE
                else None
            )
 
    def insert(self, key, rid: RID) -> None:
        payload = self._leaf_payload(key, rid)
 
        if self.is_empty:
            leaf = self.pager.new_page(PageType.BT_LEAF)
            leaf.insert_sorted(payload, key, self.node_schema)
            self.pager.write_page(leaf)
            self.root_page_id = leaf.page_id
            self.height = 1
            return
 
        split = self._insert(self.root_page_id, key, payload)
        if split is not None:
            sep_key, right_child_id = split
            old_root_id = self.root_page_id
            new_root = self.pager.new_page(PageType.BT_INTERNAL)
            new_root.aux_page_id = old_root_id  # P0 = la raiz vieja
            new_root.insert_sorted(
                self._internal_payload(sep_key, right_child_id), sep_key, self.node_schema
            )
            self.pager.write_page(new_root)
            self.root_page_id = new_root.page_id
            self.height += 1
 
    def delete(self, key) -> bool:
        if self.is_empty:
            return False
        page = self._descend_to_leaf(key)
        found, pos = page.binary_search(key, self.node_schema)
        if not found:
            return False
        page.delete_dense(pos)
        self.pager.write_page(page)
        return True
 
    def flush(self) -> None:
        self.pager.flush()
 
    def close(self) -> None:
        self.pager.close()
 
    def __enter__(self) -> "BPlusTree":
        return self
 
    def __exit__(self, *exc) -> None:
        self.close()
 
    def __repr__(self) -> str:
        return (
            f"BPlusTree({self.base_path}, key={self.key_column.name}, "
            f"height={self.height}, fanout={self.fanout})"
        )
 
    # ------------------------------------------------------------- descenso
 
    def _descend_to_leaf(self, key) -> Page:
        page = self.pager.read_page(self.root_page_id)
        while page.page_type == PageType.BT_INTERNAL:
            page = self.pager.read_page(self._child_for(page, key))
        return page
 
    def _child_for(self, page: Page, key) -> int:

        prev_child = page.aux_page_id
        for slot in range(page.record_count):
            k, child = self._internal_entry(page.read_record(slot))
            if key < k:
                return prev_child
            prev_child = child
        return prev_child
 
    # --------------------------------------------------------- insercion
 
    def _insert(self, page_id: int, key, payload: bytes):
        """Inserta recursivamente. Devuelve (sep_key, new_page_id) si hubo
        split que hay que propagar al padre, o None si no hubo split."""
        page = self.pager.read_page(page_id)
 
        if page.page_type == PageType.BT_LEAF:
            if page.is_full():
                new_leaf_id, sep_key = self._split_leaf(page)
                target = self.pager.read_page(new_leaf_id) if key >= sep_key else page
                target.insert_sorted(payload, key, self.node_schema)
                self.pager.write_page(target)
                return sep_key, new_leaf_id
            page.insert_sorted(payload, key, self.node_schema)
            self.pager.write_page(page)
            return None
 
        # Nodo interno: desciende, y si el hijo se dividio, propaga la
        # nueva entrada (separador, nuevo hijo derecho) hacia este nivel.
        child_id = self._child_for(page, key)
        result = self._insert(child_id, key, payload)
        if result is None:
            return None
 
        sep_key, new_child_id = result
        if page.is_full():
            return self._split_internal(page, sep_key, new_child_id)
        page.insert_sorted(
            self._internal_payload(sep_key, new_child_id), sep_key, self.node_schema
        )
        self.pager.write_page(page)
        return None
 
    # -------------------------------------------------------------- splits
 
    def _split_leaf(self, page: Page) -> tuple[int, object]:
        """Divide una hoja llena al 50% y la enlaza en la lista de hojas."""
        new_leaf = self.pager.new_page(PageType.BT_LEAF)
        mid = page.record_count // 2
 
        moved = [page.read_record(slot) for slot in range(mid, page.record_count)]
        for slot in range(page.record_count - 1, mid - 1, -1):
            page.delete_dense(slot)
        for raw in moved:
            k = self.node_schema.key_from_bytes(raw)
            new_leaf.insert_sorted(raw, k, self.node_schema)
 
        # Relinkea next_leaf_id / prev_leaf_id (la busqueda por rango depende de esto).
        new_leaf.next_page_id = page.next_page_id
        new_leaf.prev_page_id = page.page_id
        if page.next_page_id != NULL_PAGE:
            old_next = self.pager.read_page(page.next_page_id)
            old_next.prev_page_id = new_leaf.page_id
            self.pager.write_page(old_next)
        page.next_page_id = new_leaf.page_id
 
        self.pager.write_page(page)
        self.pager.write_page(new_leaf)
 
        sep_key = new_leaf.first_key(self.node_schema)
        return new_leaf.page_id, sep_key
 
    def _split_internal(self, page: Page, incoming_key, incoming_child: int):

        entries = [self._internal_entry(page.read_record(s)) for s in range(page.record_count)]
        entries.append((incoming_key, incoming_child))
        entries.sort(key=lambda e: e[0])
 
        mid = len(entries) // 2
        promoted_key, promoted_child = entries[mid]
        left_entries = entries[:mid]
        right_entries = entries[mid + 1 :]
 
        old_p0 = page.aux_page_id
        for slot in range(page.record_count - 1, -1, -1):
            page.delete_dense(slot)
        for k, child in left_entries:
            page.insert_sorted(self._internal_payload(k, child), k, self.node_schema)
        page.aux_page_id = old_p0
 
        new_page = self.pager.new_page(PageType.BT_INTERNAL)
        new_page.aux_page_id = promoted_child
        for k, child in right_entries:
            new_page.insert_sorted(self._internal_payload(k, child), k, self.node_schema)
 
        self.pager.write_page(page)
        self.pager.write_page(new_page)
 
        return promoted_key, new_page.page_id
 