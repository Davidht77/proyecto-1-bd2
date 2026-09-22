"""Heap File: almacenamiento sin ordenamiento con free-list.

Estructura simple sin invariantes de ordenamiento:
    <base>.heap.dat  paginas sin orden particular

Operaciones:
    Insercion: O(1) de I/O ubicando la primera página con espacio mediante free-list.
    Búsqueda: Escaneo secuencial completo (Full Table Scan) con costo de P lecturas.
    Borrado: Eliminación lógica con move-the-last para compactación inmediata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, NamedTuple

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE, Page, PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Schema


class DuplicateKeyError(Exception):
    """La clave primaria ya existe."""


class RID(NamedTuple):
    page_id: int
    slot: int


@dataclass
class InsertStats:
    rid: RID
    disk_reads: int
    disk_writes: int


class HeapFile:
    def __init__(
        self,
        base_path: str,
        schema: Schema,
        page_size: int = 4096,
        cache_size: int = 0,
    ):
        self.base_path = base_path
        self.schema = schema
        self.page_size = page_size
        self.cache_size = cache_size

        self.counter = DiskCounter()
        self._capacity = page_capacity(page_size, schema.record_size)
        self._open_pager()

    # ------------------------------------------------------------------ apertura

    @property
    def _data_path(self) -> str:
        return f"{self.base_path}.heap.dat"

    def _open_pager(self) -> None:
        self.pager = Pager(
            self._data_path,
            record_size=self.schema.record_size,
            # El esquema se escribe en la pagina 0 para que el archivo sea
            # autodescriptivo, igual que el Sequential File: el catalogo se
            # reconstruye leyendo esa cabecera, sin metadatos externos.
            schema=self.schema,
            page_size=self.page_size,
            cache_size=self.cache_size,
            counter=self.counter,
        )
        # Si es un archivo nuevo, inicializa la free-list a NULL_PAGE
        # (user_b se inicializa a 0 por defecto, pero necesitamos NULL_PAGE)
        if self.pager.page_count == 1 and self.pager._user_b == 0:
            self.free_list_head = NULL_PAGE

    # ------------------------------------------------------------------ metadatos

    @property
    def n_records(self) -> int:
        return self.pager.user_a

    @n_records.setter
    def n_records(self, value: int) -> None:
        self.pager.user_a = value

    @property
    def free_list_head(self) -> int:
        """Identificador de la pagina con espacio disponible, NULL_PAGE si no hay."""
        return self.pager.user_b

    @free_list_head.setter
    def free_list_head(self, value: int) -> None:
        self.pager.user_b = value

    @property
    def page_count(self) -> int:
        """Paginas logicas del archivo (la fisica 0 es la cabecera)."""
        return self.pager.page_count - 1

    # ------------------------------------------------- traduccion logico <-> fisico

    def _read_page(self, p: int) -> Page:
        return self.pager.read_page(p + 1)

    def _write_page(self, page: Page) -> None:
        self.pager.write_page(page)

    def _new_page(self) -> Page:
        return self.pager.new_page(PageType.DATA)

    # ----------------------------------------------------------------- recorridos

    def scan(self) -> Iterator[tuple]:
        """Escaneo secuencial completo de todos los registros."""
        for p in range(self.page_count):
            page = self._read_page(p)
            for slot in range(self.capacity):
                if page.is_occupied(slot):
                    raw = page.read_record(slot)
                    yield self.schema.unpack(raw)

    def scan_with_rid(self) -> Iterator[tuple[RID, tuple]]:
        for p in range(self.page_count):
            page = self._read_page(p)
            for slot in range(self.capacity):
                if page.is_occupied(slot):
                    yield RID(page.page_id, slot), self.schema.unpack(page.read_record(slot))

    # ------------------------------------------------------------------ insercion

    def insert(self, record: tuple) -> RID:
        """Inserta un registro y retorna su RID.

        Costo O(1) de I/O promedio: busca la primera página con espacio mediante
        la free-list. En el peor caso (todas las páginas llenas), crea una nueva.
        """
        payload = self.schema.pack(record)

        # Intenta insertar en la página con espacio de la free-list
        if self.free_list_head != NULL_PAGE:
            page = self.pager.read_page(self.free_list_head)
            if not page.is_full():
                slot = page.first_free_slot()
                page.write_record(slot, payload)
                page.set_occupied(slot, True)
                page.record_count = page.record_count + 1
                self._write_page(page)

                # Si la página se llena, actualiza la free-list
                if page.is_full():
                    self.free_list_head = NULL_PAGE

                self.n_records += 1
                return RID(page.page_id, slot)

        # No hay página con espacio: crea una nueva
        page = self._new_page()
        slot = page.first_free_slot()
        page.write_record(slot, payload)
        page.set_occupied(slot, True)
        page.record_count = 1
        self._write_page(page)

        # La nueva página es la cabeza de la free-list
        if not page.is_full():
            self.free_list_head = page.page_id

        self.n_records += 1
        return RID(page.page_id, slot)

    # ------------------------------------------------------------------ busqueda

    def search(self, key) -> tuple | None:
        """Búsqueda lineal secuencial por clave primaria.

        Costo: O(P) lecturas en el peor caso (P es el número total de páginas).
        """
        for registro in self.scan():
            if self.schema.key_of(registro) == key:
                return registro
        return None

    def _find_rid(self, key) -> RID | None:
        """Localiza la clave y devuelve su RID."""
        for p in range(self.page_count):
            page = self._read_page(p)
            for slot in range(self.capacity):
                if page.is_occupied(slot):
                    raw = page.read_record(slot)
                    if self.schema.key_from_bytes(raw) == key:
                        return RID(page.page_id, slot)
        return None

    # -------------------------------------------------------------------- borrado

    def delete(self, key) -> bool:
        """Elimina un registro por clave primaria usando move-the-last.

        Busca la clave, la elimina moviendo el último registro del archivo
        a su lugar (compactación inmediata).
        """
        rid = self._find_rid(key)
        if rid is None:
            return False

        # Encuentra la última página y slot ocupado del archivo
        last_page_id = None
        last_slot = None
        for p in range(self.page_count - 1, -1, -1):
            page = self._read_page(p)
            for s in range(self.capacity - 1, -1, -1):
                if page.is_occupied(s):
                    last_page_id = page.page_id
                    last_slot = s
                    break
            if last_page_id is not None:
                break

        if last_page_id is None:
            # Archivo vacío (no debería pasar si rid existe)
            return False

        if rid.page_id == last_page_id and rid.slot == last_slot:
            # Es el último registro: simplemente bórralo
            del_page = self.pager.read_page(rid.page_id)
            del_page.set_occupied(rid.slot, False)
            del_page.record_count = del_page.record_count - 1
            self._write_page(del_page)
        else:
            # Mueve el último registro al lugar del borrado
            # Lee ambas páginas (pueden ser iguales)
            del_page = self.pager.read_page(rid.page_id)
            last_page = self.pager.read_page(last_page_id)

            last_record = last_page.read_record(last_slot)
            del_page.write_record(rid.slot, last_record)
            self._write_page(del_page)

            # Solo escribe last_page si es diferente de del_page
            if rid.page_id != last_page_id:
                last_page.set_occupied(last_slot, False)
                last_page.record_count = last_page.record_count - 1
                self._write_page(last_page)
            else:
                # Es la misma página, actualiza directamente
                del_page.set_occupied(last_slot, False)
                del_page.record_count = del_page.record_count - 1
                self._write_page(del_page)

        # Actualiza la free-list si es necesario
        if not self.pager.read_page(rid.page_id).is_full():
            self.free_list_head = rid.page_id

        self.n_records -= 1
        return True

    # -------------------------------------------------------------- carga masiva

    def bulk_load(self, records: Iterable[tuple]) -> None:
        """Carga masiva de registros secuencialmente. Cero lecturas.

        Útil para cargar datasets grandes sin O(N) lecturas de página.
        """
        records_list = list(records)
        if not records_list:
            return

        if self.page_count > 0:
            raise ValueError("bulk_load requiere un archivo vacío")

        last_page = None
        for registro in records_list:
            payload = self.schema.pack(registro)

            # Reutiliza la última página si no está llena
            if last_page is None or last_page.is_full():
                last_page = self._new_page()

            slot = last_page.first_free_slot()
            last_page.write_record(slot, payload)
            last_page.set_occupied(slot, True)
            last_page.record_count = last_page.record_count + 1
            self._write_page(last_page)

        self.n_records = len(records_list)
        # Actualiza free-list: la última página si no está llena
        if last_page is not None:
            self.free_list_head = last_page.page_id if not last_page.is_full() else NULL_PAGE
        else:
            self.free_list_head = NULL_PAGE

    # ------------------------------------------------------------------ range

    def range_search(self, lo, hi) -> Iterator[tuple]:
        """Búsqueda por rango: requiere O(P) lecturas en el peor caso.

        Para el Experimento 3, necesario para comparar Sequential File vs Heap File.
        """
        for registro in self.scan():
            key = self.schema.key_of(registro)
            if lo <= key <= hi:
                yield registro

    # ---------------------------------------------------------------------- misc

    def flush(self) -> None:
        self.pager.flush()

    def close(self) -> None:
        self.pager.close()

    def __enter__(self) -> "HeapFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"HeapFile({self.base_path}, P={self.page_count}, "
            f"n={self.n_records})"
        )

    @property
    def capacity(self) -> int:
        """Capacidad de registros por página."""
        return self._capacity
