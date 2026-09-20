"""Cache LRU write-back opcional.

Esta desactivada por defecto (capacity=0) para que los benchmarks del informe
midan el I/O teorico puro: cada read_page es una transferencia fisica y una
busqueda en un arbol de altura h cuesta exactamente h lecturas. Con la cache
activa los niveles altos quedan residentes y el conteo deja de coincidir con
la formula, que es lo que ocurre en un motor real.
"""

from __future__ import annotations

from collections import OrderedDict

from backend.storage.page import Page


class BufferPool:
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("la capacidad no puede ser negativa")
        self.capacity = capacity
        self._entries: OrderedDict[int, list] = OrderedDict()  # page_id -> [Page, dirty]
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0

    def get(self, page_id: int) -> Page | None:
        entry = self._entries.get(page_id)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(page_id)
        self.hits += 1
        return entry[0]

    def put(self, page: Page, dirty: bool) -> Page | None:
        """Inserta o actualiza. Devuelve la pagina sucia desalojada, si la hay."""
        if not self.enabled:
            return None
        pid = page.page_id
        if pid in self._entries:
            entry = self._entries[pid]
            entry[0] = page
            entry[1] = entry[1] or dirty
            self._entries.move_to_end(pid)
            return None

        evicted = None
        if len(self._entries) >= self.capacity:
            _, victima = self._entries.popitem(last=False)
            if victima[1]:
                evicted = victima[0]
        self._entries[pid] = [page, dirty]
        return evicted

    def mark_dirty(self, page_id: int) -> None:
        entry = self._entries.get(page_id)
        if entry is not None:
            entry[1] = True

    def dirty_pages(self) -> list[Page]:
        return [page for page, dirty in self._entries.values() if dirty]

    def mark_all_clean(self) -> None:
        for entry in self._entries.values():
            entry[1] = False

    def clear(self) -> None:
        """Vacia la cache. Exige que no queden paginas sucias.

        Descartar una pagina sucia perderia datos ya confirmados por la capa
        superior. Convertir eso en un error ruidoso evita una corrupcion
        silenciosa; usa Pager.invalidate_cache(), que escribe antes de vaciar.
        """
        sucias = [p.page_id for p, dirty in self._entries.values() if dirty]
        if sucias:
            raise ValueError(f"hay paginas sucias sin escribir: {sucias}")
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"BufferPool({len(self._entries)}/{self.capacity}, hits={self.hits}, misses={self.misses})"
