"""Listas de postings en disco: paginas encadenadas por next_page_id.

Un posting es (page, slot, tf, dl): el RID de la fila, cuantas veces aparece el
termino en ella y el largo del documento. Guardar dl en el posting deja a BM25
calcular el score sin otra lectura. 16 B -> 252 postings por pagina de 4 KB.
"""

from collections.abc import Iterable

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema

POSTING = Schema([Column(n, ColumnType.INT) for n in ("page", "slot", "tf", "dl")])


class PostingsFile:
    def __init__(self, path: str, page_size: int = 4096, cache_size: int = 0,
                 counter: DiskCounter | None = None):
        self.pager = Pager(path, record_size=POSTING.record_size, schema=POSTING,
                           page_size=page_size, cache_size=cache_size, counter=counter)

    def write(self, postings: Iterable[tuple]) -> int:
        """Escribe la lista en paginas nuevas y devuelve el page_id de la primera."""
        head, page = NULL_PAGE, None
        for p in postings:
            if page is None or page.is_full():
                new = self.pager.new_page()
                if page is None:
                    head = new.page_id
                else:
                    page.next_page_id = new.page_id
                    self.pager.write_page(page)
                page = new
            n = page.record_count
            page.write_record(n, POSTING.pack(p))
            page.set_occupied(n, True)
            page.record_count = n + 1
        if page is not None:
            self.pager.write_page(page)
        return head

    def read(self, head: int) -> list[tuple]:
        out, pid = [], head
        while pid != NULL_PAGE:
            page = self.pager.read_page(pid)
            out.extend(POSTING.unpack(r) for r in page.records())
            pid = page.next_page_id
        return out

    def close(self) -> None:
        self.pager.close()

    def __enter__(self) -> "PostingsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
