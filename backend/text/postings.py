"""Postings en disco: un solo arreglo continuo de registros de ancho fijo.

Un posting es (page, slot, tf, dl): el RID de la fila, cuantas veces aparece el
termino en ella y el largo del documento. Guardar dl en el posting deja a BM25
calcular el score sin otra lectura. 16 B -> 252 postings por pagina de 4 KB.

Las listas van una detras de otra: la posicion global `pos` vive en la pagina
1 + pos // cap, slot pos % cap. Una lista se ubica con (inicio, largo) y leerla
cuesta las paginas que ocupa. Darle a cada lista su propia pagina desperdiciaba
casi todo el archivo: la mayoria de los terminos aparece en un solo documento.
"""

from collections.abc import Iterable, Iterator

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema

POSTING = Schema([Column(n, ColumnType.INT) for n in ("page", "slot", "tf", "dl")])


class PostingsFile:
    """`schema` cambia el registro: los bloques de SPIMI guardan ademas el termino."""

    def __init__(self, path: str, page_size: int = 4096, cache_size: int = 0,
                 counter: DiskCounter | None = None, schema: Schema = POSTING):
        self.schema = schema
        self.cap = page_capacity(page_size, schema.record_size)
        self.pager = Pager(path, record_size=schema.record_size, schema=schema,
                           page_size=page_size, cache_size=cache_size, counter=counter)
        self._tail = None  # ultima pagina, a medio llenar; se escribe al llenarse o al cerrar
        self._dirty = False

    def __len__(self) -> int:
        if self._tail is None and self.pager.page_count > 1:
            self._tail = self.pager.read_page(self.pager.page_count - 1)
        return 0 if self._tail is None else (self._tail.page_id - 1) * self.cap + self._tail.record_count

    def write(self, postings: Iterable[tuple]) -> int:
        """Agrega la lista al final del arreglo y devuelve su posicion inicial."""
        start = len(self)
        for p in postings:
            if self._tail is None or self._tail.is_full():
                if self._dirty:
                    self.pager.write_page(self._tail)
                self._tail = self.pager.new_page()
            n = self._tail.record_count
            self._tail.write_record(n, self.schema.pack(p))
            self._tail.set_occupied(n, True)
            self._tail.record_count = n + 1
            self._dirty = True
        return start

    def scan(self, start: int, count: int) -> Iterator[tuple]:
        """Recorre la lista pagina por pagina: en RAM solo vive la pagina actual."""
        pos, end = start, start + count
        while pos < end:
            pid = 1 + pos // self.cap
            page = self._tail if self._tail is not None and self._tail.page_id == pid else self.pager.read_page(pid)
            stop = min(end, pid * self.cap)
            for slot in range(pos % self.cap, pos % self.cap + stop - pos):
                yield self.schema.unpack(page.read_record(slot))
            pos = stop

    def read(self, start: int, count: int) -> list[tuple]:
        return list(self.scan(start, count))

    def close(self) -> None:
        if self._dirty:
            self.pager.write_page(self._tail)
        self.pager.close()

    def __enter__(self) -> "PostingsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
