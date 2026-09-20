"""El unico componente del motor que toca el disco.

Contrato con el resto del equipo: ningun modulo fuera de backend/storage/ puede
llamar a open(), read(), write() o seek() sobre archivos de datos o indices.
Si esa regla se rompe, el DiskCounter deja de ser confiable y los cuatro
experimentos del informe quedan invalidados.

Todo acceso es por desplazamiento de puntero binario:
    os.pread(fd, page_size, page_id * page_size)
Nunca .read() global ni serializadores de alto nivel, como exige el enunciado.
"""

from __future__ import annotations

import os
import struct

from backend.storage.buffer_pool import BufferPool
from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE, Page, PageType
from backend.storage.schema import Column, ColumnType, Schema

FILE_MAGIC = b"MINIDB01"

# magic(8) page_size(4) page_count(4) free_list_head(4) record_size(4)
# n_columns(2) pk_index(2) user_a(4) user_b(4) = 36 bytes
_FILE_HDR = struct.Struct("<8sIIIIHHII")
_COL_HDR = struct.Struct("<32sBH")  # nombre, type_code, length


class InvalidFileError(Exception):
    """El archivo no es un archivo de este motor, o sus parametros no coinciden."""


# os.pread/os.pwrite solo existen en Unix. En Windows se emula con
# lseek + read/write, que es exactamente el "seek(offset) + read/write" que
# describe el enunciado. En ambos casos la transferencia es de un bloque
# completo en una posicion explicita: nunca una lectura global del archivo.
if hasattr(os, "pread"):

    def _pread_at(fd: int, n: int, offset: int) -> bytes:
        return os.pread(fd, n, offset)

    def _pwrite_at(fd: int, data: bytes, offset: int) -> None:
        os.pwrite(fd, data, offset)

else:

    def _pread_at(fd: int, n: int, offset: int) -> bytes:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, n)

    def _pwrite_at(fd: int, data: bytes, offset: int) -> None:
        os.lseek(fd, offset, os.SEEK_SET)
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])


class Pager:
    """Gestiona un archivo paginado: cabecera, free-list y transferencias de bloque.

    Es agnostico del contenido: entrega objetos Page (bytearray + page_id) y la
    capa superior los interpreta. Por eso el mismo Pager sirve para datos, nodos
    B+ y buckets de hash.

    La pagina fisica 0 es la cabecera del archivo; page_count la incluye.
    """

    def __init__(
        self,
        path: str,
        record_size: int,
        schema: Schema | None = None,
        page_size: int = 4096,
        cache_size: int = 0,
        counter: DiskCounter | None = None,
    ):
        if page_size < 64 or page_size & (page_size - 1):
            raise ValueError("page_size debe ser una potencia de 2 >= 64")

        self.path = path
        self.page_size = page_size
        self.record_size = record_size
        self.schema = schema
        self.counter = counter if counter is not None else DiskCounter()
        self.pool = BufferPool(cache_size)

        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        self._closed = False
        self._header_dirty = False

        if os.fstat(self._fd).st_size == 0:
            self.page_count = 1  # solo la cabecera
            self.free_list_head = NULL_PAGE
            self._user_a = 0
            self._user_b = 0
            self._header_dirty = True
            self._write_file_header()
        else:
            self._read_file_header()

    # ------------------------------------------------------- cabecera del archivo

    def _write_file_header(self) -> None:
        buf = bytearray(self.page_size)
        cols = self.schema.columns if self.schema else ()
        _FILE_HDR.pack_into(
            buf, 0,
            FILE_MAGIC, self.page_size, self.page_count, self.free_list_head,
            self.record_size, len(cols),
            self.schema.pk_index if self.schema else 0,
            self._user_a, self._user_b,
        )
        off = _FILE_HDR.size
        for col in cols:
            _COL_HDR.pack_into(buf, off, col.name.encode("utf-8"), int(col.type), col.length)
            off += _COL_HDR.size
        self._pwrite(0, bytes(buf))
        self._header_dirty = False

    def _read_file_header(self) -> None:
        raw = self._pread(0)
        (magic, page_size, page_count, free_head, record_size,
         n_cols, pk_index, user_a, user_b) = _FILE_HDR.unpack_from(raw, 0)

        if magic != FILE_MAGIC:
            raise InvalidFileError(f"{self.path}: magic invalido ({magic!r})")
        if page_size != self.page_size:
            raise InvalidFileError(
                f"{self.path}: el archivo usa page_size={page_size}, se pidio {self.page_size}"
            )
        if record_size != self.record_size:
            raise InvalidFileError(
                f"{self.path}: el archivo usa record_size={record_size}, se pidio {self.record_size}"
            )

        self.page_count = page_count
        self.free_list_head = free_head
        self._user_a = user_a
        self._user_b = user_b

        if self.schema is None and n_cols:
            off = _FILE_HDR.size
            cols = []
            for _ in range(n_cols):
                name, type_code, length = _COL_HDR.unpack_from(raw, off)
                off += _COL_HDR.size
                cols.append(
                    Column(name.split(b"\x00", 1)[0].decode("utf-8"),
                           ColumnType(type_code), length)
                )
            self.schema = Schema(cols, pk_index=pk_index)

    # ------------------------------------------------------------------ I/O crudo

    def _pread(self, page_id: int) -> bytes:
        raw = _pread_at(self._fd, self.page_size, page_id * self.page_size)
        if len(raw) < self.page_size:
            raw = raw.ljust(self.page_size, b"\x00")
        self.counter.record_read(self.page_size)
        return raw

    def _pwrite(self, page_id: int, payload: bytes) -> None:
        _pwrite_at(self._fd, payload, page_id * self.page_size)
        self.counter.record_write(self.page_size)

    # --------------------------------------------------------------- API publica

    def read_page(self, page_id: int) -> Page:
        self._check_open()
        if not 0 < page_id < self.page_count:
            raise IndexError(
                f"page_id {page_id} fuera de rango [1, {self.page_count})"
            )
        cached = self.pool.get(page_id)
        if cached is not None:
            return cached

        page = Page(page_id, self.page_size, self.record_size, data=self._pread(page_id))
        if page.stored_page_id != page_id:
            raise InvalidFileError(
                f"{self.path}: la pagina {page_id} dice ser la {page.stored_page_id}"
            )
        evicted = self.pool.put(page, dirty=False)
        if evicted is not None:
            self._pwrite(evicted.page_id, evicted.to_bytes())
        return page

    def write_page(self, page: Page) -> None:
        self._check_open()
        if not 0 < page.page_id < self.page_count:
            raise IndexError(f"page_id {page.page_id} fuera de rango")
        if self.pool.enabled:
            evicted = self.pool.put(page, dirty=True)
            self.pool.mark_dirty(page.page_id)
            if evicted is not None:
                self._pwrite(evicted.page_id, evicted.to_bytes())
        else:
            self._pwrite(page.page_id, page.to_bytes())

    def allocate_page(self, page_type: int = PageType.DATA) -> int:
        return self.new_page(page_type).page_id

    def new_page(self, page_type: int = PageType.DATA) -> Page:
        """Asigna una pagina y devuelve el objeto recien construido.

        Evita el read_page de vuelta que haria allocate_page + read_page: una
        pagina nueva esta vacia por definicion, leerla del disco no aporta nada
        y falsearia el conteo de I/O de bulk_load.
        """
        self._check_open()
        if self.free_list_head != NULL_PAGE:
            page_id = self.free_list_head
            self.free_list_head = self.read_page(page_id).next_page_id
        else:
            page_id = self.page_count
            self.page_count += 1
        self._header_dirty = True

        page = Page(page_id, self.page_size, self.record_size)
        page.page_type = page_type
        # write_page valida contra page_count, que ya fue incrementado arriba.
        self.write_page(page)
        return page

    def free_page(self, page_id: int) -> None:
        self._check_open()
        if page_id == 0:
            raise ValueError("la pagina 0 es la cabecera del archivo")
        if not 0 < page_id < self.page_count:
            raise IndexError(f"page_id {page_id} fuera de rango")
        page = Page(page_id, self.page_size, self.record_size)
        page.page_type = PageType.FREE
        page.next_page_id = self.free_list_head
        self.write_page(page)
        self.free_list_head = page_id
        self._header_dirty = True

    def flush(self) -> None:
        if self._closed:
            return
        for page in self.pool.dirty_pages():
            self._pwrite(page.page_id, page.to_bytes())
        self.pool.mark_all_clean()
        if self._header_dirty:
            self._write_file_header()

    def invalidate_cache(self) -> None:
        """Escribe las paginas sucias y vacia la cache.

        Util en benchmarks para medir desde una cache fria sin cerrar el archivo.
        """
        self.flush()
        self.pool.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        os.close(self._fd)
        self._closed = True

    # --------------------------------------------------------------------- misc

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError(f"{self.path}: el pager esta cerrado")

    # user_a / user_b: dos uint32 libres en la cabecera para que la capa
    # superior guarde metadatos persistentes. SequentialFile los usa para
    # n_records y n_overflow; el B+ Tree puede usar user_a como root_page_id y
    # el Hash Extensible como profundidad global.

    @property
    def user_a(self) -> int:
        return self._user_a

    @user_a.setter
    def user_a(self, value: int) -> None:
        self._user_a = value
        self._header_dirty = True

    @property
    def user_b(self) -> int:
        return self._user_b

    @user_b.setter
    def user_b(self, value: int) -> None:
        self._user_b = value
        self._header_dirty = True

    def __enter__(self) -> "Pager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Pager({self.path}, B={self.page_size}, pages={self.page_count})"
