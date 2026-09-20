"""Sequential File: area principal ordenada + area de overflow encadenada.

Estructura dual (enunciado 3.2.2):
    <base>.seq.dat   paginas ordenadas fisicamente por la clave primaria
    <base>.seq.ovf   paginas de overflow, una cadena por pagina principal

Invariante central:
    Los registros con clave en [low_key(p), low_key(p+1)) viven en la pagina
    principal p o en la cadena de overflow de p.

low_key es el rango *declarado* de la pagina, no su contenido actual: solo lo
escriben bulk_load y reorganize. Si dependiera del primer registro, borrar ese
registro subiria el minimo de la pagina y las claves menores que quedaran en su
cadena de overflow dejarian de ser alcanzables por la busqueda binaria.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, NamedTuple

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE, Page, PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import ColumnType, Schema


class UnsupportedKeyType(Exception):
    """low_key ocupa 8 bytes: la PK debe ser INT o FLOAT."""


class DuplicateKeyError(Exception):
    """La clave primaria ya existe."""


class RID(NamedTuple):
    page_id: int
    slot: int
    area: str  # "main" | "overflow"


@dataclass
class ReorgStats:
    records: int
    pages_before: int
    pages_after: int
    overflow_pages_freed: int
    disk_reads: int
    disk_writes: int
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "records": self.records,
            "pages_before": self.pages_before,
            "pages_after": self.pages_after,
            "overflow_pages_freed": self.overflow_pages_freed,
            "disk_reads": self.disk_reads,
            "disk_writes": self.disk_writes,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class SequentialFile:
    def __init__(
        self,
        base_path: str,
        schema: Schema,
        page_size: int = 4096,
        fill_factor: float = 0.75,
        cache_size: int = 0,
    ):
        if schema.pk_column.type not in (ColumnType.INT, ColumnType.FLOAT):
            raise UnsupportedKeyType(
                f"SequentialFile requiere PK INT o FLOAT, recibio "
                f"{schema.pk_column.type.name}"
            )
        if not 0.1 <= fill_factor <= 1.0:
            raise ValueError("fill_factor debe estar en (0.1, 1.0]")

        self.base_path = base_path
        self.schema = schema
        self.page_size = page_size
        self.fill_factor = fill_factor
        self.cache_size = cache_size

        # Un unico contador compartido: si cada area llevara el suyo, el
        # desglose de I/O por consulta saldria partido en dos.
        self.counter = DiskCounter()
        self._capacity = page_capacity(page_size, schema.record_size)
        self._open_pagers()

    # ------------------------------------------------------------------ apertura

    @property
    def _main_path(self) -> str:
        return f"{self.base_path}.seq.dat"

    @property
    def _ovf_path(self) -> str:
        return f"{self.base_path}.seq.ovf"

    def _open_pagers(self) -> None:
        kw = dict(
            record_size=self.schema.record_size,
            page_size=self.page_size,
            cache_size=self.cache_size,
            counter=self.counter,
        )
        self.main = Pager(self._main_path, schema=self.schema, **kw)
        self.ovf = Pager(self._ovf_path, **kw)

    # ------------------------------------------------------------------ metadatos

    @property
    def n_records(self) -> int:
        return self.main.user_a

    @n_records.setter
    def n_records(self, value: int) -> None:
        self.main.user_a = value

    @property
    def n_overflow(self) -> int:
        return self.main.user_b

    @n_overflow.setter
    def n_overflow(self, value: int) -> None:
        self.main.user_b = value

    @property
    def page_count(self) -> int:
        """Paginas logicas del area principal (la fisica 0 es la cabecera)."""
        return self.main.page_count - 1

    # ------------------------------------------------- traduccion logico <-> fisico
    # El indice logico p va de 0 a P-1; el identificador fisico es p + 1.
    # La conversion vive solo aqui: mezclar los dos espacios de numeracion es
    # el off-by-one clasico de los archivos paginados.

    def _read_main(self, p: int) -> Page:
        return self.main.read_page(p + 1)

    def _write_main(self, page: Page) -> None:
        self.main.write_page(page)

    def _new_main_page(self) -> Page:
        return self.main.new_page(PageType.DATA)

    # ----------------------------------------------------------------- recorridos

    def _chain_pages(self, head: int) -> Iterator[Page]:
        pid = head
        while pid != NULL_PAGE:
            page = self.ovf.read_page(pid)
            yield page
            pid = page.next_page_id

    def _chain_records(self, head: int) -> Iterator[bytes]:
        for page in self._chain_pages(head):
            yield from page.records()

    def _merged(self, page: Page) -> Iterator[bytes]:
        """Orden logico de una pagina principal: sus registros mezclados con su cadena."""
        if page.aux_page_id == NULL_PAGE:
            yield from page.records()
        else:
            yield from heapq.merge(
                page.records(),
                self._chain_records(page.aux_page_id),
                key=self.schema.key_from_bytes,
            )

    def scan(self) -> Iterator[tuple]:
        for p in range(self.page_count):
            for raw in self._merged(self._read_main(p)):
                yield self.schema.unpack(raw)

    # ------------------------------------------------------------ busqueda binaria

    def _locate(self, key) -> tuple[int, Page]:
        """Ultima pagina logica cuyo low_key <= key, junto con la pagina misma.

        Como low_key(0) es -inf, una clave por debajo de todo cae en la pagina 0,
        que es exactamente su rango.

        Devolver tambien la pagina evita releerla despues: la binaria ya la tuvo
        en memoria en su ultimo paso util. Es una lectura menos por busqueda, que
        sobre las 1000 consultas del Experimento 2 son 1000 I/O menos.
        """
        lo, hi = 0, self.page_count - 1
        resultado, pagina = 0, None
        while lo <= hi:
            mid = (lo + hi) // 2
            page = self._read_main(mid)
            if page.get_low_key(self.schema) <= key:
                resultado, pagina, lo = mid, page, mid + 1
            else:
                hi = mid - 1
        if pagina is None:
            pagina = self._read_main(resultado)
        return resultado, pagina

    def _locate_page(self, key) -> int:
        return self._locate(key)[0]

    def search(self, key) -> tuple | None:
        if self.page_count == 0:
            return None
        _, page = self._locate(key)
        found, slot = page.binary_search(key, self.schema)
        if found:
            return self.schema.unpack(page.read_record(slot))
        for ov in self._chain_pages(page.aux_page_id):
            if key < ov.first_key(self.schema):
                return None  # salida temprana: la cadena esta ordenada
            found, slot = ov.binary_search(key, self.schema)
            if found:
                return self.schema.unpack(ov.read_record(slot))
        return None

    def _find_rid(self, key) -> tuple[RID, Page] | None:
        """Localiza la clave y devuelve su RID junto con la pagina que la contiene."""
        if self.page_count == 0:
            return None
        _, page = self._locate(key)
        found, slot = page.binary_search(key, self.schema)
        if found:
            return RID(page.page_id, slot, "main"), page
        for ov in self._chain_pages(page.aux_page_id):
            if key < ov.first_key(self.schema):
                return None
            found, slot = ov.binary_search(key, self.schema)
            if found:
                return RID(ov.page_id, slot, "overflow"), ov
        return None

    # ------------------------------------------------------------------ insercion

    def insert(self, record: tuple) -> RID:
        key = self.schema.key_of(record)
        payload = self.schema.pack(record)

        if self.page_count == 0:
            page = self._new_main_page()
            page.set_low_key(None, self.schema)  # centinela -inf
            slot = page.insert_sorted(payload, key, self.schema)
            self._write_main(page)
            self.n_records += 1
            return RID(page.page_id, slot, "main")

        # Una sola pasada: la binaria localiza la pagina, y el recorrido de la
        # cadena sirve a la vez para detectar duplicados y para elegir donde
        # insertar. Hacerlo en pasadas separadas triplicaba el I/O, y sobre una
        # cadena larga ese factor 3 se nota.
        _, page = self._locate(key)
        found, _ = page.binary_search(key, self.schema)
        if found:
            raise DuplicateKeyError(f"la clave {key!r} ya existe")

        if not page.is_full():
            slot = page.insert_sorted(payload, key, self.schema)
            self._write_main(page)
            rid = RID(page.page_id, slot, "main")
        else:
            rid = self._insert_overflow(page, payload, key)
            self.n_overflow += 1

        self.n_records += 1
        if self._should_reorganize():
            self.reorganize()
        return rid

    def _insert_overflow(self, main_page: Page, payload: bytes, key) -> RID:
        """Inserta en la cadena de la pagina principal manteniendola ordenada."""
        head = main_page.aux_page_id

        if head == NULL_PAGE:
            page = self._new_ovf_page()
            slot = page.insert_sorted(payload, key, self.schema)
            self.ovf.write_page(page)
            main_page.aux_page_id = page.page_id
            self._write_main(main_page)
            return RID(page.page_id, slot, "overflow")

        # Ultima pagina de la cadena cuya primera clave sea <= key; si la clave
        # es menor que todas, la cabeza. El mismo recorrido detecta duplicados,
        # asi la cadena se lee una vez y no dos.
        objetivo = None
        for page in self._chain_pages(head):
            if page.first_key(self.schema) <= key:
                objetivo = page
                if page.binary_search(key, self.schema)[0]:
                    raise DuplicateKeyError(f"la clave {key!r} ya existe")
            else:
                break
        if objetivo is None:
            objetivo = self.ovf.read_page(head)

        if not objetivo.is_full():
            slot = objetivo.insert_sorted(payload, key, self.schema)
            self.ovf.write_page(objetivo)
            return RID(objetivo.page_id, slot, "overflow")

        return self._split_overflow(objetivo, payload, key)

    def _split_overflow(self, page: Page, payload: bytes, key) -> RID:
        """Parte la pagina llena en dos y enlaza la nueva justo despues."""
        nueva = self._new_ovf_page()
        mitad = page.record_count // 2
        movidos = [page.read_record(s) for s in range(mitad, page.record_count)]
        for _ in range(len(movidos)):
            page.delete_dense(page.record_count - 1)
        for raw in movidos:
            nueva.insert_sorted(raw, self.schema.key_from_bytes(raw), self.schema)

        nueva.next_page_id = page.next_page_id
        nueva.prev_page_id = page.page_id
        page.next_page_id = nueva.page_id
        if nueva.next_page_id != NULL_PAGE:
            siguiente = self.ovf.read_page(nueva.next_page_id)
            siguiente.prev_page_id = nueva.page_id
            self.ovf.write_page(siguiente)

        destino = nueva if key >= nueva.first_key(self.schema) else page
        slot = destino.insert_sorted(payload, key, self.schema)
        self.ovf.write_page(page)
        self.ovf.write_page(nueva)
        return RID(destino.page_id, slot, "overflow")

    def _new_ovf_page(self) -> Page:
        return self.ovf.new_page(PageType.OVERFLOW)

    # --------------------------------------------------------------------- rango

    def range_search(self, lo, hi) -> Iterator[tuple]:
        if self.page_count == 0 or lo > hi:
            return
        p, page = self._locate(lo)
        while p < self.page_count:
            if page is None:
                page = self._read_main(p)
            if page.get_low_key(self.schema) > hi:
                return
            for raw in self._merged(page):
                k = self.schema.key_from_bytes(raw)
                if k > hi:
                    return
                if k >= lo:
                    yield self.schema.unpack(raw)
            p += 1
            page = None

    # -------------------------------------------------------------------- borrado

    def delete(self, key) -> bool:
        hallazgo = self._find_rid(key)
        if hallazgo is None:
            return False
        rid, page = hallazgo

        page.delete_dense(rid.slot)
        if rid.area == "main":
            self._write_main(page)
        else:
            self.n_overflow -= 1
            if page.record_count == 0:
                self._unlink_overflow(page)
            else:
                self.ovf.write_page(page)
        self.n_records -= 1
        return True

    def _unlink_overflow(self, page: Page) -> None:
        """Saca una pagina de overflow vacia de su cadena y la devuelve a la free-list."""
        prev_id, next_id = page.prev_page_id, page.next_page_id

        if prev_id == NULL_PAGE:
            # Era la cabeza: hay que actualizar la pagina principal duena.
            p = self._locate_page_by_chain_head(page.page_id)
            if p is not None:
                principal = self._read_main(p)
                principal.aux_page_id = next_id
                self._write_main(principal)
        else:
            anterior = self.ovf.read_page(prev_id)
            anterior.next_page_id = next_id
            self.ovf.write_page(anterior)

        if next_id != NULL_PAGE:
            siguiente = self.ovf.read_page(next_id)
            siguiente.prev_page_id = prev_id
            self.ovf.write_page(siguiente)

        self.ovf.free_page(page.page_id)

    def _locate_page_by_chain_head(self, head_id: int) -> int | None:
        for p in range(self.page_count):
            if self._read_main(p).aux_page_id == head_id:
                return p
        return None

    # -------------------------------------------------------------- carga masiva

    def bulk_load(self, records: Iterable[tuple]) -> None:
        """Escribe paginas secuencialmente al fill factor. Cero lecturas.

        Sin esto, montar los datasets de 500k registros costaria millones de
        lecturas de pagina y los Experimentos 2, 3 y 4 no serian viables.
        """
        ordenados = sorted(records, key=self.schema.key_of)
        if not ordenados:
            return
        if self.page_count:
            raise ValueError("bulk_load requiere un archivo vacio; usa reorganize()")

        total = math.ceil(len(ordenados) / max(1, int(self._capacity * self.fill_factor)))
        i, indice = 0, 0
        while i < len(ordenados):
            lote = ordenados[i : i + self._page_budget(indice, total)]
            page = self._new_main_page()
            # low_key(0) es el centinela -inf para que absorba cualquier clave
            # por debajo del minimo actual.
            page.set_low_key(None if i == 0 else self.schema.key_of(lote[0]), self.schema)
            for registro in lote:
                page.insert_sorted(
                    self.schema.pack(registro), self.schema.key_of(registro), self.schema
                )
            self._write_main(page)
            i += len(lote)
            indice += 1

        self.n_records = len(ordenados)
        self.n_overflow = 0

    # ------------------------------------------------------------ reorganizacion

    def reorganize(self) -> ReorgStats:
        import time

        t0 = time.perf_counter()
        r0, w0 = self.counter.disk_reads, self.counter.disk_writes
        pages_before = self.page_count
        ovf_before = self.ovf.page_count - 1

        registros = list(self.scan())

        tmp_path = f"{self._main_path}.tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        tmp = Pager(
            tmp_path,
            record_size=self.schema.record_size,
            schema=self.schema,
            page_size=self.page_size,
            cache_size=self.cache_size,
            counter=self.counter,
        )
        total = math.ceil(len(registros) / max(1, int(self._capacity * self.fill_factor)))
        i, indice = 0, 0
        while i < len(registros):
            lote = registros[i : i + self._page_budget(indice, total)]
            page = tmp.new_page(PageType.DATA)
            page.set_low_key(None if i == 0 else self.schema.key_of(lote[0]), self.schema)
            for registro in lote:
                page.insert_sorted(
                    self.schema.pack(registro), self.schema.key_of(registro), self.schema
                )
            tmp.write_page(page)
            i += len(lote)
            indice += 1
        tmp.user_a = len(registros)
        tmp.user_b = 0
        tmp.close()

        self.main.close()
        self.ovf.close()
        # os.replace es atomico en Windows y POSIX: si el proceso muere a mitad
        # de la reescritura, el archivo original queda intacto.
        os.replace(tmp_path, self._main_path)
        os.remove(self._ovf_path)
        self._open_pagers()

        pages_after = self.page_count
        return ReorgStats(
            records=len(registros),
            pages_before=pages_before,
            pages_after=pages_after,
            overflow_pages_freed=ovf_before,
            disk_reads=self.counter.disk_reads - r0,
            disk_writes=self.counter.disk_writes - w0,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # --------------------------------------------------- politicas (spec seccion 5)

    @property
    def overflow_ratio(self) -> float:
        """Fraccion de registros que viven en el area de overflow."""
        return self.n_overflow / max(1, self.n_records)

    def _should_reorganize(self) -> bool:
        """DECISION ABIERTA 1: cuando dispara la reorganizacion automatica.

        Datos medidos sobre 100k registros, B=4096, 3000 inserciones posteriores:

            estado                     lecturas por busqueda
            recien reorganizado                 11.0
            3% en overflow (disperso)           40.5
            3% en overflow (monotono)           88.9

        Y una reorganizacion completa de 120k registros cuesta ~2.3k lecturas
        + 4.3k escrituras (~1.8 s).

        Opciones:
          A) ratio:   return self.overflow_ratio > 0.20
          B) cadena:  llevar la longitud maxima de cadena y comparar con un tope
          C) manual:  return False  (actual; solo POST /api/tables/reorganize)

        Trade-off: un umbral bajo mantiene las busquedas cerca de log2(P) pero
        paga reescrituras completas frecuentes; uno alto amortiza ese costo pero
        deja degradar las busquedas. Ojo con el efecto secundario: si el umbral
        es muy bajo, el Experimento 1 (costo de insercion masiva) va a medir
        sobre todo reorganizaciones, no inserciones.

        TODO: elegir politica e implementarla.
        """
        return False

    def _page_budget(self, indice_pagina: int, total_paginas: int) -> int:
        """DECISION ABIERTA 2: cuantos registros pone reorganize en cada pagina.

        Actualmente es uniforme: floor(capacity * fill_factor), el 75% que pide
        el enunciado. Con capacity=75 son 56 registros y 19 slots libres por
        pagina, que absorben las siguientes ~19 inserciones de ese rango sin
        tocar overflow.

        Alternativa a considerar: si las claves crecen de forma monotona (ids
        autoincrementales, timestamps), todas las inserciones caen en la ULTIMA
        pagina y las 19 holguras de las demas nunca se usan. Ahi conviene
        llenar las primeras paginas al 80% y dejar la ultima mucho mas vacia.
        La medicion de arriba muestra el costo de no hacerlo: 88.9 lecturas por
        busqueda con claves monotonas frente a 40.5 con claves dispersas.

        TODO: decidir si se mantiene uniforme o se sesga hacia el final.
        """
        return max(1, int(self._capacity * self.fill_factor))

    # ---------------------------------------------------------------------- misc

    def flush(self) -> None:
        self.main.flush()
        self.ovf.flush()

    def close(self) -> None:
        self.main.close()
        self.ovf.close()

    def __enter__(self) -> "SequentialFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SequentialFile({self.base_path}, P={self.page_count}, "
            f"n={self.n_records}, overflow={self.n_overflow})"
        )
