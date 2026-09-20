"""Layout de pagina fisica: cabecera de 32 bytes + bitmap + slots de longitud fija.

La cabecera es identica para todos los tipos de pagina (datos, overflow, nodos
B+, buckets de hash). Esa uniformidad hace que el offset del area de datos sea
una sola formula para todo el motor, lo que simplifica el calculo de fan-out
del Experimento 4.
"""

from __future__ import annotations

import math
import struct
from enum import IntEnum
from typing import Iterator

from backend.storage.schema import ColumnType, Schema

NULL_PAGE = 0xFFFFFFFF
HEADER_SIZE = 32

# page_id(4) type(1) record_count(2) low_key_flag(1) free_space_offset(4)
# next(4) prev(4) aux(4) reserved(8) = 32 bytes exactos.
# El prefijo "<" desactiva el padding de alineacion de struct.
_HEADER = struct.Struct("<IBHBIIII8s")
assert _HEADER.size == HEADER_SIZE

# Los 8 bytes `reserved` guardan el low_key de SequentialFile como int64 o
# float64. El byte de padding (indice 3) hace de bandera: 0 = centinela -inf,
# 1 = hay valor. Separar la bandera del valor es lo que evita el bug clasico
# de usar un numero magico y no poder representar la clave 0.
_LOW_INT = struct.Struct("<q")
_LOW_FLOAT = struct.Struct("<d")


class PageType(IntEnum):
    FREE = 0
    DATA = 1
    OVERFLOW = 2
    BT_INTERNAL = 3
    BT_LEAF = 4
    HASH = 5


def page_capacity(page_size: int, record_size: int) -> int:
    """Cuantos registros caben en una pagina.

    El bitmap y los registros compiten por el mismo espacio, asi que la
    capacidad es autorreferente. Resolviendo la desigualdad
        HEADER + ceil(C/8) + C*R <= B
    se obtiene C = floor(8*(B-H) / (8*R+1)): el +1 del denominador es el bit
    de bitmap que cuesta cada slot.
    """
    if record_size <= 0:
        raise ValueError("record_size debe ser positivo")
    capacity = (8 * (page_size - HEADER_SIZE)) // (8 * record_size + 1)
    if capacity <= 0:
        raise ValueError(
            f"un registro de {record_size}B no cabe en una pagina de {page_size}B"
        )
    return capacity


def bitmap_size(capacity: int) -> int:
    return (capacity + 7) // 8


class Page:
    """Un bloque fisico en memoria. El Pager lo lee y escribe; esta clase lo interpreta."""

    __slots__ = ("page_id", "page_size", "record_size", "capacity", "_bitmap_size",
                 "_data_offset", "data")

    def __init__(
        self,
        page_id: int,
        page_size: int,
        record_size: int,
        data: bytes | bytearray | None = None,
    ):
        self.page_id = page_id
        self.page_size = page_size
        self.record_size = record_size
        self.capacity = page_capacity(page_size, record_size)
        self._bitmap_size = bitmap_size(self.capacity)
        self._data_offset = HEADER_SIZE + self._bitmap_size

        if data is None:
            self.data = bytearray(page_size)
            self._init_header()
        else:
            if len(data) != page_size:
                raise ValueError(
                    f"la pagina mide {len(data)}B, se esperaban {page_size}B"
                )
            self.data = bytearray(data)

    def _init_header(self) -> None:
        _HEADER.pack_into(
            self.data, 0,
            self.page_id, PageType.FREE, 0, 0, self._data_offset,
            NULL_PAGE, NULL_PAGE, NULL_PAGE, b"\x00" * 8,
        )

    # ------------------------------------------------------------------ cabecera

    def _get(self, index: int):
        return _HEADER.unpack_from(self.data, 0)[index]

    def _set(self, index: int, value) -> None:
        fields = list(_HEADER.unpack_from(self.data, 0))
        fields[index] = value
        _HEADER.pack_into(self.data, 0, *fields)

    @property
    def stored_page_id(self) -> int:
        return self._get(0)

    @property
    def page_type(self) -> PageType:
        return PageType(self._get(1))

    @page_type.setter
    def page_type(self, value: int) -> None:
        self._set(1, int(value))

    @property
    def record_count(self) -> int:
        return self._get(2)

    @record_count.setter
    def record_count(self, value: int) -> None:
        if not 0 <= value <= self.capacity:
            raise ValueError(f"record_count fuera de rango: {value}")
        self._set(2, value)

    @property
    def free_space_offset(self) -> int:
        return self._get(4)

    @free_space_offset.setter
    def free_space_offset(self, value: int) -> None:
        self._set(4, value)

    @property
    def next_page_id(self) -> int:
        return self._get(5)

    @next_page_id.setter
    def next_page_id(self, value: int) -> None:
        self._set(5, value)

    @property
    def prev_page_id(self) -> int:
        return self._get(6)

    @prev_page_id.setter
    def prev_page_id(self, value: int) -> None:
        self._set(6, value)

    @property
    def aux_page_id(self) -> int:
        return self._get(7)

    @aux_page_id.setter
    def aux_page_id(self, value: int) -> None:
        self._set(7, value)

    # -------------------------------------------------------------------- bitmap

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError(f"slot {slot} fuera de rango [0, {self.capacity})")

    def is_occupied(self, slot: int) -> bool:
        self._check_slot(slot)
        return bool(self.data[HEADER_SIZE + slot // 8] & (1 << (slot % 8)))

    def set_occupied(self, slot: int, value: bool) -> None:
        self._check_slot(slot)
        i = HEADER_SIZE + slot // 8
        mask = 1 << (slot % 8)
        if value:
            self.data[i] |= mask
        else:
            self.data[i] &= ~mask & 0xFF

    def first_free_slot(self) -> int | None:
        for slot in range(self.capacity):
            if not self.is_occupied(slot):
                return slot
        return None

    def is_full(self) -> bool:
        return self.record_count >= self.capacity

    # ----------------------------------------------------------------- registros

    def _slot_offset(self, slot: int) -> int:
        return self._data_offset + slot * self.record_size

    def read_record(self, slot: int) -> bytes:
        self._check_slot(slot)
        off = self._slot_offset(slot)
        return bytes(self.data[off : off + self.record_size])

    def write_record(self, slot: int, payload: bytes) -> None:
        self._check_slot(slot)
        if len(payload) != self.record_size:
            raise ValueError(
                f"el registro mide {len(payload)}B, se esperaban {self.record_size}B"
            )
        off = self._slot_offset(slot)
        self.data[off : off + self.record_size] = payload

    def records(self) -> Iterator[bytes]:
        """Los registros densos, en orden de slot."""
        for slot in range(self.record_count):
            yield self.read_record(slot)

    # ------------------------------------------- operaciones ordenadas (densas)
    # Asumen el invariante de densidad: los slots 0..record_count-1 estan
    # ocupados y ordenados por clave. Lo mantiene SequentialFile; HeapFile usa
    # el bitmap de forma dispersa y no llama a estos metodos.

    def keys(self, schema: Schema) -> list:
        return [schema.key_from_bytes(self.read_record(s)) for s in range(self.record_count)]

    def first_key(self, schema: Schema):
        if self.record_count == 0:
            return None
        return schema.key_from_bytes(self.read_record(0))

    def last_key(self, schema: Schema):
        if self.record_count == 0:
            return None
        return schema.key_from_bytes(self.read_record(self.record_count - 1))

    def binary_search(self, key, schema: Schema) -> tuple[bool, int]:
        """Devuelve (encontrado, posicion). Si no lo encuentra, la posicion es
        donde habria que insertarlo."""
        lo, hi = 0, self.record_count - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k = schema.key_from_bytes(self.read_record(mid))
            if k == key:
                return True, mid
            if k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return False, lo

    def insert_sorted(self, payload: bytes, key, schema: Schema) -> int:
        if self.is_full():
            raise OverflowError(f"la pagina {self.page_id} esta llena")
        _, pos = self.binary_search(key, schema)
        n = self.record_count
        if pos < n:
            # Desplaza [pos, n) un registro a la derecha. Es un slice sobre el
            # bytearray: ocurre en RAM, sin I/O adicional.
            src = self._slot_offset(pos)
            end = self._slot_offset(n)
            self.data[src + self.record_size : end + self.record_size] = self.data[src:end]
        self.write_record(pos, payload)
        self.record_count = n + 1
        self.set_occupied(n, True)
        self.free_space_offset = self._slot_offset(n + 1)
        return pos

    def delete_dense(self, slot: int) -> None:
        n = self.record_count
        if not 0 <= slot < n:
            raise IndexError(f"slot {slot} no contiene un registro activo")
        if slot < n - 1:
            src = self._slot_offset(slot + 1)
            end = self._slot_offset(n)
            self.data[self._slot_offset(slot) : end - self.record_size] = self.data[src:end]
        self.set_occupied(n - 1, False)
        self.record_count = n - 1
        self.free_space_offset = self._slot_offset(n - 1)

    # --------------------------------------------------- low_key (rango declarado)

    def get_low_key(self, schema: Schema):
        """Cota inferior del rango de claves de la pagina.

        Es el rango *declarado*, no el contenido actual: solo lo escriben
        bulk_load y reorganize. Si dependiera del primer registro, borrarlo
        haria inalcanzables las claves menores que quedaran en su cadena de
        overflow.

        La bandera vive en el byte de padding de la cabecera, no dentro de los
        8 bytes del valor: asi el centinela -inf se distingue de la clave 0 sin
        gastar rango util.
        """
        if self._get(3) == 0:
            return -math.inf
        raw = self._get(8)
        if schema.pk_column.type is ColumnType.FLOAT:
            return _LOW_FLOAT.unpack(raw)[0]
        return _LOW_INT.unpack(raw)[0]

    def set_low_key(self, key, schema: Schema) -> None:
        if key is None or key == -math.inf:
            fields = list(_HEADER.unpack_from(self.data, 0))
            fields[3] = 0
            fields[8] = b"\x00" * 8
            _HEADER.pack_into(self.data, 0, *fields)
            return
        if schema.pk_column.type is ColumnType.FLOAT:
            payload = _LOW_FLOAT.pack(float(key))
        elif schema.pk_column.type is ColumnType.INT:
            payload = _LOW_INT.pack(int(key))
        else:
            raise TypeError("low_key solo admite claves INT o FLOAT")
        fields = list(_HEADER.unpack_from(self.data, 0))
        fields[3] = 1
        fields[8] = payload
        _HEADER.pack_into(self.data, 0, *fields)

    # ---------------------------------------------------------------------- misc

    def to_bytes(self) -> bytes:
        return bytes(self.data)

    def __repr__(self) -> str:
        return (
            f"Page(id={self.page_id}, type={self.page_type.name}, "
            f"n={self.record_count}/{self.capacity})"
        )
