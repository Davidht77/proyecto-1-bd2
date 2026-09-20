"""Esquema de registros de longitud fija.

Todos los tipos del Entregable 1 (INT, FLOAT, CHAR(n)) tienen ancho fijo, lo
que permite que el slot `i` de una pagina viva siempre en el mismo offset y
que la busqueda binaria intra-pagina sea aritmetica pura.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class ColumnType(IntEnum):
    INT = 1
    FLOAT = 2
    CHAR = 3


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    length: int = 0  # solo se usa para CHAR(n)

    def __post_init__(self) -> None:
        if self.type is ColumnType.CHAR and self.length <= 0:
            raise ValueError(f"CHAR requiere longitud positiva: {self.name}")
        if len(self.name.encode("utf-8")) > 32:
            raise ValueError(f"nombre de columna demasiado largo: {self.name}")

    @property
    def size(self) -> int:
        if self.type is ColumnType.INT:
            return 4
        if self.type is ColumnType.FLOAT:
            return 8
        return self.length

    @property
    def struct_fmt(self) -> str:
        if self.type is ColumnType.INT:
            return "i"
        if self.type is ColumnType.FLOAT:
            return "d"
        return f"{self.length}s"


class Schema:
    """Describe la forma binaria de un registro y sabe extraer su clave primaria."""

    def __init__(self, columns: Sequence[Column], pk_index: int = 0):
        if not columns:
            raise ValueError("el esquema necesita al menos una columna")
        if not 0 <= pk_index < len(columns):
            raise ValueError(f"pk_index fuera de rango: {pk_index}")

        self.columns: tuple[Column, ...] = tuple(columns)
        self.pk_index = pk_index

        # El prefijo "<" fija little-endian y desactiva el padding de alineacion,
        # de modo que record_size es exactamente la suma de los anchos.
        self._struct = struct.Struct("<" + "".join(c.struct_fmt for c in self.columns))
        self.record_size = self._struct.size

        # Struct dedicado a la PK: permite decodificar 4 u 8 bytes en vez del
        # registro completo durante la busqueda binaria.
        self.pk_offset = sum(c.size for c in self.columns[:pk_index])
        self._pk_struct = struct.Struct("<" + self.pk_column.struct_fmt)

    @property
    def pk_column(self) -> Column:
        return self.columns[self.pk_index]

    def pack(self, values: Sequence) -> bytes:
        if len(values) != len(self.columns):
            raise ValueError(
                f"se esperaban {len(self.columns)} valores, llegaron {len(values)}"
            )
        encoded = []
        for col, value in zip(self.columns, values):
            if col.type is ColumnType.CHAR:
                raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
                if len(raw) > col.length:
                    raise ValueError(
                        f"valor de {col.name} excede CHAR({col.length}): {len(raw)} bytes"
                    )
                encoded.append(raw)
            elif col.type is ColumnType.INT:
                encoded.append(int(value))
            else:
                encoded.append(float(value))
        return self._struct.pack(*encoded)

    def unpack(self, raw: bytes) -> tuple:
        values = self._struct.unpack(bytes(raw[: self.record_size]))
        out = []
        for col, value in zip(self.columns, values):
            if col.type is ColumnType.CHAR:
                out.append(value.split(b"\x00", 1)[0].decode("utf-8"))
            else:
                out.append(value)
        return tuple(out)

    def key_of(self, values: Sequence):
        return values[self.pk_index]

    def key_from_bytes(self, raw: bytes):
        """Decodifica solo la clave primaria, sin tocar el resto del registro."""
        key = self._pk_struct.unpack_from(raw, self.pk_offset)[0]
        if self.pk_column.type is ColumnType.CHAR:
            return key.split(b"\x00", 1)[0].decode("utf-8")
        return key

    def __repr__(self) -> str:
        cols = ", ".join(f"{c.name}:{c.type.name}" for c in self.columns)
        return f"Schema({cols}, pk={self.pk_column.name}, {self.record_size}B)"
