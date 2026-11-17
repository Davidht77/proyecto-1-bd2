"""Esquema de registros de longitud fija.

Todos los tipos del Entregable 1 (INT, FLOAT, CHAR(n)) tienen ancho fijo, lo
que permite que el slot `i` de una pagina viva siempre en el mismo offset y
que la busqueda binaria intra-pagina sea aritmetica pura.

Entregable 2 agrega VECTOR(D): D float32 contiguos (4*D bytes). Sigue siendo
de ancho fijo, asi que toda la maquinaria de paginas, bitmap y RID funciona
igual. struct aplana los D componentes en la tupla; pack/unpack los
reagrupan para que la capa superior vea un solo valor por columna.
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
    VECTOR = 4  # length = dimension D; cada componente es float32
    POINT = 5   # (lon, lat) en float64; 16 bytes


def parse_point(value) -> tuple[float, float]:
    """Normaliza un punto geografico a ``(lon, lat)``.

    Acepta una tupla/lista de dos numeros o el literal SQL ``POINT(x, y)``.
    """
    if isinstance(value, str):
        texto = value.strip()
        if texto.upper().startswith("POINT"):
            texto = texto[texto.find("(") + 1 :]
            if texto.endswith(")"):
                texto = texto[:-1]
        elif texto.startswith("(") and texto.endswith(")"):
            texto = texto[1:-1]
        partes = [p.strip() for p in texto.split(",")]
        if len(partes) != 2 or not all(partes):
            raise ValueError(f"un POINT se escribe como POINT(x, y), no {value!r}")
        try:
            return (float(partes[0]), float(partes[1]))
        except ValueError as exc:
            raise ValueError(f"componente no numerico en el punto: {exc}") from None
    try:
        x, y = value
    except (TypeError, ValueError):
        raise ValueError(f"un POINT necesita dos coordenadas, no {value!r}") from None
    return (float(x), float(y))


def parse_vector(value, dim: int | None = None) -> tuple[float, ...]:
    """Normaliza un literal vectorial: '[0.1, -2, 3e-1]', lista o tupla.

    Es el formato del enunciado (`embedding <=> '[0.12, -0.34, ...]'`), que
    llega al motor como un string SQL comun.
    """
    if isinstance(value, str):
        texto = value.strip()
        if not (texto.startswith("[") and texto.endswith("]")):
            raise ValueError(f"un vector se escribe como '[x1, x2, ...]', no {value[:30]!r}")
        cuerpo = texto[1:-1].strip()
        try:
            out = tuple(float(x) for x in cuerpo.split(",")) if cuerpo else ()
        except ValueError as exc:
            raise ValueError(f"componente no numerico en el vector: {exc}") from None
    else:
        out = tuple(float(x) for x in value)
    if dim is not None and len(out) != dim:
        raise ValueError(f"se esperaba un vector de dimension {dim}, llego uno de {len(out)}")
    return out


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    length: int = 0  # solo se usa para CHAR(n)

    def __post_init__(self) -> None:
        if self.type is ColumnType.CHAR and self.length <= 0:
            raise ValueError(f"CHAR requiere longitud positiva: {self.name}")
        if self.type is ColumnType.VECTOR and not 0 < self.length <= 4096:
            raise ValueError(f"VECTOR requiere una dimension entre 1 y 4096: {self.name}")
        if len(self.name.encode("utf-8")) > 32:
            raise ValueError(f"nombre de columna demasiado largo: {self.name}")

    @property
    def size(self) -> int:
        if self.type is ColumnType.INT:
            return 4
        if self.type is ColumnType.FLOAT:
            return 8
        if self.type is ColumnType.VECTOR:
            return 4 * self.length
        if self.type is ColumnType.POINT:
            return 16
        return self.length

    @property
    def struct_fmt(self) -> str:
        if self.type is ColumnType.INT:
            return "i"
        if self.type is ColumnType.FLOAT:
            return "d"
        if self.type is ColumnType.VECTOR:
            return f"{self.length}f"
        if self.type is ColumnType.POINT:
            return "dd"
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
        if self.pk_column.type in (ColumnType.VECTOR, ColumnType.POINT):
            raise ValueError("un VECTOR o un POINT no puede ser clave primaria")
        self._has_composite = any(
            c.type in (ColumnType.VECTOR, ColumnType.POINT) for c in self.columns
        )

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
            elif col.type is ColumnType.VECTOR:
                # struct espera los D componentes como argumentos sueltos.
                encoded.extend(parse_vector(value, col.length))
            elif col.type is ColumnType.POINT:
                encoded.extend(parse_point(value))
            elif col.type is ColumnType.INT:
                encoded.append(int(value))
            else:
                encoded.append(float(value))
        return self._struct.pack(*encoded)

    def unpack(self, raw: bytes) -> tuple:
        values = self._struct.unpack(bytes(raw[: self.record_size]))
        out = []
        if self._has_composite:
            # struct devolvio los componentes sueltos: se reagrupan por columna.
            i = 0
            for col in self.columns:
                if col.type is ColumnType.VECTOR:
                    out.append(values[i : i + col.length])
                    i += col.length
                    continue
                if col.type is ColumnType.POINT:
                    out.append(values[i : i + 2])
                    i += 2
                    continue
                value = values[i]
                i += 1
                if col.type is ColumnType.CHAR:
                    value = value.split(b"\x00", 1)[0].decode("utf-8")
                out.append(value)
            return tuple(out)
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
