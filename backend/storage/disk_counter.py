"""Instrumentacion de I/O fisico exigida por el enunciado (seccion 3.1).

Un unico DiskCounter se comparte entre todos los Pager de una misma tabla
(principal, overflow e indices) para que el desglose por consulta sea total.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Metrics:
    disk_reads: int
    disk_writes: int
    bytes_read: int
    bytes_written: int
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "disk_reads": self.disk_reads,
            "disk_writes": self.disk_writes,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class MeasureBox:
    """Recipiente que `DiskCounter.measure()` rellena al salir del bloque."""

    __slots__ = ("metrics",)

    def __init__(self) -> None:
        self.metrics: Metrics | None = None


class DiskCounter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.disk_reads = 0
        self.disk_writes = 0
        self.bytes_read = 0
        self.bytes_written = 0

    def record_read(self, nbytes: int) -> None:
        self.disk_reads += 1
        self.bytes_read += nbytes

    def record_write(self, nbytes: int) -> None:
        self.disk_writes += 1
        self.bytes_written += nbytes

    @contextmanager
    def measure(self) -> Iterator[MeasureBox]:
        """Mide los deltas de una operacion sin perturbar los acumulados."""
        r0, w0 = self.disk_reads, self.disk_writes
        br0, bw0 = self.bytes_read, self.bytes_written
        t0 = time.perf_counter()
        box = MeasureBox()
        try:
            yield box
        finally:
            # En el finally: si la consulta lanza, las metricas parciales
            # igual quedan disponibles para el endpoint REST.
            box.metrics = Metrics(
                disk_reads=self.disk_reads - r0,
                disk_writes=self.disk_writes - w0,
                bytes_read=self.bytes_read - br0,
                bytes_written=self.bytes_written - bw0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

    def snapshot(self) -> Metrics:
        return Metrics(
            self.disk_reads, self.disk_writes, self.bytes_read, self.bytes_written, 0.0
        )

    def __repr__(self) -> str:
        return f"DiskCounter(reads={self.disk_reads}, writes={self.disk_writes})"
