"""Corre los 4 experimentos del enunciado (seccion 4) a escala real y
vuelca los resultados a CSV bajo benchmarks/results/ para el informe.

Uso: python -m backend.benchmarks.run_all
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path

from backend.benchmarks import experiment1, experiment2, experiment3, experiment4

RESULTS_DIR = Path("benchmarks/results")


def _volcar(nombre: str, filas: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not filas:
        return
    path = RESULTS_DIR / f"{nombre}.csv"
    campos = list(asdict(filas[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in filas:
            writer.writerow(asdict(fila))
    print(f"  -> {path} ({len(filas)} filas)", flush=True)


def main() -> None:
    experimentos = [
        ("exp1_insercion", experiment1.run),
        ("exp2_igualdad", experiment2.run),
        ("exp3_rango", experiment3.run),
        ("exp4_bloque", experiment4.run),
    ]
    for nombre, run in experimentos:
        print(f"[{nombre}] arrancando...", flush=True)
        t0 = time.perf_counter()
        resultados = run()
        elapsed = time.perf_counter() - t0
        print(f"[{nombre}] listo en {elapsed:.1f}s", flush=True)
        _volcar(nombre, resultados)


if __name__ == "__main__":
    main()
