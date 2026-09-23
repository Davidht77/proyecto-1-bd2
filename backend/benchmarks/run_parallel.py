"""Corre los 4 experimentos del enunciado (seccion 4) a escala real, uno
por proceso del sistema operativo en paralelo, para aprovechar los nucleos
disponibles en vez de la ejecucion serial de run_all.py.

El Experimento 1 se parte en un proceso por cada N de N_VALUES (cada uno
en su propio subdirectorio bajo data/) porque SEQUENTIAL_CON_REORG a
N=500_000 es, por diseno, el mas lento de lejos: paralelizar no lo acelera
(cada insercion depende de la anterior), pero evita que bloquee a los
demas N y a los Experimentos 2-4 mientras corre.

Uso: python -m backend.benchmarks.run_parallel
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from backend.benchmarks import experiment1

RESULTS_DIR = Path("benchmarks/results")

_WORKER = """
import csv, sys
from dataclasses import asdict
from pathlib import Path

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
modulo = sys.argv[1]

if modulo == "exp1":
    n = int(sys.argv[2])
    from backend.benchmarks import experiment1 as exp
    exp.N_VALUES = [n]
    filas = exp.run(data_dir=f"data/benchmarks/exp1/n{n}")
    nombre = f"exp1_insercion_n{n}"
elif modulo == "exp2":
    from backend.benchmarks import experiment2 as exp
    filas = exp.run()
    nombre = "exp2_igualdad"
elif modulo == "exp3":
    from backend.benchmarks import experiment3 as exp
    filas = exp.run()
    nombre = "exp3_rango"
elif modulo == "exp4":
    from backend.benchmarks import experiment4 as exp
    filas = exp.run()
    nombre = "exp4_bloque"
else:
    raise SystemExit(f"modulo desconocido: {modulo}")

path = RESULTS_DIR / f"{nombre}.csv"
campos = list(asdict(filas[0]).keys())
with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    for fila in filas:
        writer.writerow(asdict(fila))
print(f"[{nombre}] listo -> {path} ({len(filas)} filas)", flush=True)
"""


def _fusionar_exp1(n_values: list[int]) -> None:
    """Concatena los CSV por-N del Experimento 1 en uno solo."""
    import csv

    filas_totales = []
    campos = None
    for n in n_values:
        path = RESULTS_DIR / f"exp1_insercion_n{n}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            campos = campos or reader.fieldnames
            filas_totales.extend(reader)
        path.unlink()
    if not filas_totales:
        return
    with (RESULTS_DIR / "exp1_insercion.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_totales)


def main() -> None:
    n_values = list(experiment1.N_VALUES)
    tareas = [["exp1", str(n)] for n in n_values] + [["exp2"], ["exp3"], ["exp4"]]

    t0 = time.perf_counter()
    procesos = [subprocess.Popen([sys.executable, "-c", _WORKER, *tarea]) for tarea in tareas]
    for p in procesos:
        p.wait()

    _fusionar_exp1(n_values)
    print(f"[run_parallel] todo listo en {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
