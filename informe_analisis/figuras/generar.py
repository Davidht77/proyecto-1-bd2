import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
OUT = Path(__file__).resolve().parent
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARCAS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def leer(nombre):
    with (RES / nombre).open() as f:
        return list(csv.DictReader(f))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / f"{nombre}.pdf")
    plt.close(fig)


ETIQUETAS1 = {
    "HEAP": "Heap", "SEQUENTIAL_CON_REORG": "Sequential con reorg.",
    "SEQUENTIAL_SIN_REORG": "Sequential sin reorg.", "BTREE": "Árbol B+", "HASH": "Hash extensible",
}
series = defaultdict(list)
for r in leer("exp1_insercion.csv"):
    series[r["engine"]].append((int(r["n"]), float(r["total_ms"]) / 1000, int(r["disk_writes"])))
for campo, ylabel, nombre in ((1, "Tiempo total (s)", "exp1_tiempo"), (2, "Escrituras en disco", "exp1_escrituras")):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, (k, et) in enumerate(ETIQUETAS1.items()):
        pts = sorted(series[k])
        ax.plot([p[0] for p in pts], [p[campo] for p in pts], marker=MARCAS[i], color=COLORES[i], label=et)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Registros insertados (N)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    guardar(fig, nombre)

ETIQUETAS2 = {
    "FULL_SCAN_HEAP": "Full scan\n(Heap)", "BINARY_SEARCH_SEQUENTIAL": "Búsqueda binaria\n(Sequential)",
    "BTREE": "Árbol B+", "HASH": "Hash\nextensible",
}
filas = {r["access"]: r for r in leer("exp2_igualdad.csv")}
for media, desv, ylabel, nombre in (
    ("avg_reads", "std_reads", "Lecturas por consulta", "exp2_lecturas"),
    ("avg_latency_ms", "std_latency_ms", "Latencia por consulta (ms)", "exp2_latencia"),
):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = list(ETIQUETAS2.values())
    ys = [float(filas[k][media]) for k in ETIQUETAS2]
    es = [float(filas[k][desv]) for k in ETIQUETAS2]
    ax.bar(xs, ys, color=COLORES[0], width=0.55, yerr=es, capsize=4, error_kw={"elinewidth": 1})
    ax.set_yscale("log")
    ax.set_ylabel(ylabel + " (escala log)")
    ax.grid(axis="x", visible=False)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}".rstrip("0").rstrip("."), (x, y), textcoords="offset points", xytext=(10, 3), fontsize=8)
    guardar(fig, nombre)

ETIQUETAS3 = {"FULL_SCAN_HEAP": "Full scan (Heap)", "SEQUENTIAL": "Sequential", "BTREE": "Árbol B+"}
s3 = defaultdict(list)
for r in leer("exp3_rango.csv"):
    s3[r["access"]].append((float(r["selectividad"]) * 100, int(r["disk_reads"]), float(r["latency_ms"])))
for campo, ylabel, nombre in ((1, "Lecturas en disco", "exp3_lecturas"), (2, "Latencia (ms)", "exp3_latencia")):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, (k, et) in enumerate(ETIQUETAS3.items()):
        pts = sorted(s3[k])
        ax.plot([p[0] for p in pts], [p[campo] for p in pts], marker=MARCAS[i], color=COLORES[i], label=et)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 5, 10, 25], ["0,1", "1", "5", "10", "25"])
    ax.set_xlabel("Selectividad (% de tuplas)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    guardar(fig, nombre)

f4 = leer("exp4_bloque.csv")
bs = [r["page_size"] for r in f4]
fig, axs = plt.subplots(1, 3, figsize=(7, 2.6))
for ax, campo, titulo in zip(axs, ("fanout", "height", "disk_reads"), ("Fan-out", "Altura del árbol", "Lecturas totales")):
    vals = [int(r[campo]) for r in f4]
    ax.bar(bs, vals, color=COLORES[0], width=0.55)
    ax.set_title(titulo, fontsize=9)
    ax.margins(y=0.15)
    ax.set_xlabel("Tamaño de página (B)")
    ax.grid(axis="x", visible=False)
    for x, v in zip(bs, vals):
        ax.annotate(f"{v:,}".replace(",", " "), (x, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
guardar(fig, "exp4_bloque")
