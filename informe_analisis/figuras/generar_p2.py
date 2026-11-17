"""Figuras del Entregable 2 a partir de benchmarks/results/*.csv.

    python informe_analisis/figuras/generar_p2.py

Genera (si existe el CSV correspondiente):
    exp7_precision.pdf      Precision@5 / @10 por metodo (BoVW K vs ResNet exacto / IVF)
    exp7_costos.pdf         extraccion por imagen, latencia de consulta y bloques leidos
    exp7_texto.pdf          hit@10 y MRR de TF-IDF / BM25 / SBERT (7b)
    exp8_recall.pdf         Recall@10 vs probes con latencia y bloques (IVF)
    exp8_hibrido.pdf        PRE vs POST segun selectividad
Las figuras del Experimento 5 las produce generar.py (modulo espacial).
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
OUT = Path(__file__).resolve().parent
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6c5ce7"]
MARCAS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def leer(nombre):
    path = RES / nombre
    if not path.exists():
        print(f"  (sin {nombre}: se omite)")
        return None
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / f"{nombre}.pdf")
    plt.close(fig)
    print(f"  -> {nombre}.pdf")


def etiqueta7(r):
    if r["method"] == "bovw":
        return f"BoVW K={r['param']}"
    if r["method"] == "resnet_exact":
        return "ResNet-18\nexacto"
    return f"ResNet-18\nIVF p={r['param']}"


# ---------------------------------------------------------------- Exp 7a

filas = leer("exp7_imagenes.csv")
if filas:
    xs = [etiqueta7(r) for r in filas]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    w = 0.38
    pos = range(len(xs))
    ax.bar([p - w / 2 for p in pos], [100 * float(r["p_at_5"]) for r in filas], w, color=COLORES[0], label="Precision@5")
    ax.bar([p + w / 2 for p in pos], [100 * float(r["p_at_10"]) for r in filas], w, color=COLORES[1], label="Precision@10")
    ax.set_xticks(list(pos), xs, fontsize=8)
    ax.set_ylabel("Precisión (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8)
    guardar(fig, "exp7_precision")

    # barras horizontales: las etiquetas de los 7 metodos no caben bajo el eje x
    cortas = [etiqueta7(r).replace(chr(10), " ") for r in filas]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    series = [
        ("extract_ms_per_image", "Extracción por imagen (ms)", False),
        ("avg_query_ms", "Latencia de consulta (ms, log)", True),
        ("avg_reads", "Bloques leídos por consulta (log)", True),
    ]
    ypos = list(range(len(cortas)))[::-1]
    for ax, (campo, titulo, log) in zip(axes, series):
        ys = [float(r[campo]) for r in filas]
        colores = [COLORES[0] if r["method"] == "bovw" else COLORES[1] for r in filas]
        ax.barh(ypos, ys, color=colores, height=0.6)
        ax.set_title(titulo, fontsize=9)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        for y, v in zip(ypos, ys):
            ax.annotate(f"{v:,.0f}" if v >= 100 else f"{v:.1f}", (v, y), textcoords="offset points",
                        xytext=(3, -3), fontsize=7)
    axes[0].set_yticks(ypos, cortas, fontsize=8)
    guardar(fig, "exp7_costos")

# ---------------------------------------------------------------- Exp 7b

filas = leer("exp7_texto.csv")
if filas:
    def et(r):
        return {"tfidf": "TF-IDF\ncoseno", "bm25": "BM25", "sbert_exact": "SBERT\nexacto"}.get(
            r["method"], f"SBERT\nIVF p={r['param']}")

    xs = [et(r) for r in filas]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].bar(xs, [100 * float(r["hit_at_10"]) for r in filas], color=COLORES[2], width=0.6, label="hit@10")
    axes[0].bar(xs, [100 * float(r["hit_at_1"]) for r in filas], color=COLORES[0], width=0.6, label="hit@1")
    axes[0].set_ylabel("Aciertos (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", visible=False)
    axes[1].bar(xs, [float(r["avg_reads"]) for r in filas], color=COLORES[1], width=0.6)
    axes[1].set_ylabel("Bloques leídos por consulta")
    axes[1].set_yscale("log")
    axes[1].grid(axis="x", visible=False)
    for ax in axes:
        ax.set_xticks(range(len(xs)), xs, fontsize=7)
    guardar(fig, "exp7_texto")

# ----------------------------------------------------------------- Exp 8

filas = leer("exp8_ann.csv")
if filas:
    exacto = next(r for r in filas if r["method"] == "exact")
    ivf = sorted((r for r in filas if r["method"] == "ivf"), key=lambda r: int(r["probes"]))
    probes = [int(r["probes"]) for r in ivf]
    fig, ax1 = plt.subplots(figsize=(6.0, 3.4))
    ax1.plot(probes, [100 * float(r["recall_at_10"]) for r in ivf], marker="o", color=COLORES[0], label="Recall@10 (%)")
    ax1.set_xscale("log")
    ax1.set_xticks(probes, [str(p) for p in probes])
    ax1.set_xlabel("probes (celdas sondeadas)")
    ax1.set_ylabel("Recall@10 (%)", color=COLORES[0])
    ax1.set_ylim(0, 112)
    ax2 = ax1.twinx()
    ax2.plot(probes, [float(r["avg_reads"]) for r in ivf], marker="s", color=COLORES[1], label="bloques leídos")
    ax2.axhline(float(exacto["avg_reads"]), color=COLORES[1], linestyle="--", linewidth=1)
    ax2.annotate(f"k-NN exacto: {float(exacto['avg_reads']):.0f} bloques, {float(exacto['avg_ms']):.0f} ms",
                 (probes[-1], float(exacto["avg_reads"])), textcoords="offset points", xytext=(-14, -40),
                 fontsize=7.5, ha="right", color=COLORES[1])
    ax2.set_yscale("log")
    ax2.set_ylabel("Bloques leídos por consulta (log)", color=COLORES[1])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    for p, r in zip(probes, ivf):
        ax1.annotate(f"{float(r['avg_ms']):.0f} ms", (p, 100 * float(r["recall_at_10"])), textcoords="offset points",
                     xytext=(0, 7), fontsize=7, ha="center")
    guardar(fig, "exp8_recall")

filas = leer("exp8_hibrido.csv")
if filas:
    s = defaultdict(list)
    for r in filas:
        s[r["strategy"]].append((100 * float(r["selectivity"]), float(r["avg_reads"]), float(r["avg_ms"])))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for i, (estr, et) in enumerate((("PRE", "Pre-filtering (B+ → k-NN exacto)"), ("POST", "Post-filtering (IVF → filtro)"))):
        pts = sorted(s[estr])
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker=MARCAS[i], color=COLORES[i], label=et)
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker=MARCAS[i], color=COLORES[i], label=et)
    for ax, yl in zip(axes, ("Bloques leídos por consulta", "Latencia (ms)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 1, 5, 25, 100], ["0,1", "1", "5", "25", "100"])
        ax.set_xlabel("Selectividad del filtro (% de filas)")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=7.5)
    guardar(fig, "exp8_hibrido")

print("listo")
