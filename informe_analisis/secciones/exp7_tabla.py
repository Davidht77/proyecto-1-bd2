"""Convierte benchmarks/results/exp7_*.csv en tablas LaTeX (secciones/exp7_tabla.tex).

    python informe_analisis/secciones/exp7_tabla.py

Asi los numeros del informe salen del CSV y no se copian a mano.
"""

import csv
import pathlib

RES = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "results"
OUT = pathlib.Path(__file__).resolve().parent / "exp7_tabla.tex"


def f(x, nd=1):
    return f"{float(x):,.{nd}f}".replace(",", r"\,").replace(".", ",")


def pct(x):
    return f"{100 * float(x):.1f}".replace(".", ",")


def metodo7a(r):
    return {"bovw": f"BoVW, $K={r['param']}$", "resnet_exact": "ResNet-18, k-NN exacto",
            "resnet_ivf": f"ResNet-18, IVF probes={r['param']}"}[r["method"]]


def metodo7b(r):
    return {"tfidf": "TF-IDF + coseno", "bm25": "BM25", "sbert_exact": "SBERT, k-NN exacto",
            "sbert_ivf": f"SBERT, IVF probes={r['param']}"}[r["method"]]


partes = []
p = RES / "exp7_imagenes.csv"
if p.exists():
    filas = list(csv.DictReader(p.open(encoding="utf-8")))
    n, nc, nq = filas[0]["n_images"], filas[0]["n_categories"], filas[0]["n_queries"]
    cuerpo = "\n".join(
        f"{metodo7a(r)} & {pct(r['p_at_5'])} & {pct(r['p_at_10'])} & {f(r['extract_ms_per_image'])} & "
        f"{f(r['avg_query_ms'])} & {f(r['avg_reads'], 0)} & {f(r['avg_index_reads'], 0)} / {f(r['avg_table_reads'], 0)} & "
        f"{f(int(r['index_bytes']) / 1024, 0)} \\\\"
        for r in filas)
    partes.append(f"""\\begin{{table}}[H]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
Método & P@5 (\\%) & P@10 (\\%) & extracción (ms/img) & consulta (ms) & bloques & índice / tabla & índice (KB) \\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Experimento 7a (\\code{{exp7\\_imagenes.csv}}): Caltech-101, {n} imágenes de {nc} categorías, {nq} consultas. Precisión = fracción del top-$k$ con la misma categoría que la consulta (sin contar la propia imagen).}}
\\label{{tab:exp7a}}
\\end{{table}}""")

p = RES / "exp7_texto.csv"
if p.exists():
    filas = list(csv.DictReader(p.open(encoding="utf-8")))
    n, nq = filas[0]["n_docs"], filas[0]["n_queries"]
    cuerpo = "\n".join(
        f"{metodo7b(r)} & {pct(r['hit_at_1'])} & {pct(r['hit_at_10'])} & {f(r['mrr_at_10'], 3)} & "
        f"{f(r['extract_ms_per_query'])} & {f(r['avg_query_ms'])} & {f(r['avg_reads'], 0)} & "
        f"{f(r['avg_index_reads'], 0)} / {f(r['avg_table_reads'], 0)} \\\\"
        for r in filas)
    partes.append(f"""\\begin{{table}}[H]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
Método & hit@1 (\\%) & hit@10 (\\%) & MRR@10 & consulta$\\to$vector (ms) & búsqueda (ms) & bloques & índice / tabla \\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Experimento 7b (\\code{{exp7\\_texto.csv}}): Wikipedia es/en, {n} artículos, {nq} consultas de ítem conocido (la consulta es el título; acierto = recuperar ese artículo).}}
\\label{{tab:exp7b}}
\\end{{table}}""")

OUT.write_text("\n\n".join(partes) + "\n", encoding="utf-8")
print(f"-> {OUT} ({len(partes)} tablas)")
