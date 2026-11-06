"""Descarga articulos de Wikipedia (wikimedia/wikipedia, dump 20231101) a un CSV.

Cada fila es (lang, id, title, text). `text` es el inicio del articulo sin
saltos de linea y cortado a MAX_TEXT_BYTES, porque la tabla lo guarda en
CHAR(1000). Usa la API de filas de Hugging Face (100 por pedido), sin
dependencias extra.

    python -m backend.text.wikipedia --n 500 --out tests/fixtures/wiki_muestra.csv
"""

import argparse
import csv
import json
import urllib.request

MAX_TEXT_BYTES = 1000
_API = ("https://datasets-server.huggingface.co/rows?dataset=wikimedia%2Fwikipedia"
        "&config=20231101.{lang}&split=train&offset={offset}&length={length}")


def clip(text: str) -> str:
    return " ".join(text.split()).encode("utf-8")[:MAX_TEXT_BYTES].decode("utf-8", "ignore")


def fetch(lang: str, n: int):
    for offset in range(0, n, 100):
        url = _API.format(lang=lang, offset=offset, length=min(100, n - offset))
        with urllib.request.urlopen(url, timeout=60) as r:
            rows = json.load(r)["rows"]
        for row in rows:
            yield lang, row["row"]["id"], row["row"]["title"], clip(row["row"]["text"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="articulos por idioma")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "id", "title", "text"])
        for lang in ("es", "en"):
            w.writerows(fetch(lang, args.n))
