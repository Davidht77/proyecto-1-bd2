"""Descarga articulos de Wikipedia (wikimedia/wikipedia, dump 20231101) a un CSV.

Cada fila es (lang, id, title, text). `text` es el inicio del articulo sin
saltos de linea y cortado a MAX_TEXT_BYTES, porque la tabla lo guarda en
CHAR(1000). Usa la API de filas de Hugging Face (100 por pedido), sin
dependencias extra. Cada pedido se reintenta con espera creciente: son ~1000
pedidos para 100k articulos y un corte de red no debe tirar la descarga.

    python -m backend.text.wikipedia --n 500 --out tests/fixtures/wiki_muestra.csv

La API de filas limita los pedidos por IP (HTTP 429), asi que para 100k se baja
el primer parquet de cada idioma (688 MB es, 420 MB en), se leen los primeros
n articulos con pyarrow y el parquet se borra. pyarrow solo se usa aqui:

    python -m backend.text.wikipedia --parquet --n 50000 --langs es --out data/text/wiki_es.csv
"""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request

MAX_TEXT_BYTES = 1000
_API = ("https://datasets-server.huggingface.co/rows?dataset=wikimedia%2Fwikipedia"
        "&config=20231101.{lang}&split=train&offset={offset}&length={length}")
_PARQUET = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.{lang}/train-00000-of-{shards}.parquet"
_SHARDS = {"es": "00013", "en": "00041"}


def clip(text: str) -> str:
    return " ".join(text.split()).encode("utf-8")[:MAX_TEXT_BYTES].decode("utf-8", "ignore")


def _rows(url: str, intentos: int = 6) -> list:
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["rows"]
        except (urllib.error.URLError, TimeoutError):
            if i == intentos - 1:
                raise
            time.sleep(2 ** i)


def fetch(lang: str, n: int):
    for offset in range(0, n, 100):
        for row in _rows(_API.format(lang=lang, offset=offset, length=min(100, n - offset))):
            yield lang, row["row"]["id"], row["row"]["title"], clip(row["row"]["text"])


def from_parquet(path: str, lang: str, n: int):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["id", "title", "text"]):
        for row in batch.to_pylist():
            if n == 0:
                return
            n -= 1
            yield lang, row["id"], row["title"], clip(row["text"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="articulos por idioma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="es,en")
    ap.add_argument("--parquet", action="store_true", help="bajar el parquet en vez de usar la API de filas")
    args = ap.parse_args()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "id", "title", "text"])
        for lang in args.langs.split(","):
            if not args.parquet:
                w.writerows(fetch(lang, args.n))
                continue
            tmp = f"{args.out}.{lang}.parquet"
            urllib.request.urlretrieve(_PARQUET.format(lang=lang, shards=_SHARDS[lang]), tmp)
            w.writerows(from_parquet(tmp, lang, args.n))
            os.remove(tmp)
