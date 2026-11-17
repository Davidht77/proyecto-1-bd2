"""Descarga reanudable del corpus de Wikipedia para la aplicacion RAG.

Reusa el cargador del modulo textual (backend/text/wikipedia.py: misma API de
filas de Hugging Face, mismo recorte a CHAR(1000)) y agrega lo que hace falta
para 15 000 articulos: reanudar desde donde quedo el CSV y respetar el limite
de tasa del servidor (HTTP 429) con espera exponencial.

    python -m backend.rag.download --n 7500 --out data/datasets/wiki_15k.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter

from backend.text.wikipedia import _API, clip

LANGS = ("es", "en")
PAGE = 100


def _rows(lang: str, offset: int, length: int, retries: int = 12) -> list[dict]:
    url = _API.format(lang=lang, offset=offset, length=length)
    espera = 5.0
    for intento in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["rows"]
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            pausa = float(retry_after) if retry_after and retry_after.isdigit() else espera
            print(f"  HTTP {exc.code} en {lang}@{offset}: espero {pausa:.0f}s "
                  f"(intento {intento + 1}/{retries})", flush=True)
            time.sleep(pausa)
            espera = min(espera * 2, 120.0)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  red: {exc}; espero {espera:.0f}s", flush=True)
            time.sleep(espera)
            espera = min(espera * 2, 120.0)
    raise RuntimeError(f"no se pudo descargar {lang}@{offset} tras {retries} intentos")


def existing_counts(path: str) -> Counter:
    if not os.path.exists(path):
        return Counter()
    with open(path, encoding="utf-8", newline="") as f:
        return Counter(r["lang"] for r in csv.DictReader(f))


def download(out: str, n_per_lang: int, langs=LANGS, pause: float = 1.0) -> Counter:
    """Deja `out` con n_per_lang filas por idioma; si ya tiene parte, solo baja el resto."""
    hechos = existing_counts(out)
    nuevo = not os.path.exists(out) or os.path.getsize(out) == 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["lang", "id", "title", "text"])
        for lang in langs:
            offset = hechos[lang]
            while offset < n_per_lang:
                length = min(PAGE, n_per_lang - offset)
                filas = _rows(lang, offset, length)
                if not filas:
                    break
                w.writerows((lang, r["row"]["id"], r["row"]["title"], clip(r["row"]["text"])) for r in filas)
                f.flush()
                offset += len(filas)
                hechos[lang] = offset
                if offset % 1000 == 0:
                    print(f"  {lang}: {offset}/{n_per_lang}", flush=True)
                time.sleep(pause)  # cortesia con el servidor: evita el 429
    return hechos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=7500, help="articulos por idioma")
    ap.add_argument("--out", default="data/datasets/wiki_15k.csv")
    ap.add_argument("--pause", type=float, default=1.0, help="segundos entre pedidos")
    args = ap.parse_args()
    print(dict(download(args.out, args.n, pause=args.pause)))
