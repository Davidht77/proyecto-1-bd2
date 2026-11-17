"""Dataset de imagenes del modulo multimedia: Caltech-101.

Descarga desde el repositorio oficial de Caltech (data.caltech.edu), que
publica `caltech-101.zip` con `101_ObjectCategories.tar.gz` adentro: 9 145
imagenes en 101 categorias mas una carpeta `BACKGROUND_Google` que se excluye
porque no es una clase. Es el benchmark clasico de Bag of Visual Words, asi
que es el dataset natural para el Experimento 7.

    python -m backend.multimedia.datasets --root data/datasets
"""

from __future__ import annotations

import argparse
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
EXCLUIR = {"BACKGROUND_Google"}


@dataclass(frozen=True)
class ImageItem:
    category: str
    relpath: str  # "accordion/image_0001.jpg", relativo a la carpeta 101_ObjectCategories

    def path(self, root: str | Path) -> Path:
        return Path(root) / self.relpath


def categories_dir(root: str | Path) -> Path:
    return Path(root) / "caltech101" / "101_ObjectCategories"


def is_ready(root: str | Path) -> bool:
    d = categories_dir(root)
    return d.is_dir() and any(d.iterdir())


def download(root: str | Path) -> Path:
    """Descarga y extrae Caltech-101 bajo <root>/caltech101. Idempotente."""
    root = Path(root)
    if is_ready(root):
        return categories_dir(root)
    import requests  # solo para la descarga: verifica certificados con certifi

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "caltech-101.zip"
    if not zip_path.exists():
        with requests.get(CALTECH_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    raw = root / "caltech101_raw"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
    for dirpath, _, files in os.walk(raw):
        if "101_ObjectCategories.tar.gz" in files:
            with tarfile.open(Path(dirpath) / "101_ObjectCategories.tar.gz") as t:
                t.extractall(root / "caltech101")
    if not is_ready(root):
        raise RuntimeError("la descarga de Caltech-101 no dejo la carpeta 101_ObjectCategories")
    return categories_dir(root)


def list_images(root: str | Path, min_per_category: int = 1, max_per_category: int | None = None,
                categories: list[str] | None = None) -> list[ImageItem]:
    """Imagenes ordenadas de forma determinista (categoria, nombre)."""
    base = categories_dir(root)
    out: list[ImageItem] = []
    for cat in sorted(os.listdir(base)):
        if cat in EXCLUIR or not (base / cat).is_dir():
            continue
        if categories and cat not in categories:
            continue
        archivos = sorted(f for f in os.listdir(base / cat) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(archivos) < min_per_category:
            continue
        if max_per_category:
            archivos = archivos[:max_per_category]
        out.extend(ImageItem(cat, f"{cat}/{f}") for f in archivos)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/datasets")
    args = ap.parse_args()
    d = download(args.root)
    items = list_images(args.root)
    cats = {i.category for i in items}
    print(f"{d}: {len(items)} imagenes en {len(cats)} categorias")
