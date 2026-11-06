"""Texto -> lista de terminos: minusculas, solo letras, sin stopwords y con stemming.

Los repetidos se conservan porque SPIMI cuenta el tf a partir de esta lista.
Cada termino se corta a MAX_TERM_BYTES porque el vocabulario lo guarda en un
CHAR de ancho fijo.

Stopwords: listas de NLTK en ingles y espanol (nltk_data/corpora/stopwords),
las mismas del full-text search de PostgreSQL. Se copiaron una vez al repo
para no depender de internet.

Bilingue: cada documento usa el stemmer de su idioma, detectado por cual lista
de stopwords aparece mas (empate -> ingles). La consulta es corta y puede no
traer stopwords, asi que pasa por los dos stemmers y busca ambas raices.
"""

import re
from pathlib import Path

import snowballstemmer

MAX_TERM_BYTES = 24

_WORD = re.compile(r"[^\W\d_]+")
_LANGS = ("english", "spanish")
_STEMMERS = {lang: snowballstemmer.stemmer(lang) for lang in _LANGS}


def _load(lang: str) -> set[str]:
    with open(Path(__file__).parent / "stopwords" / f"{lang}.txt", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


_STOP = {lang: _load(lang) for lang in _LANGS}
STOPWORDS = frozenset(_STOP["english"] | _STOP["spanish"])


def _cut(term: str) -> str:
    return term.encode("utf-8")[:MAX_TERM_BYTES].decode("utf-8", "ignore")


def preprocess(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    lang = max(_LANGS, key=lambda l: sum(w in _STOP[l] for w in words))
    return [_cut(t) for t in _STEMMERS[lang].stemWords([w for w in words if w not in STOPWORDS])]


def preprocess_query(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
    return list(dict.fromkeys(_cut(s.stemWord(w)) for w in words for s in _STEMMERS.values()))
