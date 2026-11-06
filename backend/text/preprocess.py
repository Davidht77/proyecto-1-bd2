"""Texto -> lista de terminos: minusculas, solo letras, sin stopwords y con stemming.

Los repetidos se conservan porque SPIMI cuenta el tf a partir de esta lista.
Cada termino se corta a MAX_TERM_BYTES porque el vocabulario lo guarda en un
CHAR de ancho fijo.
"""

import re

import snowballstemmer

MAX_TERM_BYTES = 24

_WORD = re.compile(r"[^\W\d_]+")
_STEMMER = snowballstemmer.stemmer("english")
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours yourself
yourselves
""".split())


def preprocess(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
    return [t.encode("utf-8")[:MAX_TERM_BYTES].decode("utf-8", "ignore")
            for t in _STEMMER.stemWords(words)]
