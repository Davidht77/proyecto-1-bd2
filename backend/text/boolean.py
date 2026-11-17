"""Consultas booleanas sobre listas de posteo ordenadas (enunciado 5.2.3).

Las listas del indice invertido estan ordenadas por RID (el orden en que SPIMI
recorrio la tabla), asi que AND, OR y AND-NOT se resuelven con la mezcla de dos
punteros en O(n + m): cada lista se recorre una sola vez y nunca hace falta un
conjunto en memoria.

Gramatica de la consulta (operadores en MAYUSCULAS, como en el enunciado):

    expr    := termino_y ( OR termino_y )*
    termino_y := unario ( [AND] unario )*      -- palabras seguidas = AND implicito
    unario  := NOT unario | '(' expr ')' | palabra

    'aprendizaje AND profundo AND NOT basico'
    '(redes OR neuronas) AND NOT biologia'

Una consulta sin operadores no es booleana: es una consulta de ranking (las
palabras se combinan con OR y el orden lo da el score). `is_boolean` decide.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

OPERATORS = ("AND", "OR", "NOT")
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


class BooleanQueryError(ValueError):
    """Consulta booleana mal formada (por ejemplo, un NOT sin termino positivo)."""


# --------------------------------------------------------------- mezclas O(n+m)


def intersect(a: Sequence, b: Sequence) -> list:
    """AND: elementos presentes en ambas listas ordenadas."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence, b: Sequence) -> list:
    """OR: todos los elementos, sin repetir, en orden."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence, b: Sequence) -> list:
    """AND-NOT: elementos de `a` que no estan en `b`."""
    out, i, j = [], 0, 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# ------------------------------------------------------------------ parser


@dataclass(frozen=True)
class Term:
    word: str


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class And:
    children: tuple


@dataclass(frozen=True)
class Or:
    children: tuple


def is_boolean(query: str) -> bool:
    return any(tok in OPERATORS or tok in "()" for tok in _TOKEN.findall(query))


def parse(query: str):
    tokens = _TOKEN.findall(query)
    if not tokens:
        raise BooleanQueryError("la consulta MATCH esta vacia")
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def expr():
        hijos = [termino_y()]
        while peek() == "OR":
            take()
            hijos.append(termino_y())
        return hijos[0] if len(hijos) == 1 else Or(tuple(hijos))

    def termino_y():
        hijos = [unario()]
        while peek() is not None and peek() not in ("OR", ")"):
            if peek() == "AND":
                take()
            hijos.append(unario())
        return hijos[0] if len(hijos) == 1 else And(tuple(hijos))

    def unario():
        tok = peek()
        if tok is None:
            raise BooleanQueryError("la consulta termina con un operador")
        if tok == "NOT":
            take()
            return Not(unario())
        if tok == "(":
            take()
            dentro = expr()
            if peek() != ")":
                raise BooleanQueryError("falta cerrar un parentesis")
            take()
            return dentro
        if tok in ("AND", "OR", ")"):
            raise BooleanQueryError(f"«{tok}» fuera de lugar")
        return Term(take())

    arbol = expr()
    if pos != len(tokens):
        raise BooleanQueryError(f"sobra «{tokens[pos]}» al final de la consulta")
    return arbol


def positive_words(node) -> list[str]:
    """Palabras que aportan al score: las que no estan bajo un NOT."""
    if isinstance(node, Term):
        return [node.word]
    if isinstance(node, Not):
        return []
    return [w for c in node.children for w in positive_words(c)]


def all_words(node) -> list[str]:
    if isinstance(node, Term):
        return [node.word]
    if isinstance(node, Not):
        return all_words(node.child)
    return [w for c in node.children for w in all_words(c)]


# --------------------------------------------------------------- evaluacion


def evaluate(node, fetch: Callable[[str], list | None]) -> list | None:
    """Devuelve la lista ordenada de documentos que cumplen `node`.

    `fetch(palabra)` da la lista de posteo de la palabra, o None si la palabra
    no aporta nada (una stopword): en un AND se ignora y en un OR no suma.
    """
    if isinstance(node, Term):
        return fetch(node.word)
    if isinstance(node, Not):
        raise BooleanQueryError(
            "NOT necesita un termino positivo al lado (usa 'a AND NOT b'): "
            "el complemento de una lista no se puede recorrer en el indice"
        )
    if isinstance(node, Or):
        listas = [r for r in (evaluate(c, fetch) for c in node.children) if r is not None]
        if not listas:
            return None
        out = listas[0]
        for lst in listas[1:]:
            out = union(out, lst)
        return out
    # And: primero las intersecciones (de la lista mas corta a la mas larga, asi
    # el resultado intermedio nunca crece) y despues las restas de los NOT.
    positivas = [evaluate(c, fetch) for c in node.children if not isinstance(c, Not)]
    negativas = [evaluate(c.child, fetch) for c in node.children if isinstance(c, Not)]
    positivas = [p for p in positivas if p is not None]
    if not positivas:
        if negativas:
            raise BooleanQueryError("NOT necesita un termino positivo al lado (usa 'a AND NOT b')")
        return None
    positivas.sort(key=len)
    out = positivas[0]
    for lst in positivas[1:]:
        if not out:
            break
        out = intersect(out, lst)
    for lst in negativas:
        if lst:
            out = difference(out, lst)
    return out
