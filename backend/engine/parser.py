"""Parser SQL por descenso recursivo.

Gramatica soportada en el Entregable 1:

    CREATE TABLE <t> (<col> <tipo> [PRIMARY KEY], ...) [USING HEAP|SEQUENTIAL]
    DROP TABLE <t>
    INSERT INTO <t> VALUES (<v>, ...)
    SELECT * | <col>, ... FROM <t> [WHERE <cond>]
    DELETE FROM <t> WHERE <cond>
    CREATE INDEX <nombre> ON <t> (<col>) USING BTREE|HASH

    <cond> := <col> = <v>
            | <col> <op> <v> [AND <col> <op> <v>]
            | <col> BETWEEN <v> AND <v>

No se usa ninguna libreria de parsing: el enunciado exige construir el parser
desde cero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.engine.errors import SQLSyntaxError
from backend.storage.schema import Column, ColumnType

# --------------------------------------------------------------------- AST


@dataclass
class Condition:
    """Predicado del WHERE, normalizado a un rango con banderas de inclusion.

    Una igualdad es el rango cerrado [v, v]. Normalizarlo asi permite que el
    planificador trate ambos casos con la misma estructura y decida la ruta de
    acceso mirando solo `is_equality`.
    """

    column: str
    lo: Any = None
    hi: Any = None
    lo_inclusive: bool = True
    hi_inclusive: bool = True

    @property
    def is_equality(self) -> bool:
        return self.lo is not None and self.lo == self.hi and self.lo_inclusive and self.hi_inclusive


@dataclass
class CreateTable:
    table: str
    columns: list[Column]
    pk_index: int
    engine: str  # "SEQUENTIAL" | "HEAP"


@dataclass
class DropTable:
    table: str


@dataclass
class Insert:
    table: str
    values: list[Any]


@dataclass
class Select:
    table: str
    columns: list[str] = field(default_factory=list)  # vacio = *
    where: Condition | None = None
    limit: int | None = None


@dataclass
class Delete:
    table: str
    where: Condition | None = None


@dataclass
class CreateIndex:
    name: str
    table: str
    column: str
    kind: str  # "BTREE" | "HASH"


Statement = CreateTable | DropTable | Insert | Select | Delete | CreateIndex


# ---------------------------------------------------------------- tokenizer

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<comment>--[^\n]*)
    | (?P<string>'(?:[^']|'')*')
    | (?P<number>-?\d+\.\d+|-?\d+)
    | (?P<op><=|>=|<>|!=|=|<|>)
    | (?P<punct>[(),;*])
    | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {
    "CREATE", "TABLE", "DROP", "INSERT", "INTO", "VALUES", "SELECT", "FROM",
    "WHERE", "DELETE", "INDEX", "ON", "USING", "AND", "BETWEEN", "PRIMARY",
    "KEY", "INT", "FLOAT", "CHAR", "HEAP", "SEQUENTIAL", "BTREE", "HASH",
    "LIMIT",
}


@dataclass
class Token:
    kind: str  # keyword | ident | number | string | op | punct
    value: Any
    pos: int


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(sql):
        m = _TOKEN_RE.match(sql, i)
        if not m:
            raise SQLSyntaxError(f"caracter inesperado {sql[i]!r} en la posicion {i}")
        i = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind in ("ws", "comment"):
            continue
        if kind == "ident":
            upper = text.upper()
            if upper in _KEYWORDS:
                tokens.append(Token("keyword", upper, m.start()))
            else:
                tokens.append(Token("ident", text, m.start()))
        elif kind == "number":
            value = float(text) if ("." in text) else int(text)
            tokens.append(Token("number", value, m.start()))
        elif kind == "string":
            tokens.append(Token("string", text[1:-1].replace("''", "'"), m.start()))
        else:
            tokens.append(Token(kind, text, m.start()))
    return tokens


# ------------------------------------------------------------------- parser


class Parser:
    def __init__(self, sql: str):
        self.sql = sql
        self.tokens = tokenize(sql)
        self.i = 0

    # -- utilidades ---------------------------------------------------------

    def _peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> Token:
        tok = self._peek()
        if tok is None:
            raise SQLSyntaxError("la consulta termina antes de lo esperado")
        self.i += 1
        return tok

    def _accept(self, kind: str, value: Any = None) -> Token | None:
        tok = self._peek()
        if tok and tok.kind == kind and (value is None or tok.value == value):
            self.i += 1
            return tok
        return None

    def _expect(self, kind: str, value: Any = None) -> Token:
        tok = self._accept(kind, value)
        if tok is None:
            actual = self._peek()
            esperado = value or kind
            visto = repr(actual.value) if actual else "el final de la consulta"
            raise SQLSyntaxError(f"se esperaba {esperado}, se encontro {visto}")
        return tok

    def _expect_name(self) -> str:
        """Identificador; tambien acepta palabras clave como nombre de tabla."""
        tok = self._peek()
        if tok and tok.kind in ("ident", "keyword"):
            self.i += 1
            return str(tok.value)
        raise SQLSyntaxError("se esperaba un nombre")

    def _value(self) -> Any:
        tok = self._next()
        if tok.kind in ("number", "string"):
            return tok.value
        raise SQLSyntaxError(f"se esperaba un valor literal, se encontro {tok.value!r}")

    # -- entrada ------------------------------------------------------------

    def parse(self) -> Statement:
        if not self.tokens:
            raise SQLSyntaxError("consulta vacia")
        tok = self._peek()
        if tok.kind != "keyword":
            raise SQLSyntaxError(f"la consulta debe empezar con una sentencia SQL, no con {tok.value!r}")

        if tok.value == "CREATE":
            self._next()
            if self._accept("keyword", "TABLE"):
                stmt = self._create_table()
            elif self._accept("keyword", "INDEX"):
                stmt = self._create_index()
            else:
                raise SQLSyntaxError("se esperaba TABLE o INDEX despues de CREATE")
        elif tok.value == "DROP":
            self._next()
            self._expect("keyword", "TABLE")
            stmt = DropTable(self._expect_name())
        elif tok.value == "INSERT":
            stmt = self._insert()
        elif tok.value == "SELECT":
            stmt = self._select()
        elif tok.value == "DELETE":
            stmt = self._delete()
        else:
            raise SQLSyntaxError(f"sentencia no soportada: {tok.value}")

        self._accept("punct", ";")
        if self._peek() is not None:
            raise SQLSyntaxError(
                f"texto sobrante despues de la sentencia: {self._peek().value!r}. "
                "Ejecuta una sentencia a la vez."
            )
        return stmt

    # -- sentencias ---------------------------------------------------------

    def _create_table(self) -> CreateTable:
        table = self._expect_name()
        self._expect("punct", "(")
        columns: list[Column] = []
        pk_index: int | None = None

        while True:
            name = self._expect_name()
            # Se acepta cualquier identificador para poder dar un mensaje util
            # ante tipos no soportados (VARCHAR, DATE...) en vez de un generico
            # "se esperaba keyword".
            tok = self._next()
            if tok.kind not in ("keyword", "ident"):
                raise SQLSyntaxError(f"se esperaba un tipo para la columna '{name}'")
            tipo = str(tok.value).upper()
            if tipo == "INT":
                col = Column(name, ColumnType.INT)
            elif tipo == "FLOAT":
                col = Column(name, ColumnType.FLOAT)
            elif tipo == "CHAR":
                self._expect("punct", "(")
                longitud = self._expect("number").value
                if not isinstance(longitud, int) or longitud <= 0:
                    raise SQLSyntaxError(f"CHAR necesita una longitud entera positiva, no {longitud}")
                self._expect("punct", ")")
                col = Column(name, ColumnType.CHAR, longitud)
            else:
                raise SQLSyntaxError(f"tipo no soportado: {tipo}. Usa INT, FLOAT o CHAR(n).")

            if self._accept("keyword", "PRIMARY"):
                self._expect("keyword", "KEY")
                if pk_index is not None:
                    raise SQLSyntaxError("la tabla solo puede tener una PRIMARY KEY")
                pk_index = len(columns)

            columns.append(col)
            if not self._accept("punct", ","):
                break

        self._expect("punct", ")")

        engine = "SEQUENTIAL"
        if self._accept("keyword", "USING"):
            tok = self._next()
            engine = str(tok.value).upper()
            if engine not in ("HEAP", "SEQUENTIAL"):
                raise SQLSyntaxError(f"motor no soportado: {engine}. Usa HEAP o SEQUENTIAL.")

        if pk_index is None:
            pk_index = 0  # por convencion, la primera columna

        return CreateTable(table, columns, pk_index, engine)

    def _create_index(self) -> CreateIndex:
        name = self._expect_name()
        self._expect("keyword", "ON")
        table = self._expect_name()
        self._expect("punct", "(")
        column = self._expect_name()
        self._expect("punct", ")")
        self._expect("keyword", "USING")
        tok = self._next()
        kind = str(tok.value).upper()
        if kind not in ("BTREE", "HASH"):
            raise SQLSyntaxError(
                f"tipo de indice no soportado: {kind}. Usa BTREE o HASH. "
                "(R-Tree, indice invertido y busqueda vectorial son del Entregable 2.)"
            )
        return CreateIndex(name, table, column, kind)

    def _insert(self) -> Insert:
        self._expect("keyword", "INSERT")
        self._expect("keyword", "INTO")
        table = self._expect_name()
        self._expect("keyword", "VALUES")
        self._expect("punct", "(")
        values = [self._value()]
        while self._accept("punct", ","):
            values.append(self._value())
        self._expect("punct", ")")
        return Insert(table, values)

    def _select(self) -> Select:
        self._expect("keyword", "SELECT")
        columns: list[str] = []
        if self._accept("punct", "*"):
            pass
        else:
            columns.append(self._expect_name())
            while self._accept("punct", ","):
                columns.append(self._expect_name())
        self._expect("keyword", "FROM")
        table = self._expect_name()
        where = self._where()
        limit = None
        if self._accept("keyword", "LIMIT"):
            limit = self._expect("number").value
        return Select(table, columns, where, limit)

    def _delete(self) -> Delete:
        self._expect("keyword", "DELETE")
        self._expect("keyword", "FROM")
        table = self._expect_name()
        where = self._where()
        return Delete(table, where)

    # -- WHERE --------------------------------------------------------------

    def _where(self) -> Condition | None:
        if not self._accept("keyword", "WHERE"):
            return None

        column = self._expect_name()

        if self._accept("keyword", "BETWEEN"):
            lo = self._value()
            self._expect("keyword", "AND")
            hi = self._value()
            return Condition(column, lo, hi)

        op = self._expect("op").value
        value = self._value()
        cond = self._condition_from(column, op, value)

        if self._accept("keyword", "AND"):
            col2 = self._expect_name()
            if col2.lower() != column.lower():
                raise SQLSyntaxError(
                    f"el WHERE solo admite predicados sobre una columna; "
                    f"se encontraron {column!r} y {col2!r}"
                )
            op2 = self._expect("op").value
            val2 = self._value()
            cond = _merge(cond, self._condition_from(col2, op2, val2))

        return cond

    def _condition_from(self, column: str, op: str, value: Any) -> Condition:
        if op == "=":
            return Condition(column, value, value)
        if op == ">":
            return Condition(column, lo=value, lo_inclusive=False)
        if op == ">=":
            return Condition(column, lo=value, lo_inclusive=True)
        if op == "<":
            return Condition(column, hi=value, hi_inclusive=False)
        if op == "<=":
            return Condition(column, hi=value, hi_inclusive=True)
        raise SQLSyntaxError(f"operador no soportado en WHERE: {op}")


def _merge(a: Condition, b: Condition) -> Condition:
    """Intersecta dos predicados sobre la misma columna."""
    out = Condition(a.column)
    for c in (a, b):
        if c.lo is not None and (out.lo is None or c.lo > out.lo):
            out.lo, out.lo_inclusive = c.lo, c.lo_inclusive
        if c.hi is not None and (out.hi is None or c.hi < out.hi):
            out.hi, out.hi_inclusive = c.hi, c.hi_inclusive
    return out


def parse(sql: str) -> Statement:
    return Parser(sql).parse()
