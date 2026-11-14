"""Parser SQL por descenso recursivo.

Gramatica soportada (Entregable 1 + extension multimodal del Entregable 2, 5.4):

    CREATE TABLE <t> (<col> <tipo> [PRIMARY KEY], ...) [USING HEAP|SEQUENTIAL]
        <tipo> := INT | FLOAT | CHAR(n) | VECTOR(d)
    DROP TABLE <t>
    INSERT INTO <t> VALUES (<v>, ...)          -- un vector se escribe '[x1, x2, ...]'
    SELECT <item>, ... FROM <t> [WHERE <pred> {AND <pred>}]
           [ORDER BY <clave> [ASC|DESC]] [LIMIT n]
        <item>  := * | <col> | <col> <-> '<vec>' [AS alias] | <col> <=> '<vec>' [AS alias]
                 | COSINE_SCORE() [AS alias] | BM25_SCORE() [AS alias]
        <clave> := <col> | <alias> | <col> <-> '<vec>' | <col> <=> '<vec>'
    DELETE FROM <t> WHERE <pred> {AND <pred>}
    CREATE INDEX <nombre> ON <t> (<col>) USING <tipo> [WITH (<k> = <v>, ...)]
        <tipo> := BTREE | HASH | IVF | HNSW | RTREE | INVERTED
    SET <nombre>[.<nombre>] = <v>              -- p. ej. SET ivf.probes = 5

    <pred> := <col> = <v> | <col> <op> <v> | <col> BETWEEN <v> AND <v>
            | <col> MATCH '<consulta>'                     (modulo textual)
            | <col> WITHIN_RADIUS ((<x>, <y>), <r>)        (modulo espacial)

Varios predicados sobre la misma columna se intersectan en un solo rango; los
predicados sobre columnas distintas quedan en `filters` para que el
planificador elija cual resuelve con un indice y cuales evalua como filtro
residual.

No se usa ninguna libreria de parsing: el enunciado exige construir el parser
desde cero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.engine.errors import SQLSyntaxError
from backend.storage.schema import Column, ColumnType, parse_vector

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
class DistanceExpr:
    """`<col> <-> '[...]'` (L2) o `<col> <=> '[...]'` (distancia coseno)."""

    column: str
    operator: str  # "<->" | "<=>"
    vector: tuple[float, ...]
    alias: str | None = None

    @property
    def metric(self) -> str:
        return "L2" if self.operator == "<->" else "COSINE"

    @property
    def label(self) -> str:
        return self.alias or f"{self.column} {self.operator} q"


@dataclass
class ScoreExpr:
    """COSINE_SCORE() / BM25_SCORE(): relevancia textual del predicado MATCH."""

    function: str
    alias: str | None = None

    @property
    def label(self) -> str:
        return self.alias or f"{self.function.lower()}()"


@dataclass
class TextMatch:
    column: str
    query: str


@dataclass
class SpatialRadius:
    column: str
    x: float
    y: float
    radius: float


@dataclass
class OrderBy:
    key: "str | DistanceExpr"
    descending: bool = False


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
    columns: list = field(default_factory=list)  # str | DistanceExpr | ScoreExpr; vacio = *
    where: Condition | None = None  # primer predicado relacional (compatibilidad P1)
    limit: int | None = None
    filters: list[Condition] = field(default_factory=list)  # todos los predicados relacionales
    order_by: OrderBy | None = None
    special: list = field(default_factory=list)  # TextMatch | SpatialRadius

    @property
    def distance(self) -> DistanceExpr | None:
        """La expresion de distancia de la consulta, venga del SELECT o del ORDER BY."""
        for c in self.columns:
            if isinstance(c, DistanceExpr):
                return c
        if self.order_by and isinstance(self.order_by.key, DistanceExpr):
            return self.order_by.key
        return None


@dataclass
class Delete:
    table: str
    where: Condition | None = None
    filters: list[Condition] = field(default_factory=list)


@dataclass
class CreateIndex:
    name: str
    table: str
    column: str
    kind: str  # "BTREE" | "HASH" | "IVF" | "HNSW" | "RTREE" | "INVERTED"
    options: dict = field(default_factory=dict)


@dataclass
class SetOption:
    name: str  # en minusculas, p. ej. "ivf.probes"
    value: Any


Statement = CreateTable | DropTable | Insert | Select | Delete | CreateIndex | SetOption

INDEX_KINDS = ("BTREE", "HASH", "IVF", "HNSW", "RTREE", "INVERTED")


# ---------------------------------------------------------------- tokenizer

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<comment>--[^\n]*)
    | (?P<string>'(?:[^']|'')*')
    | (?P<number>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    | (?P<op><->|<=>|<=|>=|<>|!=|=|<|>)
    | (?P<punct>[(),;*.])
    | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
    """,
    re.VERBOSE,
)

_KEYWORDS = {
    "CREATE", "TABLE", "DROP", "INSERT", "INTO", "VALUES", "SELECT", "FROM",
    "WHERE", "DELETE", "INDEX", "ON", "USING", "AND", "BETWEEN", "PRIMARY",
    "KEY", "INT", "FLOAT", "CHAR", "HEAP", "SEQUENTIAL", "BTREE", "HASH",
    "LIMIT", "ORDER", "BY", "ASC", "DESC", "AS", "SET", "WITH", "MATCH",
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
            value = float(text) if any(ch in text for ch in ".eE") else int(text)
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
        elif tok.value == "SET":
            stmt = self._set()
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
            elif tipo == "VECTOR":
                self._expect("punct", "(")
                dim = self._expect("number").value
                if not isinstance(dim, int) or not 0 < dim <= 4096:
                    raise SQLSyntaxError(f"VECTOR necesita una dimension entera entre 1 y 4096, no {dim}")
                self._expect("punct", ")")
                col = Column(name, ColumnType.VECTOR, dim)
            else:
                raise SQLSyntaxError(
                    f"tipo no soportado: {tipo}. Usa INT, FLOAT, CHAR(n) o VECTOR(d)."
                )

            if self._accept("keyword", "PRIMARY"):
                self._expect("keyword", "KEY")
                if pk_index is not None:
                    raise SQLSyntaxError("la tabla solo puede tener una PRIMARY KEY")
                if col.type is ColumnType.VECTOR:
                    raise SQLSyntaxError("un VECTOR no puede ser PRIMARY KEY")
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
            if columns[0].type is ColumnType.VECTOR:
                raise SQLSyntaxError(
                    "la primera columna es un VECTOR: declara explicitamente la PRIMARY KEY"
                )

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
        if kind not in INDEX_KINDS:
            raise SQLSyntaxError(
                f"tipo de indice no soportado: {kind}. Usa {', '.join(INDEX_KINDS)}."
            )
        options: dict = {}
        if self._accept("keyword", "WITH"):
            self._expect("punct", "(")
            while True:
                clave = self._expect_name().lower()
                self._expect("op", "=")
                tok = self._next()
                if tok.kind not in ("number", "string", "ident", "keyword"):
                    raise SQLSyntaxError(f"valor invalido para la opcion «{clave}»: {tok.value!r}")
                options[clave] = tok.value
                if not self._accept("punct", ","):
                    break
            self._expect("punct", ")")
        return CreateIndex(name, table, column, kind, options)

    def _set(self) -> SetOption:
        self._expect("keyword", "SET")
        partes = [self._expect_name()]
        while self._accept("punct", "."):
            partes.append(self._expect_name())
        self._expect("op", "=")
        tok = self._next()
        if tok.kind not in ("number", "string", "ident", "keyword"):
            raise SQLSyntaxError(f"valor invalido en SET: {tok.value!r}")
        return SetOption(".".join(partes).lower(), tok.value)

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
        columns: list = []
        if not self._accept("punct", "*"):
            columns.append(self._select_item())
            while self._accept("punct", ","):
                columns.append(self._select_item())
        self._expect("keyword", "FROM")
        table = self._expect_name()
        filters, special = self._where()
        order_by = None
        if self._accept("keyword", "ORDER"):
            self._expect("keyword", "BY")
            name = self._expect_name()
            key = self._distance_tail(name) or name
            descending = bool(self._accept("keyword", "DESC"))
            if not descending:
                self._accept("keyword", "ASC")
            order_by = OrderBy(key, descending)
        limit = None
        if self._accept("keyword", "LIMIT"):
            limit = self._expect("number").value
            if not isinstance(limit, int) or limit < 0:
                raise SQLSyntaxError(f"LIMIT necesita un entero no negativo, no {limit}")
        if sum(isinstance(c, DistanceExpr) for c in columns) > 1:
            raise SQLSyntaxError("solo se admite una expresion de distancia vectorial por consulta")
        return Select(table, columns, filters[0] if filters else None, limit,
                      filters, order_by, special)

    def _select_item(self):
        name = self._expect_name()
        if self._accept("punct", "("):
            self._expect("punct", ")")
            funcion = name.upper()
            if funcion not in ("COSINE_SCORE", "BM25_SCORE"):
                raise SQLSyntaxError(
                    f"funcion no soportada: {name}(). Usa COSINE_SCORE() o BM25_SCORE()."
                )
            return ScoreExpr(funcion, self._alias())
        dist = self._distance_tail(name)
        if dist is not None:
            dist.alias = self._alias()
            return dist
        return name

    def _distance_tail(self, column: str) -> DistanceExpr | None:
        """Si sigue `<->` o `<=>`, consume el literal vectorial."""
        tok = self._peek()
        if not (tok and tok.kind == "op" and tok.value in ("<->", "<=>")):
            return None
        self._next()
        lit = self._next()
        if lit.kind != "string":
            raise SQLSyntaxError(
                f"despues de {tok.value} se esperaba un vector entre comillas, p. ej. '[0.1, 0.2]'"
            )
        try:
            vector = parse_vector(lit.value)
        except ValueError as exc:
            raise SQLSyntaxError(str(exc)) from None
        if not vector:
            raise SQLSyntaxError("el vector de consulta esta vacio")
        return DistanceExpr(column, tok.value, vector)

    def _alias(self) -> str | None:
        if self._accept("keyword", "AS"):
            return self._expect_name()
        return None

    def _delete(self) -> Delete:
        self._expect("keyword", "DELETE")
        self._expect("keyword", "FROM")
        table = self._expect_name()
        filters, special = self._where()
        if special:
            raise SQLSyntaxError("DELETE no admite predicados MATCH ni WITHIN_RADIUS")
        return Delete(table, filters[0] if filters else None, filters)

    # -- WHERE --------------------------------------------------------------

    def _where(self) -> tuple[list[Condition], list]:
        """Conjuncion de predicados. Devuelve (relacionales, especiales)."""
        if not self._accept("keyword", "WHERE"):
            return [], []
        por_columna: dict[str, Condition] = {}
        special: list = []
        while True:
            pred = self._predicate()
            if isinstance(pred, Condition):
                clave = pred.column.lower()
                por_columna[clave] = (
                    _merge(por_columna[clave], pred) if clave in por_columna else pred
                )
            else:
                special.append(pred)
            if not self._accept("keyword", "AND"):
                break
        return list(por_columna.values()), special

    def _predicate(self):
        column = self._expect_name()

        if self._accept("keyword", "BETWEEN"):
            lo = self._value()
            self._expect("keyword", "AND")
            hi = self._value()
            return Condition(column, lo, hi)

        if self._accept("keyword", "MATCH"):
            tok = self._next()
            if tok.kind != "string":
                raise SQLSyntaxError("MATCH espera la consulta textual entre comillas")
            return TextMatch(column, tok.value)

        tok = self._peek()
        if tok and tok.kind == "ident" and str(tok.value).upper() == "WITHIN_RADIUS":
            self._next()
            self._expect("punct", "(")
            self._expect("punct", "(")
            x = self._number()
            self._expect("punct", ",")
            y = self._number()
            self._expect("punct", ")")
            self._expect("punct", ",")
            r = self._number()
            self._expect("punct", ")")
            return SpatialRadius(column, x, y, r)

        op = self._expect("op").value
        value = self._value()
        return self._condition_from(column, op, value)

    def _number(self) -> float:
        tok = self._next()
        if tok.kind != "number":
            raise SQLSyntaxError(f"se esperaba un numero, se encontro {tok.value!r}")
        return float(tok.value)

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
