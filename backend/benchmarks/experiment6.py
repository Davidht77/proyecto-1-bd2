"""Experimento 6 (enunciado 6.2): indice invertido en disco vs Full Text Scan vs GIN.

6a. Construccion: tiempo y tamano en disco del indice invertido (vocabulario B+
    + postings + normas) escalando el corpus de 1 000 a 20 000 documentos, y lo
    mismo para el indice GIN de PostgreSQL sobre to_tsvector.
6b. Consultas sobre el corpus completo: booleanas AND, OR y AND-NOT y ranking
    Top-10 (TF-IDF + coseno) de 1, 3 y 5 terminos. Se comparan latencia (ms) y
    bloques leidos entre:
      indice     MATCH con el indice invertido   (InvertedIndexScan, bloques de 4 KB)
      full_scan  MATCH sin indice                (TextSeqScan, bloques de 4 KB)
      gin        PostgreSQL 16 con GIN           (EXPLAIN ANALYZE BUFFERS, bloques de 8 KB)

Nuestro motor corre por SQL real a traves del Executor, asi que las metricas son
las mismas que muestra el cliente web. PostgreSQL es solo la referencia externa
que pide el enunciado: no se usa para nada del motor.

Las consultas salen del propio corpus: las palabras de un AND se toman de un
mismo documento (asi el AND no queda vacio) y la palabra negada de un AND-NOT es
una palabra frecuente que ese documento no tiene.

Uso:
    docker run -d --name bd2-pg -e POSTGRES_PASSWORD=bd2 -e POSTGRES_DB=bd2 -p 55432:5432 postgres:16-alpine
    python -m backend.benchmarks.experiment6 --csv data/datasets/wiki_20k.csv
    python -m backend.benchmarks.experiment6 --no-pg          # sin PostgreSQL
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass

from backend.engine.catalog import Catalog
from backend.engine.executor import Executor
from backend.text.inverted import files_of
from backend.text.preprocess import STOPWORDS, _WORD

SIZES = [1000, 2500, 5000, 10000, 20000]
TERMS = [1, 3, 5]
K = 10
PG_DSN = "host=localhost port=55432 dbname=bd2 user=postgres password=bd2"


@dataclass
class BuildRow:
    n_docs: int
    method: str            # "indice" | "gin"
    build_ms: float
    size_bytes: int
    pages: int             # bloques del indice (4 KB nuestro, 8 KB PostgreSQL)
    writes: int            # escrituras de bloque durante la construccion (solo nuestro)


@dataclass
class QueryRow:
    n_docs: int
    n_terms: int
    op: str                # AND | OR | AND-NOT | TOPK
    method: str            # indice | full_scan | gin
    n_queries: int
    avg_ms: float
    std_ms: float
    avg_reads: float       # bloques leidos (PostgreSQL: shared hit + read)
    avg_index_reads: float
    avg_table_reads: float
    avg_matches: float


# ----------------------------------------------------------------- datos


def leer_corpus(path: str, n: int) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    por_lang: dict[str, list] = {}
    for r in filas:
        por_lang.setdefault(r["lang"], []).append(r)
    k = len(por_lang) or 1
    out = [r for g in por_lang.values() for r in g[: n // k]]
    return out[:n]


def _palabras(texto: str) -> list[str]:
    return [w for w in dict.fromkeys(_WORD.findall(texto.lower()))
            if len(w) >= 4 and w not in STOPWORDS and w.isascii()]


def generar_consultas(docs: list[dict], por_celda: int, seed: int = 0) -> list[tuple[int, str, str]]:
    """[(n_terminos, operacion, consulta MATCH)]."""
    rng = random.Random(seed)
    df = Counter(w for d in docs for w in set(_palabras(d["text"])))
    utiles = {w for w, c in df.items() if 3 <= c <= len(docs) // 5}
    frecuentes = [w for w, c in df.most_common(400) if w in utiles]
    out = []

    def doc_con(k):
        while True:
            ws = [w for w in _palabras(rng.choice(docs)["text"]) if w in utiles]
            if len(ws) >= k:
                return rng.sample(ws, k)

    for n in TERMS:
        for _ in range(por_celda):
            out.append((n, "AND", " AND ".join(doc_con(n))))
            out.append((n, "TOPK", " ".join(doc_con(n))))
            if n > 1:
                out.append((n, "OR", " OR ".join(rng.sample(sorted(utiles), n))))
                pos = doc_con(n - 1)
                neg = next(w for w in rng.sample(frecuentes, len(frecuentes)) if w not in pos)
                out.append((n, "AND-NOT", " AND ".join(pos) + f" AND NOT {neg}"))
    return out


# ------------------------------------------------------------- nuestro motor


def cargar_motor(docs: list[dict]) -> tuple[Catalog, Executor, str]:
    tmp = tempfile.mkdtemp(prefix="exp6_")
    cat = Catalog(tmp, page_size=4096, cache_size=0)
    ex = Executor(cat)
    ex.execute("CREATE TABLE docs (id INT PRIMARY KEY, lang CHAR(2), texto CHAR(1000)) USING HEAP")
    sf = cat.open_table("docs")
    sf.bulk_load((i, d["lang"], d["text"]) for i, d in enumerate(docs))
    sf.flush()
    return cat, ex, tmp


def construir_indice(cat: Catalog, ex: Executor) -> BuildRow:
    r = ex.execute("CREATE INDEX itx ON docs (texto) USING INVERTED WITH (scoring = 'TFIDF_COSINE')")
    idx = cat.open_index("docs", "itx")
    base = cat._index_base("docs", "itx")
    size = sum(p.stat().st_size for p in files_of(base) if p.exists())
    return BuildRow(cat.open_table("docs").n_records, "indice", r.exec_ms, size, size // 4096,
                    idx.counter.disk_writes)


def sql_motor(op: str, consulta: str) -> str:
    if op == "TOPK":
        return (f"SELECT id, COSINE_SCORE() AS s FROM docs WHERE texto MATCH '{consulta}' "
                f"ORDER BY s DESC LIMIT {K}")
    return f"SELECT id FROM docs WHERE texto MATCH '{consulta}'"


def medir_motor(ex: Executor, op: str, consulta: str) -> dict:
    r = ex.execute(sql_motor(op, consulta))
    io_ = r.plan.detail.get("io", {})
    return {"ms": r.exec_ms, "reads": r.disk_reads, "index": io_.get("index_reads", 0),
            "table": io_.get("table_reads", 0), "matches": r.plan.detail["text"]["matches"],
            "ids": [row[0] for row in r.rows], "access": r.plan.access}


# --------------------------------------------------------------- PostgreSQL


_TSV = ("to_tsvector(CASE lang WHEN 'es' THEN 'spanish'::regconfig "
        "ELSE 'english'::regconfig END, texto)")


def pg_conectar(dsn: str):
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def pg_cargar(conn, docs: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS docs")
        cur.execute(f"CREATE TABLE docs (id int PRIMARY KEY, lang char(2), texto text, "
                    f"tsv tsvector GENERATED ALWAYS AS ({_TSV}) STORED)")
        buf = io.StringIO()
        w = csv.writer(buf)
        for i, d in enumerate(docs):
            w.writerow([i, d["lang"], d["text"]])
        buf.seek(0)
        cur.copy_expert("COPY docs (id, lang, texto) FROM STDIN WITH (FORMAT csv)", buf)
        cur.execute("VACUUM ANALYZE docs")


def pg_construir(conn, n: int) -> BuildRow:
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        cur.execute("CREATE INDEX docs_gin ON docs USING GIN (tsv)")
        ms = (time.perf_counter() - t0) * 1000
        cur.execute("ANALYZE docs")
        cur.execute("SELECT pg_relation_size('docs_gin')")
        size = cur.fetchone()[0]
    return BuildRow(n, "gin", ms, size, size // 8192, 0)


def _pg_palabra(w: str) -> str:
    return f"(to_tsquery('english', '{w}') || to_tsquery('spanish', '{w}'))"


def pg_tsquery(consulta: str) -> str:
    """Traduce el MATCH al tsquery equivalente; cada palabra pasa por los dos
    stemmers, igual que en nuestro preprocesamiento de consultas."""
    if " NOT " in consulta:
        pos, neg = consulta.split(" AND NOT ")
        partes = [_pg_palabra(w) for w in pos.split(" AND ")]
        return "(" + " && ".join(partes) + f") && (!!{_pg_palabra(neg)})"
    if " OR " in consulta:
        return "(" + " || ".join(_pg_palabra(w) for w in consulta.split(" OR ")) + ")"
    if " AND " in consulta:
        return "(" + " && ".join(_pg_palabra(w) for w in consulta.split(" AND ")) + ")"
    return "(" + " || ".join(_pg_palabra(w) for w in consulta.split()) + ")"


def medir_pg(conn, op: str, consulta: str) -> dict:
    import json

    q = "(" + pg_tsquery(consulta) + ")"  # @@ se evalua antes que &&: hay que agrupar
    if op == "TOPK":
        sql = f"SELECT id FROM docs WHERE tsv @@ {q} ORDER BY ts_rank(tsv, {q}) DESC LIMIT {K}"
        cnt = f"SELECT count(*) FROM docs WHERE tsv @@ {q}"
    else:
        sql = f"SELECT id FROM docs WHERE tsv @@ {q}"
        cnt = None
    with conn.cursor() as cur:
        cur.execute("SET client_min_messages = warning")
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
        plan = cur.fetchone()[0]
        plan = plan[0] if isinstance(plan, list) else json.loads(plan)[0]
        raiz = plan["Plan"]
        bloques = raiz.get("Shared Hit Blocks", 0) + raiz.get("Shared Read Blocks", 0)
        filas = raiz.get("Actual Rows", 0)
        if cnt:
            cur.execute(cnt)
            filas = cur.fetchone()[0]
    return {"ms": plan["Execution Time"], "reads": bloques, "index": 0, "table": 0, "matches": filas}


# ------------------------------------------------------------------ corrida


def _fila(n, nt, op, metodo, medidas) -> QueryRow:
    ms = [m["ms"] for m in medidas]
    return QueryRow(n, nt, op, metodo, len(medidas), statistics.mean(ms),
                    statistics.pstdev(ms) if len(ms) > 1 else 0.0,
                    statistics.mean(m["reads"] for m in medidas),
                    statistics.mean(m["index"] for m in medidas),
                    statistics.mean(m["table"] for m in medidas),
                    statistics.mean(m["matches"] for m in medidas))


def run(csv_path: str, sizes=SIZES, por_celda: int = 20, scan_por_celda: int = 2,
        pg_dsn: str | None = PG_DSN) -> tuple[list[BuildRow], list[QueryRow]]:
    todos = leer_corpus(csv_path, max(sizes))
    sizes = [n for n in sizes if n <= len(todos)]
    conn = pg_conectar(pg_dsn) if pg_dsn else None
    builds: list[BuildRow] = []

    # ---- 6a: construccion
    for n in sizes:
        docs = todos[:n]
        cat, ex, tmp = cargar_motor(docs)
        try:
            b = construir_indice(cat, ex)
            builds.append(b)
            print(f"  N={n:>6}: indice {b.build_ms / 1000:.1f} s, {b.size_bytes / 1e6:.1f} MB", flush=True)
        finally:
            cat.close_all()
            shutil.rmtree(tmp, ignore_errors=True)
        if conn is not None:
            pg_cargar(conn, docs)
            g = pg_construir(conn, n)
            builds.append(g)
            print(f"           GIN    {g.build_ms / 1000:.2f} s, {g.size_bytes / 1e6:.1f} MB", flush=True)

    # ---- 6b: consultas sobre el corpus completo
    n = sizes[-1]
    docs = todos[:n]
    consultas = generar_consultas(docs, por_celda)
    cat, ex, tmp = cargar_motor(docs)
    filas: list[QueryRow] = []
    try:
        construir_indice(cat, ex)
        if conn is not None:
            pg_cargar(conn, docs)
            pg_construir(conn, n)
        celdas = sorted({(nt, op) for nt, op, _ in consultas}, key=lambda c: (c[0], c[1]))
        for nt, op in celdas:
            qs = [c for t, o, c in consultas if (t, o) == (nt, op)]
            con = [medir_motor(ex, op, c) for c in qs]
            filas.append(_fila(n, nt, op, "indice", con))
            if conn is not None:
                filas.append(_fila(n, nt, op, "gin", [medir_pg(conn, op, c) for c in qs]))
            # Sin indice: mismo SQL, el planificador cae a TextSeqScan.
            idx_files = cat._index_base("docs", "itx")
            cat.close_index("docs", "itx")
            ocultos = []
            for p in files_of(idx_files):
                if p.exists():
                    p.rename(p.with_name(p.name + ".off"))
                    ocultos.append(p)
            try:
                sin = [medir_motor(ex, op, c) for c in qs[:scan_por_celda]]
            finally:
                for p in ocultos:
                    p.with_name(p.name + ".off").rename(p)
            for a, b in zip(con, sin):
                assert a["matches"] == b["matches"], (op, a["matches"], b["matches"])
            filas.append(_fila(n, nt, op, "full_scan", sin))
            ult = filas[-3:] if conn is not None else filas[-2:]
            print(f"  {nt} term {op:<7} " + "  ".join(
                f"{f.method}: {f.avg_ms:.1f} ms / {f.avg_reads:.0f} bloques" for f in ult), flush=True)
    finally:
        cat.close_all()
        shutil.rmtree(tmp, ignore_errors=True)
        if conn is not None:
            conn.close()
    return builds, filas


def main() -> None:
    from backend.benchmarks.run_all import _volcar

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/datasets/wiki_20k.csv")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--queries", type=int, default=20, help="consultas por celda (indice y GIN)")
    ap.add_argument("--scan-queries", type=int, default=2, help="consultas por celda para el Full Text Scan")
    ap.add_argument("--pg-dsn", default=PG_DSN)
    ap.add_argument("--no-pg", action="store_true")
    args = ap.parse_args()
    builds, filas = run(args.csv, args.sizes, args.queries, args.scan_queries,
                        None if args.no_pg else args.pg_dsn)
    _volcar("exp6_construccion", builds)
    _volcar("exp6_consultas", filas)


if __name__ == "__main__":
    main()
