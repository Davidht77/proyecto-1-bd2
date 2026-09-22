"""API REST del motor (enunciado 3.5).

Endpoints implementados:
    POST   /api/query                      ejecuta SQL y devuelve filas + metricas
    GET    /api/tables                     lista tablas, esquemas e indices
    GET    /api/tables/{t}                 detalle de una tabla
    POST   /api/tables/{t}/reorganize      reorganiza el Sequential File
    GET    /api/tables/{t}/pages           inspeccion de paginas fisicas
    POST   /api/tables/{t}/seed            carga masiva de datos sinteticos
    GET    /api/health                     estado del motor

Endpoints declarados pero pendientes (devuelven 501 con el detalle de que falta):
    POST   /api/benchmarks/run             suite de los 4 experimentos
    GET    /api/tables/{t}/index/{nombre}  inspeccion de un indice B+/Hash
"""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.engine.catalog import Catalog
from backend.engine.errors import (
    EngineError,
    NotImplementedFeature,
    SQLSyntaxError,
    TableAlreadyExists,
    TableNotFound,
)
from backend.engine.executor import Executor
from backend.storage.page import NULL_PAGE
from backend.storage.schema import ColumnType

DATA_DIR = os.environ.get("BD2_DATA_DIR", "data")
PAGE_SIZE = int(os.environ.get("BD2_PAGE_SIZE", "4096"))
CACHE_SIZE = int(os.environ.get("BD2_CACHE_SIZE", "0"))

catalog = Catalog(DATA_DIR, page_size=PAGE_SIZE, cache_size=CACHE_SIZE)
executor = Executor(catalog)

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    catalog.close_all()  # vuelca las paginas sucias del buffer pool a disco


app = FastAPI(
    lifespan=lifespan,
    title="Mini-DBMS CS2042 — Proyecto 1",
    description="Motor relacional sobre memoria secundaria: Sequential File y capa de paginacion.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- modelos


class QueryRequest(BaseModel):
    sql: str = Field(..., description="Sentencia SQL a ejecutar (una a la vez)")
    max_rows: int = Field(500, ge=1, le=5000, description="Tope de filas devueltas al cliente")


class SeedRequest(BaseModel):
    n: int = Field(1000, ge=1, le=1_000_000, description="Cantidad de registros a generar")
    start_key: int = Field(0, description="Primera clave primaria")
    mode: str = Field(
        "bulk",
        description="'bulk' usa bulk_load (rapido, requiere tabla vacia); "
        "'insert' usa la ruta insert() normal, que es la que mide el Experimento 1",
    )
    shuffle: bool = Field(True, description="Desordenar las claves antes de cargarlas")


# ------------------------------------------------------------- manejo de errores


def _http(exc: EngineError) -> HTTPException:
    if isinstance(exc, NotImplementedFeature):
        return HTTPException(501, {"error": "no_implementado", "detail": str(exc)})
    if isinstance(exc, SQLSyntaxError):
        return HTTPException(400, {"error": "sintaxis", "detail": str(exc)})
    if isinstance(exc, TableNotFound):
        return HTTPException(404, {"error": "tabla_no_encontrada", "detail": str(exc)})
    if isinstance(exc, TableAlreadyExists):
        return HTTPException(409, {"error": "tabla_duplicada", "detail": str(exc)})
    return HTTPException(400, {"error": "ejecucion", "detail": str(exc)})


# ---------------------------------------------------------------- endpoints


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_dir": str(Path(DATA_DIR).resolve()),
        "page_size": PAGE_SIZE,
        "cache_size": CACHE_SIZE,
        "tables": len(catalog.table_names()),
        "implemented": {
            "storage": True,
            "sequential_file": True,
            "heap_file": True,
            "btree": True,
            "hash": True,
        },
    }


@app.post("/api/query")
def run_query(req: QueryRequest) -> dict:
    try:
        return executor.execute(req.sql, max_rows=req.max_rows).as_dict()
    except EngineError as exc:
        raise _http(exc) from exc
    except Exception as exc:  # noqa: BLE001 - el cliente necesita ver el fallo
        raise HTTPException(500, {"error": "interno", "detail": str(exc)}) from exc


@app.get("/api/tables")
def list_tables() -> dict:
    try:
        return {"tables": [t.as_dict() for t in catalog.list_tables()]}
    except EngineError as exc:
        raise _http(exc) from exc


@app.get("/api/tables/{table}")
def describe_table(table: str) -> dict:
    try:
        return catalog.describe(table).as_dict()
    except EngineError as exc:
        raise _http(exc) from exc


@app.post("/api/tables/{table}/reorganize")
def reorganize(table: str) -> dict:
    try:
        if catalog.engine_of(table) != "SEQUENTIAL":
            raise HTTPException(
                409,
                {
                    "error": "no_aplica",
                    "detail": f"«{table}» es un Heap File. La reorganización es una "
                    "operación del Sequential File: fusiona su área de overflow y "
                    "reescribe el archivo principal ordenado. Un Heap File no "
                    "mantiene orden, así que no hay nada que reorganizar.",
                },
            )
        sf = catalog.open_table(table)
        antes = {"pages": sf.page_count, "overflow_records": sf.n_overflow}
        stats = sf.reorganize()
        sf.flush()
        return {
            "table": table,
            "before": antes,
            "after": {"pages": sf.page_count, "overflow_records": sf.n_overflow},
            "stats": stats.as_dict(),
        }
    except HTTPException:
        raise
    except EngineError as exc:
        raise _http(exc) from exc


@app.get("/api/tables/{table}/pages")
def inspect_pages(
    table: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> dict:
    """Inspeccion de las paginas fisicas del area principal.

    Existe para la demostracion en video que pide el enunciado: permite ver la
    estructura real en disco (cabecera, ocupacion, low_key, cadena de overflow)
    sin abrir un visor hexadecimal.
    """
    try:
        engine = catalog.engine_of(table)
        f = catalog.open_table(table)
        sf = f
        # El Heap File numera sus paginas igual (la 0 es la cabecera) pero no
        # tiene low_key ni cadenas de overflow.
        leer = f._read_main if engine == "SEQUENTIAL" else f._read_page
        paginas = []
        for p in range(offset, min(offset + limit, f.page_count)):
            page = leer(p)
            cadena = []
            pid = page.aux_page_id if engine == "SEQUENTIAL" else NULL_PAGE
            while pid != NULL_PAGE:
                ov = sf.ovf.read_page(pid)
                cadena.append({
                    "page_id": ov.page_id,
                    "record_count": ov.record_count,
                    "capacity": ov.capacity,
                    "next_page_id": None if ov.next_page_id == NULL_PAGE else ov.next_page_id,
                })
                pid = ov.next_page_id
            low = page.get_low_key(f.schema) if engine == "SEQUENTIAL" else None
            paginas.append({
                "logical_index": p,
                "page_id": page.page_id,
                "page_type": page.page_type.name,
                "record_count": page.record_count,
                "capacity": page.capacity,
                "fill_pct": round(100 * page.record_count / page.capacity, 1),
                "low_key": None if low is None or low == float("-inf") else low,
                "first_key": page.first_key(f.schema),
                "last_key": page.last_key(f.schema),
                "overflow_chain": cadena,
            })
        return {
            "table": table,
            "engine": engine,
            "page_size": f.page_size,
            "total_pages": f.page_count,
            "capacity_per_page": f._capacity,
            "overflow_cap": getattr(f, "overflow_cap", 0),
            "overflow_records": getattr(f, "n_overflow", 0),
            "offset": offset,
            "limit": limit,
            "pages": paginas,
        }
    except EngineError as exc:
        raise _http(exc) from exc


@app.post("/api/tables/{table}/seed")
def seed(table: str, req: SeedRequest) -> dict:
    """Carga datos sinteticos, para probar con los volumenes que exige el enunciado."""
    try:
        sf = catalog.open_table(table)
        schema = sf.schema
        claves = list(range(req.start_key, req.start_key + req.n))
        if req.shuffle:
            random.Random(0).shuffle(claves)

        def fila(k: int) -> tuple:
            out = []
            for i, col in enumerate(schema.columns):
                if i == schema.pk_index:
                    out.append(k if col.type is ColumnType.INT else float(k))
                elif col.type is ColumnType.INT:
                    out.append(k % 100)
                elif col.type is ColumnType.FLOAT:
                    out.append(round(1000 + (k % 9000) * 1.5, 2))
                else:
                    out.append(f"{col.name[:3]}{k}"[: col.length])
            return tuple(out)

        sf.counter.reset()
        import time

        t0 = time.perf_counter()
        if req.mode == "bulk":
            if sf.n_records:
                raise HTTPException(
                    409,
                    {
                        "error": "tabla_no_vacia",
                        "detail": "bulk_load requiere una tabla vacia. Usa mode='insert' "
                        "o elimina la tabla primero.",
                    },
                )
            sf.bulk_load(fila(k) for k in claves)
        elif req.mode == "insert":
            for k in claves:
                sf.insert(fila(k))
        else:
            raise HTTPException(400, {"error": "modo_invalido", "detail": "usa 'bulk' o 'insert'"})
        sf.flush()
        ms = (time.perf_counter() - t0) * 1000.0

        return {
            "table": table,
            "inserted": req.n,
            "mode": req.mode,
            "metrics": {
                "disk_reads": sf.counter.disk_reads,
                "disk_writes": sf.counter.disk_writes,
                "elapsed_ms": round(ms, 2),
            },
            "state": {
                "n_records": sf.n_records,
                "n_overflow": getattr(sf, "n_overflow", 0),
                "pages": sf.page_count,
            },
        }
    except HTTPException:
        raise
    except EngineError as exc:
        raise _http(exc) from exc


# ------------------------------------------------- endpoints declarados, pendientes


@app.post("/api/benchmarks/run")
def run_benchmarks() -> dict:
    """PENDIENTE: suite de los 4 experimentos del enunciado (seccion 4).

    Requiere Heap File, Arbol B+ y Hash Dinamico para poder comparar. Con solo
    el Sequential File la comparativa no tiene sentido.
    """
    raise HTTPException(
        501,
        {
            "error": "no_implementado",
            "detail": "La suite de benchmarks necesita el árbol B+ y el hashing "
            "dinámico para completar la comparativa de los 4 experimentos. "
            "Mientras tanto, usa POST /api/tables/{t}/seed y POST /api/query, que "
            "ya reportan I/O y latencia por operación para Heap y Sequential.",
            "missing": ["btree", "hash"],
        },
    )


@app.get("/api/tables/{table}/index/{index_name}")
def inspect_index(table: str, index_name: str) -> dict:
    """PENDIENTE: inspeccion de un indice B+ o Hash (altura, fan-out, buckets)."""
    raise HTTPException(
        501,
        {
            "error": "no_implementado",
            "detail": "Los indices BTREE y HASH todavia no estan implementados. "
            "El unico camino de acceso indexado disponible es la busqueda binaria "
            "del Sequential File sobre su clave primaria.",
            "missing": ["btree", "hash"],
        },
    )


# ------------------------------------------------------------------ frontend

_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")
