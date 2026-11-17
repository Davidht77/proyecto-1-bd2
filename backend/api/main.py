"""API REST del motor (enunciado 3.5).

Endpoints implementados:
    POST   /api/query                      ejecuta SQL y devuelve filas + metricas
    GET    /api/tables                     lista tablas, esquemas e indices
    GET    /api/tables/{t}                 detalle de una tabla
    POST   /api/tables/{t}/reorganize      reorganiza el Sequential File
    GET    /api/tables/{t}/pages           inspeccion de paginas fisicas
    POST   /api/tables/{t}/seed            carga masiva de datos sinteticos
    GET    /api/health                     estado del motor

Aplicacion de IA y modulo multimedia (enunciado 5.3 y 5.5, Rol 4):
    GET    /api/rag/status                 corpus, indices y proveedor LLM
    POST   /api/rag/build                  carga el corpus de Wikipedia (en segundo plano)
    POST   /api/rag/ask                    pregunta -> lexico vs semantico -> LLM con citas
    GET    /api/multimedia/status          tienda de imagenes (BoVW + ResNet/IVF)
    POST   /api/multimedia/build           extrae caracteristicas e indexa Caltech-101
    POST   /api/multimedia/search          imagen (multipart o id) -> top-k por ambos paradigmas
    GET    /api/multimedia/samples         imagenes del dataset para usar como consulta
    GET    /api/multimedia/image/{id}      sirve el archivo de una imagen indexada

Endpoints declarados pero pendientes (devuelven 501 con el detalle de que falta):
    POST   /api/benchmarks/run             suite de los 4 experimentos
    GET    /api/tables/{t}/index/{nombre}  inspeccion de un indice B+/Hash
"""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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
from backend.multimedia.service import MultimediaStore
from backend.rag.pipeline import RagService
from backend.storage.page import NULL_PAGE
from backend.storage.schema import ColumnType

DATA_DIR = os.environ.get("BD2_DATA_DIR", "data")
PAGE_SIZE = int(os.environ.get("BD2_PAGE_SIZE", "4096"))
CACHE_SIZE = int(os.environ.get("BD2_CACHE_SIZE", "0"))

DATASETS_DIR = os.environ.get("BD2_DATASETS_DIR", str(Path(DATA_DIR) / "datasets"))
WIKI_CSV = os.environ.get("BD2_WIKI_CSV", str(Path(DATASETS_DIR) / "wiki_15k.csv"))

catalog = Catalog(DATA_DIR, page_size=PAGE_SIZE, cache_size=CACHE_SIZE)
executor = Executor(catalog)
# Los dos servicios del Rol 4 comparten el catalogo y el executor: toda su
# persistencia son tablas e indices del motor y todo su I/O pasa por el Pager.
rag = RagService(catalog, executor)
multimedia = MultimediaStore(catalog, executor, DATASETS_DIR)

def _precargar_modelos() -> None:
    """Carga SBERT y ResNet-18 en segundo plano si ya hay corpus/tienda.

    Sin esto la primera consulta de la demo paga ~1 minuto de carga del modelo.
    """
    import threading

    def tarea():
        try:
            if rag.corpus.is_ready():
                rag.embedder.embed_one("warmup")
            if multimedia.is_ready():
                multimedia._embedder._load()
        except Exception:  # noqa: BLE001 - la precarga es una optimizacion
            pass

    threading.Thread(target=tarea, daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("BD2_WARMUP", "1") == "1":
        _precargar_modelos()
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


class RagBuildRequest(BaseModel):
    csv_path: str | None = Field(None, description="CSV (lang,id,title,text); por defecto BD2_WIKI_CSV")
    limit: int | None = Field(None, ge=10, description="Tope de documentos (mitad por idioma); None = todos")
    lists: int | None = Field(None, ge=1, description="Celdas del IVF; None = floor(sqrt(N))")


class RagAskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("semantic", description="semantic | lexical | both")
    probes: int = Field(5, ge=1, le=200)
    scorer: str = Field("tfidf", description="tfidf | bm25 para la ruta lexica")
    generate: bool = Field(True, description="False: solo recuperacion, sin llamar al LLM")


class MultimediaBuildRequest(BaseModel):
    k_words: int = Field(256, ge=8, le=4096, description="Tamano del codebook BoVW")
    max_per_category: int | None = Field(None, ge=1, description="Tope de imagenes por categoria")
    lists: int | None = Field(None, ge=1)


class BenchmarkRequest(BaseModel):
    experiment: int = Field(
        0, 
        ge=0, 
        le=5,
        description="Número de experimento a correr (1..5). 0 para correr todos (¡puede tardar mucho!)."
    )


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
            "rtree": True,
            "spatial": True,
            "ivf": True,
            "rag": rag.corpus.is_ready(),
            "multimedia": multimedia.is_ready(),
        },
        "llm": rag.llm.describe(),
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
        # La reorganizacion reescribe el archivo principal: los RID fisicos de
        # los indices espaciales dejan de ser validos y hay que reindexar.
        for info in [i for i in catalog.indexes_of(table) if i["kind"] == "RTREE"]:
            catalog.drop_index(table, info["name"])
            catalog.create_index(table, info["name"], info["column"], "RTREE")
        catalog.invalidate_inverted(table)
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
                elif col.type is ColumnType.VECTOR:
                    # Mezcla de 32 gaussianas: datos con estructura de clusters,
                    # que es donde un IVF tiene algo que aprovechar.
                    r = random.Random(k)
                    centro = random.Random(1_000_003 * (k % 32))
                    out.append(tuple(centro.gauss(0, 4) + r.gauss(0, 1) for _ in range(col.length)))
                elif col.type is ColumnType.POINT:
                    # Puntos alrededor de Lima, para probar WITHIN_RADIUS/KNN.
                    r = random.Random(k)
                    out.append((
                        round(-77.0282 + r.uniform(-0.15, 0.15), 6),
                        round(-12.1354 + r.uniform(-0.12, 0.12), 6),
                    ))
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


# ----------------------------------------------------------- aplicacion de IA


@app.get("/api/rag/status")
def rag_status() -> dict:
    return rag.status()


@app.post("/api/rag/build")
def rag_build(req: RagBuildRequest = RagBuildRequest()) -> dict:
    """Carga el corpus en segundo plano: SBERT + tabla + IVF + indice invertido.

    Tarda minutos con 15 000 documentos (el embedding en CPU domina). El
    progreso se consulta en GET /api/rag/status.
    """
    if rag.corpus.progress["building"]:
        raise HTTPException(409, {"error": "ocupado", "detail": "ya hay una construccion en curso"})
    csv_path = req.csv_path or WIKI_CSV
    if not Path(csv_path).exists():
        raise HTTPException(404, {"error": "sin_corpus", "detail":
                            f"no existe {csv_path}. Descargalo con: python -m backend.rag.download "
                            f"--n 7500 --out {csv_path}"})
    rag.invalidate()
    rag.corpus.build_async(csv_path=csv_path, limit=req.limit, lists=req.lists)
    return {"started": True, "csv_path": csv_path, "limit": req.limit}


@app.post("/api/rag/ask")
def rag_ask(req: RagAskRequest) -> dict:
    if not rag.corpus.is_ready():
        raise HTTPException(409, {"error": "sin_corpus", "detail": "el corpus no esta cargado: POST /api/rag/build"})
    if req.mode not in ("semantic", "lexical", "both"):
        raise HTTPException(400, {"error": "modo_invalido", "detail": "mode debe ser semantic, lexical o both"})
    try:
        return rag.ask(req.question, k=req.k, mode=req.mode, probes=req.probes,
                       scorer=req.scorer, generate=req.generate)
    except EngineError as exc:
        raise _http(exc) from exc
    except Exception as exc:  # noqa: BLE001 - p. ej. el proveedor LLM rechazo la llamada
        raise HTTPException(502, {"error": "llm", "detail": str(exc)}) from exc


# --------------------------------------------------------------- multimedia


@app.get("/api/multimedia/status")
def multimedia_status() -> dict:
    return multimedia.status()


@app.post("/api/multimedia/build")
def multimedia_build(req: MultimediaBuildRequest = MultimediaBuildRequest()) -> dict:
    """Descarga Caltech-101 si falta, extrae SIFT + ResNet-18 e indexa (segundo plano)."""
    if multimedia.progress["building"]:
        raise HTTPException(409, {"error": "ocupado", "detail": "ya hay una construccion en curso"})
    multimedia.build_async(k_words=req.k_words, max_per_category=req.max_per_category, lists=req.lists)
    return {"started": True, **req.model_dump()}


@app.get("/api/multimedia/samples")
def multimedia_samples(n: int = Query(12, ge=1, le=60), seed: int | None = None) -> dict:
    if not multimedia.is_ready():
        raise HTTPException(409, {"error": "sin_tienda", "detail": "la tienda no esta construida: POST /api/multimedia/build"})
    return {"samples": multimedia.sample(n, seed)}


@app.get("/api/multimedia/image/{image_id}")
def multimedia_image(image_id: int) -> FileResponse:
    ruta = multimedia.image_path(image_id) if multimedia.is_ready() else None
    if ruta is None or not ruta.exists():
        raise HTTPException(404, {"error": "no_encontrada", "detail": f"no hay imagen con id {image_id}"})
    return FileResponse(ruta)


@app.post("/api/multimedia/search")
async def multimedia_search(
    file: UploadFile | None = File(None),
    image_id: int | None = Form(None),
    k: int = Form(10),
    probes: int = Form(5),
) -> dict:
    """Consulta por imagen: sube un archivo o indica el id de una imagen indexada.

    Devuelve el top-k de los dos paradigmas (BoVW + indice invertido vs.
    ResNet-18 + IVF) con el tiempo de extraccion, la latencia y el I/O de cada uno.
    """
    if not multimedia.is_ready():
        raise HTTPException(409, {"error": "sin_tienda", "detail": "la tienda no esta construida: POST /api/multimedia/build"})
    if file is not None:
        imagen = await file.read()
        origen = {"kind": "upload", "filename": file.filename}
    elif image_id is not None:
        ruta = multimedia.image_path(image_id)
        if ruta is None:
            raise HTTPException(404, {"error": "no_encontrada", "detail": f"no hay imagen con id {image_id}"})
        imagen = str(ruta)
        origen = {"kind": "indexed", "id": image_id, "archivo": ruta.name}
    else:
        raise HTTPException(400, {"error": "sin_imagen", "detail": "envia `file` o `image_id`"})
    try:
        # se pide uno de mas: si la consulta es una imagen indexada, ella misma
        # sale primera y no cuenta como acierto
        extra = 1 if origen["kind"] == "indexed" else 0
        r = multimedia.search_image(imagen, k=max(1, min(k, 50)) + extra, probes=max(1, probes))
    except ValueError as exc:
        raise HTTPException(400, {"error": "imagen_invalida", "detail": str(exc)}) from exc
    except EngineError as exc:
        raise _http(exc) from exc
    if extra:
        r["k"] -= 1
        for ruta_ in ("bovw", "dense"):
            r[ruta_]["hits"] = [h for h in r[ruta_]["hits"] if h["id"] != image_id][: r["k"]]
    return {"query": origen, **r}


# ------------------------------------------------- endpoints declarados, pendientes


@app.post("/api/benchmarks/run")
def run_benchmarks(req: BenchmarkRequest = BenchmarkRequest(experiment=0)) -> dict:
    import shutil
    from dataclasses import asdict
    from backend.benchmarks import experiment1, experiment2, experiment3, experiment4, experiment5
    
    bench_dir = Path(DATA_DIR) / "benchmarks"
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
        
    resultados = {}
    
    if req.experiment in (0, 1):
        # Experiment 1 returns a list of InsertResult (dataclass, needs dict conversion)
        res1 = experiment1.run(str(bench_dir / "exp1"))
        resultados["experiment_1"] = [r.__dict__ for r in res1]
        
    if req.experiment in (0, 2):
        resultados["experiment_2"] = experiment2.run(str(bench_dir / "exp2"))
        
    if req.experiment in (0, 3):
        resultados["experiment_3"] = experiment3.run(str(bench_dir / "exp3"))
        
    if req.experiment in (0, 4):
        resultados["experiment_4"] = experiment4.run(str(bench_dir / "exp4"))

    if req.experiment in (0, 5):
        res5 = experiment5.run(str(bench_dir / "exp5"))
        resultados["experiment_5"] = [asdict(r) for r in res5]

    return {
        "status": "ok",
        "results": resultados
    }


@app.get("/api/tables/{table}/index/{index_name}")
def inspect_index(table: str, index_name: str) -> dict:
    """Inspeccion fisica de un indice. RTREE expone altura, MBR y nodos."""
    try:
        info = next(
            (i for i in catalog.indexes_of(table) if i["name"] == index_name), None
        )
        if info is not None and info["kind"] == "RTREE":
            tree = catalog.open_index(table, index_name)
            stats = tree.inspect()
            return {
                "table": table,
                "name": index_name,
                "kind": "RTREE",
                "column": info["column"],
                **stats,
                "nodes": tree.sample_nodes(200),
            }
    except EngineError as exc:
        raise _http(exc) from exc
    raise HTTPException(
        501,
        {
            "error": "no_implementado",
            "detail": "La inspeccion detallada solo esta implementada para indices RTREE. "
            "Los indices BTREE, HASH e IVF se inspeccionan desde el plan de ejecucion "
            "y las metricas de I/O.",
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
