"""IVF (Inverted File Flat) en disco — enunciado 5.3.3, opcion 1.

El espacio R^D se parte en C celdas de Voronoi con K-Means (C = floor(sqrt(N))
por defecto). Una consulta solo escanea las `probes` celdas cuyos centroides
estan mas cerca de q: en vez de N distancias calcula C + probes * N/C.

Dos archivos, ambos paginados por el Pager (todo I/O pasa por el DiskCounter):

    <base>.ivf.dir  Directorio de centroides. Un registro por celda:
                        <centroide VECTOR(D), cluster_id, head, tail, count>
                    La celda i vive en la pagina 1 + i // cap, slot i % cap.
                    La primera columna se llama como la columna indexada, asi
                    que peek_header() recupera columna y dimension sin
                    metadatos externos (mismo criterio que .hash.dir).
                    user_a = metrica (0 = L2, 1 = COSINE), user_b = C.

    <base>.ivf.lst  Listas invertidas paginadas. Cada celda es una cadena de
                    paginas de 4 KB enlazadas por next/prev_page_id, con
                    registros densos <pk, page, slot, area, vec>.
                    user_a = cantidad total de vectores indexados.

Por que se guarda la PK ademas del RID: en este motor los RID no son estables.
El Heap File borra con move-the-last (el ultimo registro hereda el RID del
borrado) y el Sequential File desplaza slots al insertar ordenado y renumera
todo al reorganizar. Guardar <pk, RID> permite verificar en la lectura que el
RID sigue apuntando a la fila correcta y, si no, recuperarla por su clave.

Los centroides se mantienen en RAM despues de abrir el indice (se leen una vez,
contando el I/O). Es lo que hacen los motores reales: el cuantizador es chico
(C*D floats) y releerlo en cada consulta duplicaria el costo de probes=1 sin
aportar nada al experimento de Recall vs I/O.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple

import numpy as np

from backend.storage.disk_counter import DiskCounter
from backend.storage.page import HEADER_SIZE, NULL_PAGE, Page, PageType, bitmap_size, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema
from backend.vector.distance import CODE_METRIC, METRIC_CODE, METRICS, distances
from backend.vector.kmeans import assign, kmeans

_NULL = 0  # head/tail vacios: la pagina 0 es la cabecera, nunca una lista


class VRID(NamedTuple):
    """RID tal como lo guarda el indice. area: 0 = principal/heap, 1 = overflow."""

    page_id: int
    slot: int
    area: int = 0

    @classmethod
    def from_engine_rid(cls, rid) -> "VRID":
        area = getattr(rid, "area", 0)
        if isinstance(area, str):
            area = 1 if area == "overflow" else 0
        return cls(rid.page_id, rid.slot, int(area))


class Candidate(NamedTuple):
    distance: float
    pk: object
    rid: VRID


def _np_dtype(col: Column) -> tuple[str, object]:
    if col.type is ColumnType.INT:
        return (col.name, "<i4")
    if col.type is ColumnType.FLOAT:
        return (col.name, "<f8")
    if col.type is ColumnType.CHAR:
        return (col.name, f"S{col.length}")
    return (col.name, "<f4", (col.length,))


def _decode_pk(col: Column, value):
    if col.type is ColumnType.CHAR:
        return bytes(value).split(b"\x00", 1)[0].decode("utf-8")
    if col.type is ColumnType.INT:
        return int(value)
    return float(value)


class IVFIndex:
    def __init__(
        self,
        base_path: str,
        column: Column | None = None,
        pk_column: Column | None = None,
        metric: str = "L2",
        page_size: int = 4096,
        cache_size: int = 0,
        counter: DiskCounter | None = None,
    ):
        self.base_path = base_path
        self.page_size = page_size
        self.counter = counter if counter is not None else DiskCounter()

        dir_path, lst_path = f"{base_path}.ivf.dir", f"{base_path}.ivf.lst"
        nuevo = column is not None
        if nuevo:
            if column.type is not ColumnType.VECTOR:
                raise ValueError(f"IVF solo indexa columnas VECTOR, «{column.name}» es {column.type.name}")
            if metric not in METRICS:
                raise ValueError(f"metrica desconocida: {metric}. Usa L2 o COSINE.")
            dir_schema = Schema(
                [Column(column.name, ColumnType.VECTOR, column.length),
                 Column("cluster_id", ColumnType.INT), Column("head", ColumnType.INT),
                 Column("tail", ColumnType.INT), Column("count", ColumnType.INT)],
                pk_index=1,
            )
            lst_schema = Schema(
                [Column("pk", pk_column.type, pk_column.length),
                 Column("page", ColumnType.INT), Column("slot", ColumnType.INT),
                 Column("area", ColumnType.INT), Column("vec", ColumnType.VECTOR, column.length)],
                pk_index=0,
            )
        else:
            dir_schema = lst_schema = None

        self.dir = Pager(dir_path, record_size=dir_schema.record_size if dir_schema else _peek_rs(dir_path),
                         schema=dir_schema, page_size=page_size, cache_size=cache_size, counter=self.counter)
        self.lst = Pager(lst_path, record_size=lst_schema.record_size if lst_schema else _peek_rs(lst_path),
                         schema=lst_schema, page_size=page_size, cache_size=cache_size, counter=self.counter)
        if nuevo:
            self.dir.user_a = METRIC_CODE[metric]

        self.dir_schema: Schema = self.dir.schema
        self.lst_schema: Schema = self.lst.schema
        self.column: Column = self.dir_schema.columns[0]
        self.pk_column: Column = self.lst_schema.columns[0]
        self.dim = self.column.length
        self.metric = CODE_METRIC[self.dir.user_a]

        self._dir_cap = page_capacity(page_size, self.dir_schema.record_size)
        self._lst_cap = page_capacity(page_size, self.lst_schema.record_size)
        self._lst_offset = HEADER_SIZE + bitmap_size(self._lst_cap)
        self._lst_dtype = np.dtype([
            _np_dtype(Column("pk", self.pk_column.type, self.pk_column.length)),
            ("page", "<i4"), ("slot", "<i4"), ("area", "<i4"),
            ("vec", "<f4", (self.dim,)),
        ])
        assert self._lst_dtype.itemsize == self.lst_schema.record_size

        self._load_directory()

    # --------------------------------------------------------------- directorio

    @property
    def n_lists(self) -> int:
        return self.dir.user_b

    @property
    def n_vectors(self) -> int:
        return self.lst.user_a

    @property
    def records_per_page(self) -> int:
        return self._lst_cap

    def _load_directory(self) -> None:
        C = self.n_lists
        self.centroids = np.zeros((C, self.dim), dtype=np.float32)
        self.heads = [_NULL] * C
        self.tails = [_NULL] * C
        self.counts = [0] * C
        for i in range(C):
            if i % self._dir_cap == 0:
                page = self.dir.read_page(1 + i // self._dir_cap)
            vec, cid, head, tail, cnt = self.dir_schema.unpack(page.read_record(i % self._dir_cap))
            self.centroids[cid] = vec
            self.heads[cid], self.tails[cid], self.counts[cid] = head, tail, cnt

    def _write_dir_page(self, page_no: int) -> None:
        """Reescribe una pagina del directorio desde la copia en RAM (sin leerla)."""
        page = Page(page_no, self.page_size, self.dir_schema.record_size)
        page.page_type = PageType.DATA
        first = (page_no - 1) * self._dir_cap
        n = 0
        for i in range(first, min(first + self._dir_cap, self.n_lists)):
            page.write_record(n, self.dir_schema.pack(
                (self.centroids[i], i, self.heads[i], self.tails[i], self.counts[i])))
            page.set_occupied(n, True)
            n += 1
        page.record_count = n
        while self.dir.page_count <= page_no:
            self.dir.new_page(PageType.DATA)
        self.dir.write_page(page)

    def _write_dir_entry(self, cluster: int) -> None:
        self._write_dir_page(1 + cluster // self._dir_cap)

    # -------------------------------------------------------------- construccion

    def build(
        self,
        entries: Iterable[tuple],
        n_lists: int | None = None,
        n_iter: int = 20,
        seed: int = 0,
    ) -> dict:
        """Construye el indice desde cero. `entries` = (pk, rid, vector).

        Entrena K-Means, asigna cada vector a su celda y escribe cada lista
        como una cadena de paginas contiguas: las paginas de una misma celda
        quedan juntas en el archivo, lo que favorece la lectura secuencial.
        """
        if self.n_vectors or self.n_lists:
            raise ValueError("build() espera un indice vacio")
        pks, rids, vecs = [], [], []
        for pk, rid, vec in entries:
            pks.append(pk)
            rids.append(VRID.from_engine_rid(rid))
            vecs.append(vec)
        N = len(vecs)
        if N == 0:
            return {"n_vectors": 0, "n_lists": 0}

        X = np.asarray(vecs, dtype=np.float32)
        C = n_lists if n_lists else max(1, math.isqrt(N))
        C = max(1, min(C, N))
        self.centroids = kmeans(X, C, metric=self.metric, n_iter=n_iter, seed=seed)
        C = len(self.centroids)
        labels = assign(self._for_quantizer(X), self.centroids)

        self.dir.user_b = C
        self.heads, self.tails, self.counts = [_NULL] * C, [_NULL] * C, [0] * C
        # Agrupar sin O(C*N): ordenar por celda una vez y cortar por fronteras.
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        bounds = np.searchsorted(sorted_labels, np.arange(C + 1))
        for c in range(C):
            idx = order[bounds[c] : bounds[c + 1]]
            self._write_list(c, ((pks[i], rids[i], X[i]) for i in idx))
        self.lst.user_a = N
        for page_no in range(1, 1 + math.ceil(C / self._dir_cap)):
            self._write_dir_page(page_no)
        self.flush()
        return {"n_vectors": N, "n_lists": C}

    def _write_list(self, c: int, items: Iterable[tuple]) -> None:
        page = None
        for pk, rid, vec in items:
            if page is None or page.is_full():
                nueva = self.lst.new_page(PageType.DATA)
                if page is None:
                    self.heads[c] = nueva.page_id
                else:
                    page.next_page_id = nueva.page_id
                    nueva.prev_page_id = page.page_id
                    self.lst.write_page(page)
                page = nueva
            n = page.record_count
            page.write_record(n, self.lst_schema.pack((pk, rid.page_id, rid.slot, rid.area, vec)))
            page.set_occupied(n, True)
            page.record_count = n + 1
            self.counts[c] += 1
        if page is not None:
            self.lst.write_page(page)
            self.tails[c] = page.page_id

    def _for_quantizer(self, X: np.ndarray) -> np.ndarray:
        """Con COSINE el cuantizador compara direcciones: se normaliza antes de asignar."""
        if self.metric != "COSINE":
            return X
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n > 0, n, 1.0)

    def nearest_cluster(self, vec) -> int:
        X = self._for_quantizer(np.asarray([vec], dtype=np.float32))
        return int(assign(X, self.centroids)[0])

    # ------------------------------------------------------------ mantenimiento

    def insert(self, pk, rid, vec) -> int:
        """Agrega un vector a la cola de la lista de su celda. Devuelve la celda."""
        rid = VRID.from_engine_rid(rid)
        if self.n_lists == 0:
            # Indice creado sobre una tabla vacia: el primer vector inaugura
            # una celda unica. Conviene reconstruir el indice tras la carga.
            self.centroids = np.asarray([vec], dtype=np.float32)
            self.heads, self.tails, self.counts = [_NULL], [_NULL], [0]
            self.dir.user_b = 1
        c = self.nearest_cluster(vec)
        payload = self.lst_schema.pack((pk, rid.page_id, rid.slot, rid.area, vec))
        if self.tails[c] == _NULL:
            page = self.lst.new_page(PageType.DATA)
            self.heads[c] = self.tails[c] = page.page_id
        else:
            page = self.lst.read_page(self.tails[c])
            if page.is_full():
                nueva = self.lst.new_page(PageType.DATA)
                page.next_page_id = nueva.page_id
                nueva.prev_page_id = page.page_id
                self.lst.write_page(page)
                page = nueva
                self.tails[c] = page.page_id
        n = page.record_count
        page.write_record(n, payload)
        page.set_occupied(n, True)
        page.record_count = n + 1
        self.lst.write_page(page)
        self.counts[c] += 1
        self.lst.user_a = self.n_vectors + 1
        self._write_dir_entry(c)
        return c

    def _locate(self, pk, vec) -> tuple[int, Page, int] | None:
        """(celda, pagina, slot) de la entrada con esa PK. Busca primero en la
        celda de `vec`; si no esta (no deberia pasar), recorre todas."""
        primera = self.nearest_cluster(vec) if self.n_lists else None
        orden = ([primera] if primera is not None else []) + [
            c for c in range(self.n_lists) if c != primera
        ]
        for c in orden:
            pid = self.heads[c]
            while pid != _NULL and pid != NULL_PAGE:
                page = self.lst.read_page(pid)
                for slot in range(page.record_count):
                    if self.lst_schema.key_from_bytes(page.read_record(slot)) == pk:
                        return c, page, slot
                pid = page.next_page_id
        return None

    def delete(self, pk, vec) -> bool:
        """Quita la entrada de `pk`. Compacta con move-the-last dentro de la cadena
        para que las paginas sigan densas (la lectura por numpy lo asume)."""
        hallado = self._locate(pk, vec)
        if hallado is None:
            return False
        c, page, slot = hallado
        tail = page if page.page_id == self.tails[c] else self.lst.read_page(self.tails[c])
        last = tail.record_count - 1
        if not (tail.page_id == page.page_id and slot == last):
            page.write_record(slot, tail.read_record(last))
            if page is not tail:
                self.lst.write_page(page)
        tail.set_occupied(last, False)
        tail.record_count = last
        if tail.record_count == 0:
            prev = tail.prev_page_id
            self.lst.free_page(tail.page_id)
            if prev == NULL_PAGE:
                self.heads[c] = self.tails[c] = _NULL
            else:
                prev_page = self.lst.read_page(prev)
                prev_page.next_page_id = NULL_PAGE
                self.lst.write_page(prev_page)
                self.tails[c] = prev
        else:
            self.lst.write_page(tail)
        self.counts[c] -= 1
        self.lst.user_a = self.n_vectors - 1
        self._write_dir_entry(c)
        return True

    def relocate(self, pk, vec, new_rid) -> bool:
        """Actualiza el RID de una entrada (p. ej. tras un move-the-last del Heap)."""
        hallado = self._locate(pk, vec)
        if hallado is None:
            return False
        _, page, slot = hallado
        rid = VRID.from_engine_rid(new_rid)
        pk_, _, _, _, v = self.lst_schema.unpack(page.read_record(slot))
        page.write_record(slot, self.lst_schema.pack((pk_, rid.page_id, rid.slot, rid.area, v)))
        self.lst.write_page(page)
        return True

    # ------------------------------------------------------------------ consulta

    def probe_order(self, q) -> np.ndarray:
        """Celdas ordenadas por distancia de su centroide a q."""
        if self.n_lists == 0:
            return np.empty(0, dtype=np.int64)
        qq = self._for_quantizer(np.asarray([q], dtype=np.float32))[0]
        return np.argsort(distances(qq, self.centroids, "L2"), kind="stable")

    def scan_lists(self, q, clusters: Iterable[int]) -> list[Candidate]:
        """Lee las paginas de las celdas dadas y devuelve sus candidatos
        ordenados por distancia exacta a q."""
        bloques = []
        for c in clusters:
            pid = self.heads[int(c)]
            while pid != _NULL and pid != NULL_PAGE:
                page = self.lst.read_page(pid)
                n = page.record_count
                if n:
                    # Decodifica la pagina completa de una vez: los registros son
                    # densos y de ancho fijo, asi que es un arreglo estructurado.
                    raw = page.data[self._lst_offset : self._lst_offset + n * self.lst_schema.record_size]
                    bloques.append(np.frombuffer(bytes(raw), dtype=self._lst_dtype))
                pid = page.next_page_id
        if not bloques:
            return []
        recs = np.concatenate(bloques)
        d = distances(q, recs["vec"], self.metric)
        orden = np.argsort(d, kind="stable")
        return [
            Candidate(float(d[i]), _decode_pk(self.pk_column, recs["pk"][i]),
                      VRID(int(recs["page"][i]), int(recs["slot"][i]), int(recs["area"][i])))
            for i in orden
        ]

    def search(self, q, k: int, probes: int) -> list[Candidate]:
        """k vecinos aproximados escaneando las `probes` celdas mas cercanas."""
        probes = max(1, min(probes, self.n_lists))
        return self.scan_lists(q, self.probe_order(q)[:probes])[:k]

    # --------------------------------------------------------------------- misc

    def stats(self) -> dict:
        paginas = self.lst.page_count - 1
        return {
            "metric": self.metric,
            "dim": self.dim,
            "n_lists": self.n_lists,
            "n_vectors": self.n_vectors,
            "list_pages": paginas,
            "records_per_page": self._lst_cap,
            "avg_pages_per_list": round(paginas / self.n_lists, 2) if self.n_lists else 0.0,
            "dir_pages": self.dir.page_count - 1,
        }

    def flush(self) -> None:
        self.dir.flush()
        self.lst.flush()

    def close(self) -> None:
        self.dir.close()
        self.lst.close()

    def __enter__(self) -> "IVFIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _peek_rs(path: str) -> int:
    from backend.storage.pager import peek_header

    return peek_header(path)["record_size"]
