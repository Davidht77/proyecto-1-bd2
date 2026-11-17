"""Arbol R (R-Tree) multinivel sobre paginas de tamano fijo.

Diseno fisico
-------------
Un archivo por indice::

    <base>.rtree.dat

Cada entrada de nodo guarda un MBR (Minimum Bounding Rectangle) y el puntero
al hijo o al registro de la tabla::

    MBR = (xmin, ymin, xmax, ymax)     # float32, 8 bytes por dimension
    page_id / slot / area                # 8 bytes

La cabecera de pagina de 32 B se reutiliza tal cual. En la pagina 0 del archivo
se guardan ``user_a = root_page_id`` y ``user_b = height``. El esquema de la
cabecera lleva como primera columna el nombre de la columna espacial para que
``Catalog.indexes_of`` pueda reconstruir el indice sin metadatos externos.

Se implementan:
  - insercion con eleccion de rama por minima ampliacion de area y split
    cuadratico;
  - bulk_load STR para cargas masivas;
  - busqueda por ventana con poda de MBRs;
  - k-NN best-first con MINDIST y Max-Heap de candidatos;
  - borrado con actualizacion de MBRs (sin recompactacion agresiva).

El R-Tree de produccion es 2D. El parametro ``dim`` permite construir arboles
genericos para el Experimento 5 (maldicion de la dimensionalidad).
"""

from __future__ import annotations

import heapq
import math
import struct
from typing import Iterable, Iterator, NamedTuple

from backend.index import geo
from backend.storage.disk_counter import DiskCounter
from backend.storage.page import NULL_PAGE, Page, PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema


class RTreeRID(NamedTuple):
    """RID fisico de la tabla: pagina, slot y area (0=principal, 1=overflow)."""

    page_id: int
    slot: int
    area: int = 0

    @classmethod
    def from_engine_rid(cls, rid) -> "RTreeRID":
        area = getattr(rid, "area", 0)
        if isinstance(area, str):
            area = 1 if area == "overflow" else 0
        return cls(int(rid.page_id), int(rid.slot), int(area))


_RID = struct.Struct("<IIBB")  # page_id, slot, area, flags


class RTree:
    def __init__(
        self,
        base_path: str,
        point_column: Column | None = None,
        page_size: int = 4096,
        cache_size: int = 0,
        counter: DiskCounter | None = None,
        dim: int = 2,
        metric: str | None = None,
    ):
        if dim < 1:
            raise ValueError("dim debe ser >= 1")
        self.base_path = base_path
        self.point_column = point_column
        self.page_size = page_size
        self.dim = dim
        self.metric = metric or ("geo" if dim == 2 else "l2")
        self.counter = counter if counter is not None else DiskCounter()

        self._mbr_struct = struct.Struct("<" + "f" * (2 * dim))
        self._entry_size = self._mbr_struct.size + _RID.size
        self.capacity = page_capacity(page_size, self._entry_size)
        self.fanout = self.capacity

        # El esquema solo describe la cabecera autodescriptiva. Las entradas se
        # empaquetan con _mbr_struct + _RID; el primer nombre es el que lee el
        # catalogo para saber sobre que columna se construyo el indice.
        first_name = point_column.name if point_column is not None else "mbr"
        if dim == 2:
            schema_cols = [Column(first_name, ColumnType.POINT)]
            rest = self._entry_size - 16
        else:
            schema_cols = [Column(first_name, ColumnType.CHAR, 8 * dim)]
            rest = self._entry_size - 8 * dim
        if rest > 0:
            schema_cols.append(Column("entry_tail", ColumnType.CHAR, rest))
        # El esquema solo describe la cabecera autodescriptiva del archivo; la
        # PK ficticia debe ser una columna no compuesta (un POINT no puede serlo).
        self.entry_schema = Schema(schema_cols, pk_index=1 if len(schema_cols) > 1 else 0)

        self.pager = Pager(
            f"{base_path}.rtree.dat",
            record_size=self._entry_size,
            schema=self.entry_schema,
            page_size=page_size,
            cache_size=cache_size,
            counter=self.counter,
        )
        self.last_visited_mbrs: list[tuple[float, ...]] = []

    # ------------------------------------------------------- metadatos
    @property
    def root_page_id(self) -> int:
        return self.pager.user_a

    @root_page_id.setter
    def root_page_id(self, value: int) -> None:
        self.pager.user_a = value

    @property
    def height(self) -> int:
        return self.pager.user_b

    @height.setter
    def height(self, value: int) -> None:
        self.pager.user_b = value

    @property
    def is_empty(self) -> bool:
        return self.root_page_id == 0

    # --------------------------------------------------------- empaquetado
    def _pack_entry(self, mbr: tuple[float, ...], rid: RTreeRID) -> bytes:
        return self._mbr_struct.pack(*mbr) + _RID.pack(rid.page_id, rid.slot, rid.area, 0)

    def _unpack_entry(self, raw: bytes) -> tuple[tuple[float, ...], RTreeRID]:
        mbr = self._mbr_struct.unpack(raw[: self._mbr_struct.size])
        page_id, slot, area, _flags = _RID.unpack(raw[self._mbr_struct.size :])
        return mbr, RTreeRID(page_id, slot, area)

    def _append_entry(self, page: Page, mbr, rid: RTreeRID) -> None:
        slot = page.record_count
        if slot >= page.capacity:
            raise OverflowError(f"la pagina {page.page_id} esta llena")
        page.write_record(slot, self._pack_entry(tuple(mbr), rid))
        page.set_occupied(slot, True)
        page.record_count = slot + 1

    def _entries(self, page: Page) -> list[tuple[tuple[float, ...], RTreeRID]]:
        return [self._unpack_entry(page.read_record(s)) for s in range(page.record_count)]

    def _clear(self, page: Page) -> None:
        for slot in range(page.capacity):
            page.set_occupied(slot, False)
        page.record_count = 0

    def _write_entries(self, page: Page, entries: list[tuple[tuple[float, ...], RTreeRID]]) -> None:
        self._clear(page)
        for mbr, rid in entries:
            self._append_entry(page, mbr, rid)
        self.pager.write_page(page)

    # ------------------------------------------------------------- MBRs
    @staticmethod
    def _mbr_from_point(point) -> tuple[float, ...]:
        point = tuple(float(v) for v in point)
        return tuple(point) + tuple(point)

    @staticmethod
    def _union(a, b) -> tuple[float, ...]:
        if a is None:
            return tuple(b)
        if b is None:
            return tuple(a)
        dim = len(a) // 2
        return tuple(min(a[i], b[i]) for i in range(dim)) + tuple(
            max(a[dim + i], b[dim + i]) for i in range(dim)
        )

    @staticmethod
    def _area(mbr) -> float:
        dim = len(mbr) // 2
        area = 1.0
        for i in range(dim):
            area *= max(0.0, mbr[dim + i] - mbr[i])
        return area

    def _enlargement(self, mbr, point) -> float:
        point_mbr = self._mbr_from_point(point)
        return self._area(self._union(mbr, point_mbr)) - self._area(mbr)

    @staticmethod
    def _intersects(a, b) -> bool:
        dim = len(a) // 2
        for i in range(dim):
            if a[dim + i] < b[i] or b[dim + i] < a[i]:
                return False
        return True

    @staticmethod
    def _point_in_mbr(point, mbr) -> bool:
        dim = len(point)
        for i in range(dim):
            if point[i] < mbr[i] or point[i] > mbr[dim + i]:
                return False
        return True

    def _mbr_of_page(self, page: Page):
        mbr = None
        for entry_mbr, _rid in self._entries(page):
            mbr = self._union(mbr, entry_mbr)
        return mbr

    def _new_node(self, page_type: PageType) -> Page:
        return self.pager.new_page(page_type)

    # ------------------------------------------------------------ insercion
    def insert(self, point, rid: RTreeRID) -> None:
        point = tuple(float(v) for v in point)
        if len(point) != self.dim:
            raise ValueError(f"el punto tiene {len(point)} dimensiones, se esperaban {self.dim}")

        if self.is_empty:
            leaf = self._new_node(PageType.RT_LEAF)
            self._append_entry(leaf, self._mbr_from_point(point), rid)
            self.pager.write_page(leaf)
            self.root_page_id = leaf.page_id
            self.height = 1
            return

        split = self._insert_recursive(self.root_page_id, point, rid)
        if split is not None:
            right_id, right_mbr = split
            left = self.pager.read_page(self.root_page_id)
            left_mbr = self._mbr_of_page(left)
            root = self._new_node(PageType.RT_INTERNAL)
            self._append_entry(root, left_mbr, RTreeRID(self.root_page_id, 0, 0))
            self._append_entry(root, right_mbr, RTreeRID(right_id, 0, 0))
            self.pager.write_page(root)
            self.root_page_id = root.page_id
            self.height = self.height + 1

    def _insert_recursive(self, page_id: int, point, rid: RTreeRID):
        page = self.pager.read_page(page_id)

        if page.page_type == PageType.RT_LEAF:
            entries = self._entries(page)
            entries.append((self._mbr_from_point(point), rid))
            if len(entries) > page.capacity:
                return self._split_entries(page, entries)
            self._write_entries(page, entries)
            return None

        # Nodo interno: se desciende por el hijo que menos area crece.
        idx, child_id = self._choose_child(page, point)
        split = self._insert_recursive(child_id, point, rid)

        # El MBR del hijo elegido pudo cambiar por la insercion.
        page = self.pager.read_page(page_id)
        entries = self._entries(page)
        child = self.pager.read_page(child_id)
        entries[idx] = (self._mbr_of_page(child), entries[idx][1])
        if split is not None:
            right_id, right_mbr = split
            entries.append((right_mbr, RTreeRID(right_id, 0, 0)))
        if len(entries) > page.capacity:
            return self._split_entries(page, entries)
        self._write_entries(page, entries)
        return None

    def _choose_child(self, page: Page, point) -> tuple[int, int]:
        entries = self._entries(page)
        best_idx = 0
        best_score = None
        for idx, (mbr, rid) in enumerate(entries):
            score = (self._enlargement(mbr, point), self._area(mbr))
            if best_score is None or score < best_score:
                best_score, best_idx = score, idx
        return best_idx, entries[best_idx][1].page_id

    def _split_entries(self, page: Page, entries: list[tuple[tuple[float, ...], RTreeRID]]):
        """Reparte ``entries`` (capacity + 1) entre ``page`` y una pagina nueva.

        Split cuadratico: se eligen las dos entradas que dejen mas area muerta
        como semillas y luego se asigna cada entrada al grupo cuya area crece
        menos, con desempate por balance de ocupacion.
        """
        if len(entries) < 2:
            raise ValueError("no se puede dividir un nodo con menos de 2 entradas")

        seed_a = seed_b = 0
        worst = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                union = self._union(entries[i][0], entries[j][0])
                dead = self._area(union) - self._area(entries[i][0]) - self._area(entries[j][0])
                if worst is None or dead > worst:
                    worst, seed_a, seed_b = dead, i, j

        group_a = [entries[seed_a]]
        group_b = [entries[seed_b]]
        remaining = [e for k, e in enumerate(entries) if k not in (seed_a, seed_b)]

        while remaining:
            mbr_a = self._mbr_of_entries(group_a)
            mbr_b = self._mbr_of_entries(group_b)
            best = None
            for k, (mbr, _rid) in enumerate(remaining):
                da = self._area(self._union(mbr_a, mbr)) - self._area(mbr_a)
                db = self._area(self._union(mbr_b, mbr)) - self._area(mbr_b)
                balance = abs(len(group_a) - len(group_b))
                score = (min(da, db), max(da, db), balance, k)
                if best is None or score < best[0]:
                    best = (score, k, da <= db)
            _, k, to_a = best
            item = remaining.pop(k)
            (group_a if to_a else group_b).append(item)

        self._write_entries(page, group_a)
        right = self._new_node(page.page_type)
        self._write_entries(right, group_b)
        return right.page_id, self._mbr_of_entries(group_b)

    def _mbr_of_entries(self, entries):
        mbr = None
        for entry_mbr, _rid in entries:
            mbr = self._union(mbr, entry_mbr)
        return mbr

    # ----------------------------------------------------------- bulk load
    def bulk_load(self, entries: Iterable[tuple[tuple[float, ...], RTreeRID]]) -> None:
        """Carga masiva con STR (Sort-Tile-Recursive). Deja el arbol balanceado."""
        items = [(tuple(float(v) for v in point), rid) for point, rid in entries]
        if not items:
            return
        if not self.is_empty:
            raise ValueError("bulk_load requiere un arbol vacio")

        # STR: ordenar por x, partir en franjas y ordenar cada franja por y.
        n = len(items)
        n_leaves = max(1, math.ceil(n / self.capacity))
        n_slices = max(1, math.ceil(math.sqrt(n_leaves)))
        slice_size = max(1, math.ceil(n / n_slices))
        items.sort(key=lambda item: item[0][0])

        leaves: list[tuple[tuple[float, ...], int]] = []
        for start in range(0, n, slice_size):
            tile = items[start : start + slice_size]
            if self.dim > 1:
                tile.sort(key=lambda item: item[0][1])
            for pos in range(0, len(tile), self.capacity):
                group = tile[pos : pos + self.capacity]
                page = self._new_node(PageType.RT_LEAF)
                for point, rid in group:
                    self._append_entry(page, self._mbr_from_point(point), rid)
                self.pager.write_page(page)
                leaves.append((self._mbr_of_page(page), page.page_id))

        level = leaves
        height = 1
        while len(level) > 1:
            next_level: list[tuple[tuple[float, ...], int]] = []
            for start in range(0, len(level), self.capacity):
                group = level[start : start + self.capacity]
                page = self._new_node(PageType.RT_INTERNAL)
                for mbr, child_id in group:
                    self._append_entry(page, mbr, RTreeRID(child_id, 0, 0))
                self.pager.write_page(page)
                next_level.append((self._mbr_of_page(page), page.page_id))
            level = next_level
            height += 1

        self.root_page_id = level[0][1]
        self.height = height
        self.pager.flush()

    # ------------------------------------------------------------ busquedas
    def search_window(self, window: tuple[float, ...], collect_mbrs: bool = True) -> list[RTreeRID]:
        """Devuelve los RID cuyos MBRs intersectan la ventana.

        La ventana es (xmin, ymin, xmax, ymax) en 2D o (min...max...) en D
        dimensiones. Para radio geografico, el ejecutor hace el filtro exacto
        con haversine sobre las filas devueltas.
        """
        window = tuple(float(v) for v in window)
        if len(window) != 2 * self.dim:
            raise ValueError(
                f"la ventana tiene {len(window)} valores, se esperaban {2 * self.dim}"
            )
        self.last_visited_mbrs = []
        out: list[RTreeRID] = []
        if self.is_empty:
            return out
        self._search_node(self.root_page_id, window, out, collect_mbrs)
        return out

    def _search_node(self, page_id: int, window, out, collect: bool) -> None:
        page = self.pager.read_page(page_id)
        page_mbr = self._mbr_of_page(page)
        if page_mbr is None or not self._intersects(page_mbr, window):
            return
        if collect:
            self.last_visited_mbrs.append(page_mbr)
        if page.page_type == PageType.RT_LEAF:
            for mbr, rid in self._entries(page):
                if self._intersects(mbr, window):
                    out.append(rid)
            return
        for mbr, child in self._entries(page):
            if self._intersects(mbr, window):
                self._search_node(child.page_id, window, out, collect)

    def knn(self, center, k: int, collect_mbrs: bool = False) -> list[RTreeRID]:
        if k <= 0 or self.is_empty:
            return []
        center = tuple(float(v) for v in center)
        if len(center) != self.dim:
            raise ValueError(
                f"el centro tiene {len(center)} dimensiones, se esperaban {self.dim}"
            )
        self.last_visited_mbrs = []

        root = self.pager.read_page(self.root_page_id)
        root_mbr = self._mbr_of_page(root)
        nodes = [(self._mindist(center, root_mbr), self.root_page_id)]
        best: list[tuple[float, int, int, int]] = []  # (-dist, page_id, slot, area)
        while nodes:
            dist, page_id = heapq.heappop(nodes)
            if len(best) >= k and dist > -best[0][0]:
                break
            page = self.pager.read_page(page_id)
            if collect_mbrs:
                self.last_visited_mbrs.append(self._mbr_of_page(page))
            if page.page_type == PageType.RT_LEAF:
                for mbr, rid in self._entries(page):
                    point = mbr[: self.dim]
                    d = self._distance(center, point)
                    if len(best) < k:
                        heapq.heappush(best, (-d, rid.page_id, rid.slot, rid.area))
                    elif d < -best[0][0]:
                        heapq.heapreplace(best, (-d, rid.page_id, rid.slot, rid.area))
            else:
                for mbr, child in self._entries(page):
                    d = self._mindist(center, mbr)
                    if len(best) < k or d <= -best[0][0]:
                        heapq.heappush(nodes, (d, child.page_id))

        ordered = sorted(best, key=lambda item: -item[0])
        return [RTreeRID(page_id, slot, area) for _d, page_id, slot, area in ordered]

    def _distance(self, a, b) -> float:
        if self.metric == "geo":
            return geo.haversine(a[0], a[1], b[0], b[1])
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    def _mindist(self, center, mbr) -> float:
        if self.metric == "geo":
            return geo.min_distance_to_bbox(center[0], center[1], mbr)
        dim = len(center)
        total = 0.0
        for i in range(dim):
            if center[i] < mbr[i]:
                delta = mbr[i] - center[i]
            elif center[i] > mbr[dim + i]:
                delta = center[i] - mbr[dim + i]
            else:
                delta = 0.0
            total += delta * delta
        return math.sqrt(total)

    # --------------------------------------------------------------- baja
    def delete(self, point, rid: RTreeRID) -> bool:
        if self.is_empty:
            return False
        found, empty = self._delete_node(self.root_page_id, rid)
        if found and empty:
            self.pager.free_page(self.root_page_id)
            self.root_page_id = 0
            self.height = 0
        return found

    def _delete_node(self, page_id: int, rid: RTreeRID) -> tuple[bool, bool]:
        page = self.pager.read_page(page_id)
        entries = self._entries(page)

        if page.page_type == PageType.RT_LEAF:
            for slot, (_mbr, entry_rid) in enumerate(entries):
                if entry_rid == rid:
                    del entries[slot]
                    self._write_entries(page, entries)
                    return True, not entries
            return False, False

        for slot, (_mbr, child) in enumerate(entries):
            child_id = child.page_id
            found, empty = self._delete_node(child_id, rid)
            if not found:
                continue

            page = self.pager.read_page(page_id)
            entries = self._entries(page)
            if empty:
                del entries[slot]
                self.pager.free_page(child_id)
            else:
                child_page = self.pager.read_page(child_id)
                entries[slot] = (self._mbr_of_page(child_page), entries[slot][1])
            self._write_entries(page, entries)
            return True, not entries
        return False, False

    # --------------------------------------------------------- inspeccion
    def inspect(self, max_nodes: int = 20_000) -> dict:
        stats = {
            "kind": "RTREE",
            "dim": self.dim,
            "metric": self.metric,
            "height": self.height,
            "capacity": self.capacity,
            "fanout": self.fanout,
            "root_page_id": self.root_page_id,
            "node_count": 0,
            "leaf_count": 0,
            "internal_count": 0,
            "entries": 0,
            "fill_pct": 0.0,
            "root_mbr": None,
        }
        if self.is_empty:
            return stats
        visited: set[int] = set()
        fills: list[float] = []

        def walk(page_id: int) -> None:
            if len(visited) >= max_nodes or page_id in visited:
                return
            visited.add(page_id)
            page = self.pager.read_page(page_id)
            stats["node_count"] += 1
            stats["entries"] += page.record_count
            fills.append(100.0 * page.record_count / page.capacity)
            if page.page_type == PageType.RT_LEAF:
                stats["leaf_count"] += 1
            else:
                stats["internal_count"] += 1
                for _mbr, child in self._entries(page):
                    walk(child.page_id)

        root = self.pager.read_page(self.root_page_id)
        stats["root_mbr"] = list(self._mbr_of_page(root) or ())
        walk(self.root_page_id)
        if fills:
            stats["fill_pct"] = round(sum(fills) / len(fills), 1)
        return stats

    def sample_nodes(self, limit: int = 200) -> list[dict]:
        """Muestra BFS de nodos con su MBR, para dibujar el arbol en el mapa."""
        if limit <= 0 or self.is_empty:
            return []
        out: list[dict] = []
        queue = [self.root_page_id]
        seen: set[int] = set()
        while queue and len(out) < limit:
            page_id = queue.pop(0)
            if page_id in seen:
                continue
            seen.add(page_id)
            page = self.pager.read_page(page_id)
            mbr = self._mbr_of_page(page)
            out.append({
                "page_id": page_id,
                "page_type": page.page_type.name,
                "record_count": page.record_count,
                "capacity": page.capacity,
                "mbr": list(mbr) if mbr is not None else None,
            })
            if page.page_type != PageType.RT_LEAF:
                for _mbr, child in self._entries(page):
                    queue.append(child.page_id)
        return out

    # ------------------------------------------------------------------ misc
    def flush(self) -> None:
        self.pager.flush()

    def close(self) -> None:
        self.pager.close()

    def __enter__(self) -> "RTree":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RTree({self.base_path}, dim={self.dim}, height={self.height}, "
            f"fanout={self.fanout})"
        )
