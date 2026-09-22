"""Hashing Dinamico en disco: Extendible Hashing.

Dos archivos:
    <base>.hash.dir  directorio de 2**D punteros a buckets (uint32)
    <base>.hash.bkt  buckets: una pagina = un bucket, profundidad local en
                     page.aux_page_id

Referencia: PPT "05 Hash, Bitmap Index Scan y BRIN en PostgreSQL" (Semana 05,
CS2042), diapositivas 16-27 (Extendible Hashing).
"""

from __future__ import annotations

import struct
import zlib
from typing import NamedTuple

from backend.storage.page import NULL_PAGE, Page, PageType, page_capacity
from backend.storage.pager import Pager
from backend.storage.schema import Column, ColumnType, Schema

_RID_STRUCT = struct.Struct("<IHB")  # page_id, slot, area (0=main/heap, 1=overflow)
_DIR_STRUCT = struct.Struct("<I")


class RID(NamedTuple):
    page_id: int
    slot: int
    area: int = 0

    @classmethod
    def from_engine_rid(cls, rid) -> "RID":
        area = 1 if getattr(rid, "area", "main") == "overflow" else 0
        return cls(rid.page_id, rid.slot, area)


def _key_size(column: Column) -> int:
    if column.type is ColumnType.INT:
        return 4
    if column.type is ColumnType.FLOAT:
        return 8
    return column.length


def _entry_size(column: Column) -> int:
    return _key_size(column) + _RID_STRUCT.size


def _pack_key(column: Column, key) -> bytes:
    if column.type is ColumnType.INT:
        return struct.pack("<i", int(key))
    if column.type is ColumnType.FLOAT:
        return struct.pack("<d", float(key))
    raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    return raw.ljust(column.length, b"\x00")[: column.length]


def _unpack_key(column: Column, raw: bytes):
    if column.type is ColumnType.INT:
        return struct.unpack("<i", raw)[0]
    if column.type is ColumnType.FLOAT:
        return struct.unpack("<d", raw)[0]
    return raw.split(b"\x00", 1)[0].decode("utf-8")


def _pack_entry(column: Column, key, rid: RID) -> bytes:
    return _pack_key(column, key) + _RID_STRUCT.pack(rid.page_id, rid.slot, rid.area)


def _unpack_entry(column: Column, raw: bytes) -> tuple:
    ksize = _key_size(column)
    key = _unpack_key(column, raw[:ksize])
    page_id, slot, area = _RID_STRUCT.unpack(raw[ksize : ksize + _RID_STRUCT.size])
    return key, RID(page_id, slot, area)


def hash_index(key, column: Column, depth: int) -> int:
    mask = (1 << depth) - 1
    if column.type is ColumnType.INT:
        h = int(key)
    elif column.type is ColumnType.FLOAT:
        h = zlib.crc32(struct.pack("<d", float(key)))
    else:
        raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        h = zlib.crc32(raw)
    return h & mask


class ExtendibleHash:
    def __init__(self, base_path: str, column: Column, page_size: int = 4096, cache_size: int = 0):
        self.base_path = base_path
        self.column = column
        self.page_size = page_size
        self._dir_capacity = page_capacity(page_size, 4)

        schema = Schema([column])
        self.dir_pager = Pager(
            f"{base_path}.hash.dir", record_size=4, schema=schema,
            page_size=page_size, cache_size=cache_size,
        )
        self.bkt_pager = Pager(
            f"{base_path}.hash.bkt", record_size=_entry_size(column),
            page_size=page_size, cache_size=cache_size,
        )

        if self.bkt_pager.page_count == 1:
            self._dir_page_ids: list[int] = []
            bucket = self.bkt_pager.new_page(PageType.HASH)
            bucket.aux_page_id = 0
            self.bkt_pager.write_page(bucket)
            self.dir_pager.user_a = 0
            self._dir_grow_to(1)
            self._dir_set(0, bucket.page_id)
        else:
            self._dir_page_ids = []
            pid = self.dir_pager.user_b
            while pid != NULL_PAGE:
                self._dir_page_ids.append(pid)
                pid = self.dir_pager.read_page(pid).next_page_id

    # ------------------------------------------------------------ directorio

    @property
    def global_depth(self) -> int:
        return self.dir_pager.user_a

    @global_depth.setter
    def global_depth(self, value: int) -> None:
        self.dir_pager.user_a = value

    def _dir_grow_to(self, n_entries: int) -> None:
        needed = -(-n_entries // self._dir_capacity)
        while len(self._dir_page_ids) < needed:
            page = self.dir_pager.new_page(PageType.HASH)
            if self._dir_page_ids:
                prev = self.dir_pager.read_page(self._dir_page_ids[-1])
                prev.next_page_id = page.page_id
                self.dir_pager.write_page(prev)
            else:
                self.dir_pager.user_b = page.page_id
            self._dir_page_ids.append(page.page_id)

    def _dir_get(self, index: int) -> int:
        page_idx, slot = divmod(index, self._dir_capacity)
        page = self.dir_pager.read_page(self._dir_page_ids[page_idx])
        return _DIR_STRUCT.unpack(page.read_record(slot))[0]

    def _dir_set(self, index: int, bucket_page_id: int) -> None:
        page_idx, slot = divmod(index, self._dir_capacity)
        page = self.dir_pager.read_page(self._dir_page_ids[page_idx])
        page.write_record(slot, _DIR_STRUCT.pack(bucket_page_id))
        self.dir_pager.write_page(page)

    def _double_directory(self) -> None:
        old_size = 1 << self.global_depth
        self._dir_grow_to(old_size * 2)
        for j in range(old_size):
            self._dir_set(old_size + j, self._dir_get(j))
        self.global_depth += 1

    # --------------------------------------------------------------- bucket

    def _bucket_for(self, key) -> tuple[int, Page]:
        idx = hash_index(key, self.column, self.global_depth)
        page = self.bkt_pager.read_page(self._dir_get(idx))
        return idx, page

    def _chain(self, bucket: Page):
        page = bucket
        yield page
        while page.next_page_id != NULL_PAGE:
            page = self.bkt_pager.read_page(page.next_page_id)
            yield page

    def _chain_tail(self, bucket: Page) -> Page:
        tail = bucket
        for page in self._chain(bucket):
            tail = page
        return tail

    def _entries(self, bucket: Page) -> list[tuple]:
        out = []
        for page in self._chain(bucket):
            out.extend(_unpack_entry(self.column, page.read_record(s)) for s in range(page.record_count))
        return out

    def _clear(self, bucket: Page) -> None:
        bucket.record_count = 0
        for slot in range(bucket.capacity):
            bucket.set_occupied(slot, False)

    def _append(self, page: Page, key, rid: RID) -> None:
        slot = page.record_count
        page.write_record(slot, _pack_entry(self.column, key, rid))
        page.set_occupied(slot, True)
        page.record_count = slot + 1
        self.bkt_pager.write_page(page)

    def _append_overflow(self, bucket: Page, key, rid: RID) -> None:
        tail = self._chain_tail(bucket)
        if tail.is_full():
            new_page = self.bkt_pager.new_page(PageType.HASH)
            new_page.aux_page_id = tail.aux_page_id
            tail.next_page_id = new_page.page_id
            self.bkt_pager.write_page(tail)
            tail = new_page
        self._append(tail, key, rid)

    def _repack(self, bucket: Page, entries: list[tuple]) -> None:
        pages = list(self._chain(bucket))
        for page in pages:
            self._clear(page)
        i = 0
        for page in pages:
            for k, r in entries[i : i + page.capacity]:
                slot = page.record_count
                page.write_record(slot, _pack_entry(self.column, k, r))
                page.set_occupied(slot, True)
                page.record_count = slot + 1
            self.bkt_pager.write_page(page)
            i += page.capacity

    # ------------------------------------------------------------- consulta

    def search(self, key) -> list[RID]:
        _, bucket = self._bucket_for(key)
        return [rid for k, rid in self._entries(bucket) if k == key]

    # ---------------------------------------------------------------- alta

    def insert(self, key, rid: RID) -> None:
        self._insert(key, rid, 0)

    def _insert(self, key, rid: RID, guard: int) -> None:
        idx, bucket = self._bucket_for(key)
        tail = self._chain_tail(bucket)
        if not tail.is_full():
            self._append(tail, key, rid)
            return
        if bucket.next_page_id == NULL_PAGE and guard < 8 and self._split(bucket, idx):
            self._insert(key, rid, guard + 1)
            return
        self._append_overflow(bucket, key, rid)

    def _split(self, bucket: Page, dir_index: int) -> bool:
        """Divide un bucket sin overflow. Devuelve False si sus claves no se
        pueden separar a esta profundidad (todas caen del mismo lado): en ese
        caso quien llama debe encadenar overflow en vez de seguir dividiendo."""
        local_depth = bucket.aux_page_id
        entries = self._entries(bucket)
        new_depth = local_depth + 1
        sides = [(hash_index(k, self.column, new_depth) >> local_depth) & 1 for k, _ in entries]
        if len(set(sides)) < 2:
            return False

        if local_depth == self.global_depth:
            self._double_directory()

        sibling = self.bkt_pager.new_page(PageType.HASH)
        sibling.aux_page_id = new_depth
        bucket.aux_page_id = new_depth
        self._clear(bucket)
        self.bkt_pager.write_page(bucket)

        for j in range(1 << self.global_depth):
            if self._dir_get(j) == bucket.page_id and (j >> local_depth) & 1:
                self._dir_set(j, sibling.page_id)

        for (k, r), side in zip(entries, sides):
            self._append(sibling if side else bucket, k, r)
        return True

    # --------------------------------------------------------------- baja

    def delete(self, key, rid: RID | None = None) -> int:
        idx, bucket = self._bucket_for(key)
        entries = self._entries(bucket)
        kept = [(k, r) for k, r in entries if not (k == key and (rid is None or r == rid))]
        removed = len(entries) - len(kept)
        if removed == 0:
            return 0

        self._repack(bucket, kept)
        if bucket.next_page_id == NULL_PAGE:
            self._maybe_merge(bucket, idx)
        return removed

    def _maybe_merge(self, bucket: Page, dir_index: int) -> None:
        local_depth = bucket.aux_page_id
        if local_depth == 0:
            return
        buddy_index = dir_index ^ (1 << (local_depth - 1))
        buddy_id = self._dir_get(buddy_index)
        if buddy_id == bucket.page_id:
            return
        buddy = self.bkt_pager.read_page(buddy_id)
        if buddy.aux_page_id != local_depth or buddy.next_page_id != NULL_PAGE:
            return
        if bucket.record_count + buddy.record_count > bucket.capacity:
            return

        entries = self._entries(bucket) + self._entries(buddy)
        bucket.aux_page_id = local_depth - 1
        self._clear(bucket)
        for k, r in entries:
            self._append(bucket, k, r)

        for j in range(1 << self.global_depth):
            if self._dir_get(j) in (bucket.page_id, buddy_id):
                self._dir_set(j, bucket.page_id)

        self.bkt_pager.free_page(buddy_id)

    # ---------------------------------------------------------------- misc

    def flush(self) -> None:
        self.dir_pager.flush()
        self.bkt_pager.flush()

    def close(self) -> None:
        self.dir_pager.close()
        self.bkt_pager.close()

    def __enter__(self) -> "ExtendibleHash":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ExtendibleHash({self.base_path}, D={self.global_depth})"
