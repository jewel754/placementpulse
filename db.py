"""Database persistence for PlacementPulse (PostgreSQL + SQLite support)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urldefrag

# Detect if we are using Neon Postgres or local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "")
IS_POSTGRES = DATABASE_URL.startswith("postgres")

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import DictCursor
else:
    import sqlite3

DB_PATH = os.getenv("DB_PATH", "placementpulse.db")

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id SERIAL PRIMARY KEY,
    created_at TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT,
    source_title TEXT,
    content_hash TEXT NOT NULL,
    raw_text TEXT,
    headline TEXT NOT NULL,
    domain TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    data_json TEXT NOT NULL,
    UNIQUE(content_hash, headline)
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_domain ON entries(domain);
CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(content_hash);
"""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT,
    source_title TEXT,
    content_hash TEXT NOT NULL,
    raw_text TEXT,
    headline TEXT NOT NULL,
    domain TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    data_json TEXT NOT NULL,
    UNIQUE(content_hash, headline)
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_domain ON entries(domain);
CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(content_hash);
"""

@contextmanager
def conn():
    if IS_POSTGRES:
        c = psycopg2.connect(DATABASE_URL)
        try:
            yield c
            c.commit()
        finally:
            c.close()
    else:
        c = sqlite3.connect(DB_PATH, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

def init_db() -> None:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor() as cur:
                cur.execute(PG_SCHEMA)
        else:
            c.executescript(SQLITE_SCHEMA)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def text_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())[:20000]
    return "t:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()

def url_hash(url: str) -> str:
    u = urldefrag(url.strip())[0].rstrip("/").lower()
    return "u:" + hashlib.sha1(u.encode("utf-8")).hexdigest()

def _row(r: Any) -> dict[str, Any]:
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "source_type": r["source_type"],
        "source_url": r["source_url"],
        "source_title": r["source_title"],
        "data": json.loads(r["data_json"]),
    }

def insert_entry(
    source_type: str,
    source_url: str | None,
    source_title: str | None,
    content_hash: str,
    raw_text: str,
    story: dict[str, Any],
) -> int:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """INSERT INTO entries 
                       (created_at, source_type, source_url, source_title, content_hash, raw_text, 
                        headline, domain, tags, data_json)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (content_hash, headline) DO NOTHING""",
                    (
                        now_iso(), source_type, source_url, source_title, content_hash, raw_text[:200000],
                        story["headline"], story["domain"], json.dumps(story.get("secondary_tags", [])), json.dumps(story)
                    )
                )
                cur.execute(
                    "SELECT id FROM entries WHERE content_hash=%s AND headline=%s",
                    (content_hash, story["headline"])
                )
                return int(cur.fetchone()["id"])
        else:
            cur = c.execute(
                """INSERT OR IGNORE INTO entries 
                   (created_at, source_type, source_url, source_title, content_hash, raw_text, 
                    headline, domain, tags, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    now_iso(), source_type, source_url, source_title, content_hash, raw_text[:200000],
                    story["headline"], story["domain"], json.dumps(story.get("secondary_tags", [])), json.dumps(story)
                )
            )
            r = c.execute(
                "SELECT id FROM entries WHERE content_hash=? AND headline=?",
                (content_hash, story["headline"])
            ).fetchone()
            return int(r["id"])

def get_entry(entry_id: int) -> dict[str, Any] | None:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT * FROM entries WHERE id=%s", (entry_id,))
                r = cur.fetchone()
        else:
            r = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return _row(r) if r else None

def find_by_hash(content_hash: str) -> list[dict[str, Any]]:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT * FROM entries WHERE content_hash=%s ORDER BY id", (content_hash,))
                rows = cur.fetchall()
        else:
            rows = c.execute("SELECT * FROM entries WHERE content_hash=? ORDER BY id", (content_hash,)).fetchall()
        return [_row(r) for r in rows]

def hash_exists(content_hash: str) -> bool:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor() as cur:
                cur.execute("SELECT 1 FROM entries WHERE content_hash=%s LIMIT 1", (content_hash,))
                return cur.fetchone() is not None
        else:
            return c.execute("SELECT 1 FROM entries WHERE content_hash=? LIMIT 1", (content_hash,)).fetchone() is not None

def list_entries(q: str = "", domain: str = "", days: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
    sql = "SELECT * FROM entries WHERE 1=1"
    params: list[Any] = []
    placeholder = "%s" if IS_POSTGRES else "?"
    
    if domain and domain != "All":
        sql += f" AND domain={placeholder}"
        params.append(domain)
    if days and days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        sql += f" AND created_at>={placeholder}"
        params.append(cutoff)
        
    for term in [t for t in re.split(r"\s+", q.strip()) if t]:
        like = f"%{term}%"
        sql += f" AND (headline LIKE {placeholder} OR tags LIKE {placeholder} OR data_json LIKE {placeholder})"
        params.extend([like, like, like])
        
    sql += f" ORDER BY created_at DESC, id DESC LIMIT {placeholder}"
    params.append(max(1, min(limit, 5000)))
    
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, params)
                return [_row(r) for r in cur.fetchall()]
        else:
            return [_row(r) for r in c.execute(sql, params).fetchall()]

def delete_entry(entry_id: int) -> bool:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor() as cur:
                cur.execute("DELETE FROM entries WHERE id=%s", (entry_id,))
                return cur.rowcount > 0
        else:
            cur = c.execute("DELETE FROM entries WHERE id=?", (entry_id,))
            return cur.rowcount > 0

def count() -> int:
    with conn() as c:
        if IS_POSTGRES:
            with c.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("SELECT COUNT(*) AS n FROM entries")
                return int(cur.fetchone()["n"])
        else:
            return int(c.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"])