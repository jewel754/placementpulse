"""PlacementPulse backend. Run: uvicorn main:app --reload"""
from __future__ import annotations

import csv
import html
import io
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import analyzer  # noqa: E402
import db  # noqa: E402
import scraper  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="PlacementPulse", version="1.0.0", lifespan=lifespan)


# ---------- models ----------
class TextIn(BaseModel):
    text: str
    max_stories: int | None = Field(default=None, ge=1, le=8)


class UrlIn(BaseModel):
    url: str


class FeedItemIn(BaseModel):
    url: str
    title: str = ""
    summary: str = ""
    source: str = ""


class SynthIn(BaseModel):
    days: int = Field(default=7, ge=1, le=60)


# ---------- helpers ----------
def get_cfg(request: Request) -> analyzer.LLMConfig:
    h = request.headers
    return analyzer.LLMConfig.build(
        h.get("x-provider") or None,
        h.get("x-api-key") or None,
        h.get("x-model") or None,
        h.get("x-base-url") or None,
    )


async def run_analysis(
    cfg: analyzer.LLMConfig,
    *,
    text: str,
    source_type: str,
    url: str | None,
    title: str | None,
    key: str,
    max_stories: int,
    label: str,
) -> dict[str, Any]:
    existing = db.find_by_hash(key)
    if existing:
        return {"entries": existing, "created": 0, "duplicate": True}
    try:
        stories = await analyzer.analyze_text(cfg, text, label, max_stories)
    except analyzer.LLMError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    saved = []
    for s in stories:
        eid = db.insert_entry(source_type, url, title, key, text, s)
        saved.append(db.get_entry(eid))
    return {"entries": saved, "created": len(saved), "duplicate": False}


def auto_max_stories(text: str) -> int:
    words = len(text.split())
    if words < 600:
        return 1
    if words < 2000:
        return 3
    return 5


# ---------- pages ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"domains": analyzer.DOMAINS})


@app.get("/api/status")
async def status(request: Request):
    cfg = get_cfg(request)
    return {"configured": bool(cfg.api_key), "provider": cfg.provider, "model": cfg.model, "entries": db.count()}


# ---------- analysis ----------
@app.post("/api/analyze/text")
async def analyze_text(body: TextIn, request: Request):
    text = body.text.strip()
    if len(text) < 200:
        raise HTTPException(400, "Paste at least a few paragraphs (200+ characters) of news text.")
    return await run_analysis(
        get_cfg(request),
        text=text,
        source_type="dump",
        url=None,
        title=None,
        key=db.text_hash(text),
        max_stories=body.max_stories or auto_max_stories(text),
        label="pasted news dump",
    )


@app.post("/api/analyze/url")
async def analyze_url(body: UrlIn, request: Request):
    cfg = get_cfg(request)
    try:
        url = await scraper.asyncio.to_thread(scraper.validate_url, body.url)
    except scraper.ScrapeError as e:
        raise HTTPException(422, str(e))
    key = db.url_hash(url)
    existing = db.find_by_hash(key)
    if existing:
        return {"entries": existing, "created": 0, "duplicate": True}
    try:
        art = await scraper.scrape(url)
    except scraper.ScrapeError as e:
        raise HTTPException(422, str(e))
    return await run_analysis(
        cfg,
        text=art["text"],
        source_type="url",
        url=art["url"],
        title=art["title"],
        key=key,
        max_stories=1,
        label=f"article: {art['title'] or art['url']}",
    )


@app.post("/api/analyze/feed-item")
async def analyze_feed_item(body: FeedItemIn, request: Request):
    cfg = get_cfg(request)
    url = body.url.strip()
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(422, "Invalid article URL.")
    key = db.url_hash(url)
    existing = db.find_by_hash(key)
    if existing:
        return {"entries": existing, "created": 0, "duplicate": True}
    title = body.title.strip()
    try:
        art = await scraper.scrape(url)
        text, title = art["text"], art["title"] or title
    except scraper.ScrapeError as e:
        fallback = f"{body.title}\n\n{body.summary}".strip()
        if len(fallback) < 120:
            raise HTTPException(422, f"{e} The RSS summary was too short to analyse instead.")
        text = fallback + "\n\n[Note: only the headline and RSS summary were available; the article body could not be retrieved.]"
    return await run_analysis(
        cfg,
        text=text,
        source_type="rss",
        url=url,
        title=title,
        key=key,
        max_stories=1,
        label=f"RSS article: {title}" + (f" ({body.source})" if body.source else ""),
    )


# ---------- feeds ----------
@app.get("/api/feeds")
async def feeds():
    return {"feeds": scraper.FEEDS}


@app.get("/api/feed")
async def feed(url: str, limit: int = Query(15, ge=1, le=50)):
    try:
        items = await scraper.fetch_feed(url, limit)
    except scraper.ScrapeError as e:
        raise HTTPException(422, str(e))
    for it in items:
        it["analyzed"] = db.hash_exists(db.url_hash(it["link"]))
    return {"items": items}


# ---------- entries ----------
@app.get("/api/entries")
def entries(q: str = "", domain: str = "", days: int = Query(0, ge=0, le=3650), limit: int = Query(1000, ge=1, le=5000)):
    return {"entries": db.list_entries(q, domain, days, limit)}


@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int):
    if not db.delete_entry(entry_id):
        raise HTTPException(404, "Entry not found.")
    return {"deleted": entry_id}


# ---------- synthesis ----------
@app.post("/api/synthesis")
async def synthesis(body: SynthIn, request: Request):
    items = db.list_entries(days=body.days, limit=200)
    if not items:
        raise HTTPException(404, f"No cards from the last {body.days} days. Analyse some news first.")
    try:
        result = await analyzer.synthesize(get_cfg(request), items, body.days)
    except analyzer.LLMError as e:
        raise HTTPException(e.status_code, e.message)
    return {"synthesis": result, "count": len(items), "days": body.days}


# ---------- export ----------
def _anki_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    esc = html.escape
    rows = []
    for e in items:
        d = e["data"]
        steps = [s.strip() for s in re.split(r"\s*->\s*", d["causal_mechanism"]) if s.strip()]
        chain = "<ol>" + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol>"
        imp = d["finance_and_business_impact"]
        front = (
            f"<b>{esc(d['headline'])}</b><br><br>"
            "Q: Explain the causal mechanism step by step, the finance impact, and the one number you would cite in a GD."
        )
        back = (
            f"<b>Causal chain</b>{chain}"
            f"<b>GD power stat</b><br>{esc(d['gd_power_stat'])}<br><br>"
            f"<b>Finance angle</b><br>Corporate finance and valuation: {esc(imp['corporate_finance_valuation'])}<br>"
            f"Markets and margins: {esc(imp['markets_margins'])}<br><br>"
            f"<b>Memory anchor</b><br>{esc(d['memory_anchor'])}"
        )
        tags = " ".join(
            re.sub(r"[^\w]+", "_", t).strip("_") for t in [d["domain"], *d["secondary_tags"]] if t
        )
        rows.append({"id": str(e["id"]), "front": front, "back": back, "tags": tags})
    return rows


@app.get("/api/export")
def export(
    kind: str = Query("anki_csv", pattern="^(anki_csv|anki_json|backup)$"),
    q: str = "",
    domain: str = "",
    days: int = Query(0, ge=0, le=3650),
):
    items = db.list_entries(q, domain, days, 5000)
    if not items:
        raise HTTPException(404, "Nothing to export for the current filters.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    if kind == "anki_csv":
        rows = _anki_rows(items)
        buf = io.StringIO()
        buf.write("#separator:Comma\n#html:true\n#tags column:3\n")
        w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
        for r in rows:
            w.writerow([r["front"], r["back"], r["tags"]])
        return Response(
            buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="placementpulse_anki_{stamp}.csv"'},
        )
    if kind == "anki_json":
        payload = json.dumps(_anki_rows(items), indent=2, ensure_ascii=False)
        name = f"placementpulse_anki_{stamp}.json"
    else:
        payload = json.dumps(items, indent=2, ensure_ascii=False)
        name = f"placementpulse_backup_{stamp}.json"
    return Response(
        payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
