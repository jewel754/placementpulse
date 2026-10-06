"""Article scraping, boilerplate cleaning and RSS ingestion."""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urldefrag, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup


class ScrapeError(Exception):
    pass


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_BYTES = 4_000_000

FEEDS: list[dict[str, Any]] = [
    {"id": "et-economy", "name": "Economic Times · Economy", "category": "Macroeconomy",
     "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms", "paywalled": False},
    {"id": "et-markets", "name": "Economic Times · Markets", "category": "Finance",
     "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "paywalled": False},
    {"id": "et-industry", "name": "Economic Times · Industry", "category": "Corporate Strategy/M&A",
     "url": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms", "paywalled": False},
    {"id": "mint-markets", "name": "Mint · Markets", "category": "Finance",
     "url": "https://www.livemint.com/rss/markets", "paywalled": False},
    {"id": "mint-economy", "name": "Mint · Economy", "category": "Macroeconomy",
     "url": "https://www.livemint.com/rss/economy", "paywalled": False},
    {"id": "mint-companies", "name": "Mint · Companies", "category": "Corporate Strategy/M&A",
     "url": "https://www.livemint.com/rss/companies", "paywalled": False},
    {"id": "bl-economy", "name": "BusinessLine · Economy", "category": "Macroeconomy",
     "url": "https://www.thehindubusinessline.com/economy/feeder/default.rss", "paywalled": False},
    {"id": "ft-world", "name": "Financial Times · World", "category": "Geopolitics",
     "url": "https://www.ft.com/world?format=rss", "paywalled": True},
    {"id": "ft-economy", "name": "Financial Times · Global Economy", "category": "Macroeconomy",
     "url": "https://www.ft.com/global-economy?format=rss", "paywalled": True},
    {"id": "ft-companies", "name": "Financial Times · Companies", "category": "Corporate Strategy/M&A",
     "url": "https://www.ft.com/companies?format=rss", "paywalled": True},
    {"id": "bbc-business", "name": "BBC · Business", "category": "Finance",
     "url": "https://feeds.bbci.co.uk/news/business/rss.xml", "paywalled": False},
]

_BOILER = re.compile(
    r"(subscribe to|sign in to|log in to|advertisement|cookie|newsletter|follow us|download the app|"
    r"read more|also read|click here|all rights reserved|privacy policy|already a subscriber|"
    r"share this|join our|continue reading|to read the full|terms of use|terms and conditions|"
    r"whatsapp|telegram channel|add as preferred source)",
    re.I,
)


def validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ScrapeError("URL is empty.")
    if not re.match(r"^https?://", url, re.I):
        if " " not in url and "." in url:
            url = "https://" + url
        else:
            raise ScrapeError("Enter a full URL starting with http:// or https://.")
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ScrapeError("Invalid URL.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ScrapeError(f"Could not resolve host '{p.hostname}'. Check the URL.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ScrapeError("Refusing to fetch private or internal network addresses.")
    return urldefrag(url)[0]


async def _get(url: str, headers: dict[str, str] | None = None) -> httpx.Response:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers=headers or HEADERS,
        ) as c:
            r = await c.get(url)
    except httpx.TimeoutException:
        raise ScrapeError("The site took too long to respond (timeout).")
    except httpx.TooManyRedirects:
        raise ScrapeError("Too many redirects.")
    except httpx.HTTPError as e:
        raise ScrapeError(f"Network error while fetching the page ({e.__class__.__name__}).")
    if r.status_code in (401, 402, 403, 451):
        raise ScrapeError(
            f"The site blocked access (HTTP {r.status_code}), likely a paywall or bot protection. "
            "Paste the article text into Brain Dump instead."
        )
    if r.status_code == 404:
        raise ScrapeError("Page not found (HTTP 404). Check the URL.")
    if r.status_code == 429:
        raise ScrapeError("The site is rate-limiting requests (HTTP 429). Try again shortly.")
    if r.status_code >= 400:
        raise ScrapeError(f"The site returned HTTP {r.status_code}.")
    if len(r.content) > MAX_BYTES:
        raise ScrapeError("Page is too large to process.")
    return r


def clean_text(text: str) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()
        if not ln:
            continue
        if len(ln) < 140 and _BOILER.search(ln):
            continue
        if len(ln) < 25 and not ln.endswith((".", "!", "?", '"', "”", ":")):
            continue
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return "\n".join(out)


def _extract_sync(html: str, url: str) -> tuple[str, str]:
    title, text = "", ""
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    title = re.split(r"\s+[|•]\s+", title)[0].strip()

    try:
        import trafilatura

        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_precision=True
        ) or ""
    except Exception:
        text = ""

    if len(text) < 400:
        try:
            from newspaper import Article

            a = Article(url)
            a.download(input_html=html)
            a.parse()
            if len(a.text or "") > len(text):
                text = a.text
        except Exception:
            pass

    if len(text) < 400:
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "iframe", "svg"]):
            tag.decompose()
        container = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        fallback = "\n".join(p for p in paras if p)
        if len(fallback) > len(text):
            text = fallback

    return title, clean_text(text)


async def scrape(url: str) -> dict[str, str]:
    url = await asyncio.to_thread(validate_url, url)
    r = await _get(url)
    ctype = r.headers.get("content-type", "").lower()
    if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
        raise ScrapeError(f"URL does not point to a web page (content-type: {ctype.split(';')[0]}).")
    title, text = await asyncio.to_thread(_extract_sync, r.text, url)
    if len(text) < 300:
        raise ScrapeError(
            f"Only {len(text)} characters of article text could be extracted. The page is likely paywalled "
            "or rendered with JavaScript. Paste the text into Brain Dump instead."
        )
    return {"url": url, "title": title, "text": text}


def _strip_html(s: str) -> str:
    return BeautifulSoup(s or "", "lxml").get_text(" ", strip=True)


async def fetch_feed(url: str, limit: int = 15) -> list[dict[str, Any]]:
    url = await asyncio.to_thread(validate_url, url)
    r = await _get(url, {**HEADERS, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"})
    parsed = await asyncio.to_thread(feedparser.parse, r.content)
    if not parsed.entries:
        raise ScrapeError("The feed returned no entries (invalid RSS, empty, or blocked).")
    items: list[dict[str, Any]] = []
    for e in parsed.entries[: max(1, min(limit, 50))]:
        link = (e.get("link") or "").strip()
        title = _strip_html(e.get("title") or "").strip()
        if not link or not title:
            continue
        published = ""
        t = e.get("published_parsed") or e.get("updated_parsed")
        if t:
            try:
                published = datetime(*t[:6], tzinfo=timezone.utc).isoformat(timespec="seconds")
            except Exception:
                published = ""
        items.append(
            {
                "title": title,
                "link": link,
                "summary": _strip_html(e.get("summary") or e.get("description") or "")[:700],
                "published": published,
            }
        )
    items.sort(key=lambda i: i["published"], reverse=True)
    return items
