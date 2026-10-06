"""First-principles reasoning engine: LLM calls, strict schema validation, weekly synthesis."""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError, field_validator

DOMAINS = [
    "Finance",
    "Macroeconomy",
    "Geopolitics",
    "Corporate Strategy/M&A",
    "Tech & Regulation",
    "Supply Chain",
]

MAX_CHARS = 60000

_DOMAIN_HINTS = [
    ("supply", "Supply Chain"),
    ("logistic", "Supply Chain"),
    ("m&a", "Corporate Strategy/M&A"),
    ("merger", "Corporate Strategy/M&A"),
    ("acqui", "Corporate Strategy/M&A"),
    ("strategy", "Corporate Strategy/M&A"),
    ("corporate", "Corporate Strategy/M&A"),
    ("tech", "Tech & Regulation"),
    ("regulat", "Tech & Regulation"),
    ("policy", "Tech & Regulation"),
    ("geopol", "Geopolitics"),
    ("sanction", "Geopolitics"),
    ("war", "Geopolitics"),
    ("trade", "Geopolitics"),
    ("macro", "Macroeconomy"),
    ("econom", "Macroeconomy"),
    ("inflation", "Macroeconomy"),
    ("monetary", "Macroeconomy"),
    ("financ", "Finance"),
    ("market", "Finance"),
    ("bank", "Finance"),
    ("equity", "Finance"),
]


def normalize_domain(value: Any) -> str:
    s = str(value or "").strip()
    for d in DOMAINS:
        if s.lower() == d.lower():
            return d
    low = s.lower()
    for key, dom in _DOMAIN_HINTS:
        if key in low:
            return dom
    return "Finance"


def _s(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_s(x) for x in v if _s(x))
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_s(x)}" for k, x in v.items() if _s(x))
    return str(v).strip()


def _non_empty(v: Any) -> str:
    s = _s(v)
    if not s:
        raise ValueError("must not be empty")
    return s


class Story(BaseModel):
    headline: str
    domain: str
    secondary_tags: list[str]
    core_trigger: str
    causal_mechanism: str
    finance_and_business_impact: dict[str, str]
    geopolitical_and_policy_linkage: str
    gd_power_stat: str
    gd_discussion_hook: str
    memory_anchor: str

    @field_validator(
        "headline",
        "core_trigger",
        "geopolitical_and_policy_linkage",
        "gd_power_stat",
        "gd_discussion_hook",
        "memory_anchor",
        mode="before",
    )
    @classmethod
    def _req_str(cls, v: Any) -> str:
        return _non_empty(v)

    @field_validator("domain", mode="before")
    @classmethod
    def _domain(cls, v: Any) -> str:
        return normalize_domain(v)

    @field_validator("secondary_tags", mode="before")
    @classmethod
    def _tags(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = re.split(r"[,;|]", v)
        if not isinstance(v, (list, tuple)):
            raise ValueError("secondary_tags must be a list")
        out: list[str] = []
        for t in v:
            t = _s(t)
            if t and t.lower() not in [x.lower() for x in out]:
                out.append(t)
        if not out:
            raise ValueError("secondary_tags must not be empty")
        return out[:8]

    @field_validator("causal_mechanism", mode="before")
    @classmethod
    def _chain(cls, v: Any) -> str:
        if isinstance(v, (list, tuple)):
            steps = [re.sub(r"^\s*(?:step\s*)?\d+[.):]\s*", "", _s(x)) for x in v if _s(x)]
            v = " -> ".join(steps)
        v = _s(v)
        if "->" not in v and "→" not in v and "\n" in v:
            steps = [re.sub(r"^\s*(?:step\s*)?\d+[.):]\s*", "", ln).strip() for ln in v.splitlines()]
            v = " -> ".join(s for s in steps if s)
        v = v.replace("→", "->")
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("finance_and_business_impact", mode="before")
    @classmethod
    def _impact(cls, v: Any) -> dict[str, str]:
        corp = mkt = ""
        if isinstance(v, dict):
            for k, val in v.items():
                kl = str(k).lower()
                if any(x in kl for x in ("corp", "valuation", "capital", "dcf")):
                    corp = (corp + " " + _s(val)).strip()
                elif any(x in kl for x in ("market", "margin", "p&l", "sector")):
                    mkt = (mkt + " " + _s(val)).strip()
                else:
                    mkt = (mkt + " " + _s(val)).strip()
            if not corp and mkt:
                corp = mkt
            if not mkt and corp:
                mkt = corp
        else:
            corp = mkt = _s(v)
        if not corp or not mkt:
            raise ValueError("finance_and_business_impact needs both parts")
        return {"corporate_finance_valuation": corp, "markets_margins": mkt}


SYSTEM_PROMPT = """You are PlacementPulse, a first-principles analyst coaching MBA candidates for finance, consulting and management interviews and group discussions (GDs).

You turn news into mental models. You REJECT superficial one-line summaries. Every output must teach the underlying mechanism: who is incentivised to do what, which variable moves, how that travels through cost of capital, cash flows, margins, prices, currencies and policy, and who wins or loses.

QUALITY RULES
1. Ground every claim in the supplied text. Where you extend beyond it, hedge ("likely", "would typically"). Never fabricate facts, quotes, figures or sources.
2. causal_mechanism: 4 to 7 sequential steps joined by " -> ". Each step is a full causal link with direction and, where possible, magnitude (e.g. "RBI cuts repo 25 bps -> banks' marginal cost of funds falls -> lending rates reprice with a lag -> ..."). No vague steps like "markets react".
3. finance_and_business_impact.corporate_finance_valuation: cost of capital (WACC, cost of debt/equity), DCF cash flows, terminal value, multiples, capital structure. finance_and_business_impact.markets_margins: P&L and margin impact, currency, bond yields, equity sectors, named winners and losers.
4. geopolitical_and_policy_linkage: connect to global trade, central-bank mandates, regulation or great-power dynamics. If the link is thin, say what the nearest real structural link is.
5. gd_power_stat: EXACTLY ONE number or metric with a short unit/context. Use a figure present in the source text whenever one exists. Only if none exists, use a widely documented benchmark and end it with "(approx., verify before use)". Never invent.
6. gd_discussion_hook: 1 to 2 sentences, contrarian or structural framing, e.g. "While the room is debating X, the real structural risk is Y because...".
7. memory_anchor: a vivid everyday analogy (1 to 2 sentences) that makes the mechanism impossible to forget under pressure.
8. core_trigger: EXACTLY 2 sentences on what happened.
9. secondary_tags: 3 to 6 specific concepts (e.g. "WACC", "Yield Curve", "OPEC+", "Term Premium").
10. domain must be exactly one of: __DOMAINS__.
11. Input may be a single article or a messy dump of several stories mixed with ads, navigation text and boilerplate. Ignore boilerplate. Identify distinct stories, merge duplicates, and return at most MAX_STORIES stories, choosing the highest-signal ones for interviews. A single article yields exactly one story. If the text says only a headline and RSS summary were available, keep claims modest and hedged.

OUTPUT: Return ONLY one JSON object, no markdown, no commentary:
{"stories":[{"headline":"string","domain":"string","secondary_tags":["string"],"core_trigger":"string","causal_mechanism":"step -> step -> step","finance_and_business_impact":{"corporate_finance_valuation":"string","markets_margins":"string"},"geopolitical_and_policy_linkage":"string","gd_power_stat":"string","gd_discussion_hook":"string","memory_anchor":"string"}]}
""".replace("__DOMAINS__", " | ".join(DOMAINS))


SYNTH_PROMPT = """You are PlacementPulse's chief of staff for an MBA candidate preparing for interviews and GDs. You receive the week's analysed news cards. Produce a cohesive macro-economic debrief that stitches them together: do not list the cards one by one, find the connective tissue (shared drivers, second-order effects, contradictions) across domains.

Ground everything in the supplied cards. Never invent figures. Return ONLY one JSON object:
{
 "title": "string",
 "executive_summary": "150 to 220 words, a connected narrative of the week",
 "themes": [{"theme":"string","insight":"string","linked_headlines":["exact headlines from the cards"]}],
 "cross_domain_linkages": [{"chain":"A -> B -> C","domains":["string"],"explanation":"string"}],
 "power_stats": [{"stat":"string","use_case":"how to deploy it in a GD"}],
 "likely_interview_questions": ["string"],
 "watchlist_next_week": ["string"]
}
Give 3 to 5 themes, 3 to 5 cross-domain linkages, 4 to 6 power stats taken from the cards, 5 to 7 questions and 3 to 5 watchlist items."""


class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code if 400 <= status_code <= 599 else 502


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str

    @classmethod
    def build(
        cls,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> "LLMConfig":
        provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
        if provider not in ("gemini", "openai"):
            provider = "gemini"
        if provider == "gemini":
            key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            mdl = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            base = ""
        else:
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            base = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        return cls(provider=provider, api_key=key.strip(), model=mdl.strip(), base_url=base)


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err)
    except Exception:
        return resp.text[:300] or f"HTTP {resp.status_code}"


async def call_llm(cfg: LLMConfig, system: str, user: str, temperature: float = 0.3) -> str:
    if not cfg.api_key:
        raise LLMError(
            "No API key configured. Open Settings and add a Gemini or OpenAI key (or set it in .env).", 400
        )

    if cfg.provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent"
        headers = {"x-goog-api-key": cfg.api_key, "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": 16384,
            },
        }
    else:
        url = f"{cfg.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        body = {
            "model": cfg.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    attempts = 4
    last_status = 502
    last_msg = "Unknown error"
    async with httpx.AsyncClient(timeout=httpx.Timeout(150.0, connect=15.0)) as client:
        for i in range(attempts):
            try:
                resp = await client.post(url, headers=headers, json=body)
            except httpx.TimeoutException:
                last_status, last_msg = 504, "The AI provider timed out."
            except httpx.HTTPError as e:
                last_status, last_msg = 502, f"Network error contacting the AI provider ({e.__class__.__name__})."
            else:
                if resp.status_code == 200:
                    return _extract_text(cfg, resp.json())
                last_status, last_msg = resp.status_code, _error_message(resp)
                if resp.status_code not in (429, 500, 502, 503, 504):
                    if resp.status_code in (401, 403):
                        raise LLMError(f"API key rejected by {cfg.provider}: {last_msg}", 401)
                    if resp.status_code == 404:
                        raise LLMError(f"Model '{cfg.model}' not found for {cfg.provider}: {last_msg}", 400)
                    raise LLMError(f"{cfg.provider} error: {last_msg}", resp.status_code)
                retry_after = resp.headers.get("retry-after", "")
                if i < attempts - 1:
                    delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2 ** (i + 1)
                    await asyncio.sleep(min(delay, 20))
                    continue
            if i < attempts - 1:
                await asyncio.sleep(2 ** (i + 1))

    if last_status == 429:
        raise LLMError(
            "Rate limit reached on the AI provider. Wait a minute, switch model, or use a key with higher quota.", 429
        )
    raise LLMError(f"AI provider failed after {attempts} attempts: {last_msg}", last_status)


def _extract_text(cfg: LLMConfig, data: dict[str, Any]) -> str:
    if cfg.provider == "gemini":
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "empty response")
            raise LLMError(f"Gemini returned no content ({reason}).", 502)
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned empty text (finishReason: {cands[0].get('finishReason')}).", 502)
        return text
    choices = data.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
    if not text.strip():
        raise LLMError("The model returned an empty response.", 502)
    return text


def parse_json(text: str) -> Any:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a : b + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Response was not valid JSON.")


def _validate_stories(obj: Any, max_stories: int) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(obj, dict) and isinstance(obj.get("stories"), list):
        items = obj["stories"]
    elif isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and "headline" in obj:
        items = [obj]
    else:
        return [], ["No 'stories' array found."]
    good: list[dict[str, Any]] = []
    errors: list[str] = []
    for it in items:
        try:
            good.append(Story.model_validate(it).model_dump())
        except ValidationError as e:
            errors.append("; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:3]))
    return good[:max_stories], errors


async def analyze_text(
    cfg: LLMConfig, text: str, source_label: str = "pasted text", max_stories: int = 1
) -> list[dict[str, Any]]:
    text = text.strip()
    clipped = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    user = (
        f"SOURCE: {source_label}\nMAX_STORIES: {max_stories}\n"
        + ("NOTE: input was truncated for length.\n" if clipped else "")
        + f'\nARTICLE TEXT:\n"""\n{text}\n"""'
    )
    last_err = "unknown"
    for attempt in range(2):
        prompt = user
        if attempt == 1:
            prompt += (
                "\n\nYour previous reply failed validation. Reply with ONLY the JSON object matching the schema; "
                "every field must be present and non-empty."
            )
        raw = await call_llm(cfg, SYSTEM_PROMPT, prompt)
        try:
            obj = parse_json(raw)
        except ValueError as e:
            last_err = str(e)
            continue
        stories, errors = _validate_stories(obj, max_stories)
        if stories:
            return stories
        last_err = "; ".join(errors) or "no valid stories"
    raise LLMError(f"The model's output did not match the required schema ({last_err}). Try again.", 502)


async def synthesize(cfg: LLMConfig, entries: list[dict[str, Any]], days: int) -> dict[str, Any]:
    lines = []
    for e in entries[:60]:
        d = e["data"]
        lines.append(
            f"[{e['created_at'][:10]}] ({d['domain']}) {d['headline']}\n"
            f"  Trigger: {d['core_trigger']}\n  Chain: {d['causal_mechanism']}\n"
            f"  Power stat: {d['gd_power_stat']}\n  Tags: {', '.join(d['secondary_tags'])}"
        )
    user = f"WINDOW: last {days} days, {len(lines)} cards.\n\n" + "\n\n".join(lines)
    last_err = "unknown"
    for attempt in range(2):
        raw = await call_llm(cfg, SYNTH_PROMPT, user + ("\n\nReply with ONLY the JSON object." if attempt else ""), 0.4)
        try:
            obj = parse_json(raw)
        except ValueError as e:
            last_err = str(e)
            continue
        if not isinstance(obj, dict) or not _s(obj.get("executive_summary")):
            last_err = "missing executive_summary"
            continue
        return {
            "title": _s(obj.get("title")) or f"Weekly macro debrief ({days} days)",
            "executive_summary": _s(obj.get("executive_summary")),
            "themes": [t for t in (obj.get("themes") or []) if isinstance(t, dict)],
            "cross_domain_linkages": [t for t in (obj.get("cross_domain_linkages") or []) if isinstance(t, dict)],
            "power_stats": [t if isinstance(t, dict) else {"stat": _s(t), "use_case": ""} for t in (obj.get("power_stats") or [])],
            "likely_interview_questions": [_s(x) for x in (obj.get("likely_interview_questions") or []) if _s(x)],
            "watchlist_next_week": [_s(x) for x in (obj.get("watchlist_next_week") or []) if _s(x)],
        }
    raise LLMError(f"Synthesis output was invalid ({last_err}). Try again.", 502)
