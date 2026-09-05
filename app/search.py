"""Parse DeepSeek Web SSE fragments into Anthropic web_search result blocks."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+", re.IGNORECASE)
SEARCH_QUERY_PREFIX = "Perform a web search for the query: "


def extract_search_query(body: Dict[str, Any]) -> str:
    """Pull the last user text out of an Anthropic Messages body."""
    texts: List[str] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    texts.append(part)
    raw = next((item.strip() for item in reversed(texts) if item and item.strip()), "")
    if raw.startswith(SEARCH_QUERY_PREFIX):
        return raw[len(SEARCH_QUERY_PREFIX):].strip()
    return raw


def request_wants_web_search(body: Dict[str, Any]) -> bool:
    """True when the Messages request includes DeepSeek's native search tool."""
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "")
        tool_type = str(tool.get("type") or "")
        if name == "web_search" or tool_type.startswith("web_search"):
            return True
    return False


def apply_sse_event(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Fold one DeepSeek JSON-patch SSE event into reconstructed fragments."""
    if isinstance(event, list):
        for item in event:
            apply_sse_event(state, item)
        return
    if not isinstance(event, dict):
        return

    value = event.get("v")
    path = event.get("p")
    op = event.get("o")

    if op == "BATCH" and isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                nested = item if "p" in item or "v" in item else {"p": path, "o": item.get("o"), "v": item.get("v")}
                apply_sse_event(state, nested)
        return

    if path is None and isinstance(value, dict) and isinstance(value.get("response"), dict):
        _ingest_response(state, value["response"])
        return

    if path is None and isinstance(value, str):
        fragments = state.setdefault("fragments", [])
        if fragments and isinstance(fragments[-1], dict):
            fragments[-1]["content"] = (fragments[-1].get("content") or "") + value
        return

    _apply_path(state, path, op, value)


def collect_search_sources(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """Deduped url/title/snippet records from fragments, extras, and prose."""
    found: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add(url: str, title: str = "", snippet: str = "", page_age: str = "") -> None:
        normalized = _normalize_url(url)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        item: Dict[str, str] = {"url": normalized}
        if title:
            item["title"] = title
        if snippet:
            item["snippet"] = snippet[:500]
        if page_age:
            item["page_age"] = page_age
        found.append(item)

    extras = state.get("extras") or {}
    fragments = state.get("fragments") or []
    _harvest_obj(extras, add)
    for fragment in fragments:
        if isinstance(fragment, dict):
            _harvest_obj(fragment, add)

    if not found:
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            frag_type = str(fragment.get("type") or "")
            if frag_type in ("SEARCH", "RESPONSE", "SEARCH_RESULT"):
                for url in URL_RE.findall(str(fragment.get("content") or "")):
                    add(url)

    return found


def anthropic_search_message(model: str, sources: List[Dict[str, str]], text: str = "") -> Dict[str, Any]:
    """Build the Anthropic Messages envelope DSH's DeepSeek search provider parses."""
    results = []
    citations = []
    for source in sources:
        results.append({
            "type": "web_search_result",
            "url": source["url"],
            **({"title": source["title"]} if source.get("title") else {}),
            **({"page_age": source["page_age"]} if source.get("page_age") else {}),
        })
        if source.get("snippet"):
            citations.append({
                "type": "web_search_result_location",
                "url": source["url"],
                "cited_text": source["snippet"],
            })

    content: List[Dict[str, Any]] = [{
        "type": "web_search_tool_result",
        "content": results,
    }]
    if text or citations:
        content.append({
            "type": "text",
            "text": text or "Search completed.",
            **({"citations": citations} if citations else {}),
        })
    return {
        "id": f"msg_{model}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def fragment_text(state: Dict[str, Any], types: Iterable[str] = ("RESPONSE", "SEARCH")) -> str:
    allowed = set(types)
    parts: List[str] = []
    for fragment in state.get("fragments") or []:
        if isinstance(fragment, dict) and fragment.get("type") in allowed:
            content = fragment.get("content")
            if content:
                parts.append(str(content))
    return "".join(parts)


def _ingest_response(state: Dict[str, Any], response: Dict[str, Any]) -> None:
    fragments = response.get("fragments")
    if isinstance(fragments, list):
        state["fragments"] = [dict(item) if isinstance(item, dict) else item for item in fragments]
    extras = state.setdefault("extras", {})
    for key, value in response.items():
        if key == "fragments":
            continue
        lowered = key.lower()
        if "search" in lowered or "cite" in lowered or key in ("links", "sources", "results"):
            extras[key] = value


def _apply_path(state: Dict[str, Any], path: Optional[str], op: Optional[str], value: Any) -> None:
    if not path:
        return
    local = path[9:] if path.startswith("response/") else path
    fragments: List[Any] = state.setdefault("fragments", [])
    extras: Dict[str, Any] = state.setdefault("extras", {})

    if local == "fragments" and op == "APPEND":
        items = value if isinstance(value, list) else [value]
        for item in items:
            fragments.append(dict(item) if isinstance(item, dict) else item)
        return

    match = re.fullmatch(r"fragments/(-?\d+)(?:/(.+))?", local)
    if match:
        index = int(match.group(1))
        if index == -1:
            index = len(fragments) - 1
        field = match.group(2)
        if 0 <= index < len(fragments):
            if field is None:
                if op == "SET" and isinstance(value, dict):
                    fragments[index] = dict(value)
                elif op == "APPEND" and isinstance(value, dict):
                    current = fragments[index]
                    if isinstance(current, dict):
                        current.update(value)
                    else:
                        fragments[index] = dict(value)
                return
            if isinstance(fragments[index], dict):
                append_text = field == "content" and isinstance(value, str) and op in (None, "APPEND")
                if op == "APPEND" or append_text:
                    if isinstance(value, str):
                        fragments[index][field] = (fragments[index].get(field) or "") + value
                    elif isinstance(value, list):
                        existing = fragments[index].get(field)
                        if isinstance(existing, list):
                            existing.extend(value)
                        else:
                            fragments[index][field] = list(value)
                    else:
                        fragments[index][field] = value
                else:
                    fragments[index][field] = value
        return

    if "search" in local.lower() or "cite" in local.lower():
        extras[local] = value


def _harvest_obj(obj: Any, add, depth: int = 0) -> None:
    if depth > 8:
        return
    if isinstance(obj, list):
        for item in obj:
            _harvest_obj(item, add, depth + 1)
        return
    if not isinstance(obj, dict):
        return

    url = obj.get("url") or obj.get("link") or obj.get("href")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        title = _first_str(obj, ("title", "name", "headline"))
        snippet = _first_str(obj, ("snippet", "cite", "cited_text", "description", "summary"))
        page_age = _first_str(obj, ("page_age", "date", "published")) or _epoch_date(
            obj.get("published_at") or obj.get("publishedAt")
        )
        add(url, title, snippet, page_age)

    for key, value in obj.items():
        if key in ("content",) and isinstance(value, str):
            continue
        _harvest_obj(value, add, depth + 1)


def _first_str(obj: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _epoch_date(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and value > 0:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    return ""


def _normalize_url(url: str) -> str:
    cleaned = url.rstrip(").,;]}>")
    try:
        parsed = urlparse(cleaned)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return cleaned
