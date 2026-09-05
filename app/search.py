"""Local web search for the DeepSeek Harness Anthropic Messages shim."""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx

SEARCH_QUERY_PREFIX = "Perform a web search for the query: "
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


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
    """True when the Messages request includes the harness web_search tool."""
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "")
        tool_type = str(tool.get("type") or "")
        if name == "web_search" or tool_type.startswith("web_search"):
            return True
    return False


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


async def local_web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """Search the web from this host. Does not call DeepSeek."""
    q = query.strip()
    if not q:
        return []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        sources = await _search_duckduckgo(client, q)
        if not sources:
            sources = await _search_bing(client, q)
    return sources[:max_results]


async def _search_duckduckgo(client: httpx.AsyncClient, query: str) -> List[Dict[str, str]]:
    response = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"Referer": "https://html.duckduckgo.com/html/"},
    )
    response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    return parser.results


async def _search_bing(client: httpx.AsyncClient, query: str) -> List[Dict[str, str]]:
    response = await client.get("https://www.bing.com/search", params={"q": query})
    response.raise_for_status()
    parser = _BingParser()
    parser.feed(response.text)
    return parser.results


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._mode = ""
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = _class_attr(attrs)
        if tag == "a" and "result__a" in cls:
            href = _attr(attrs, "href")
            url = _unwrap_ddg_url(href)
            if url:
                self.results.append({"url": url, "title": "", "snippet": ""})
                self._mode = "title"
                self._buf = []
        elif tag == "a" and "result__snippet" in cls and self.results:
            self._mode = "snippet"
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._mode or not self.results:
            return
        text = _clean_text("".join(self._buf))
        if self._mode == "title":
            self.results[-1]["title"] = text
        elif self._mode == "snippet":
            self.results[-1]["snippet"] = text[:500]
        self._mode = ""

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._in_algo = False
        self._mode = ""
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = _class_attr(attrs)
        if tag == "li" and "b_algo" in cls:
            self._in_algo = True
        elif self._in_algo and tag == "h2":
            self._mode = "await_title"
        elif self._in_algo and tag == "a" and self._mode == "await_title":
            href = _normalize_http_url(_attr(attrs, "href"))
            if href:
                self.results.append({"url": href, "title": "", "snippet": ""})
                self._mode = "title"
                self._buf = []
        elif self._in_algo and tag == "p" and self.results and not self.results[-1].get("snippet"):
            self._mode = "snippet"
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "li":
            self._in_algo = False
            self._mode = ""
            return
        if not self.results:
            return
        text = _clean_text("".join(self._buf))
        if tag == "a" and self._mode == "title":
            self.results[-1]["title"] = text
            self._mode = ""
            self._buf = []
        elif tag == "p" and self._mode == "snippet":
            self.results[-1]["snippet"] = text[:500]
            self._mode = ""
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode in ("title", "snippet"):
            self._buf.append(data)


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    for key, value in attrs:
        if key == name and value:
            return value
    return ""


def _class_attr(attrs: list[tuple[str, str | None]]) -> str:
    return _attr(attrs, "class")


def _unwrap_ddg_url(href: str) -> str:
    if not href:
        return ""
    absolute = urljoin("https://html.duckduckgo.com", href)
    parsed = urlparse(absolute)
    if parsed.path == "/l/" or parsed.netloc.endswith("duckduckgo.com") and "uddg" in parsed.query:
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        if uddg:
            return _normalize_http_url(unquote(uddg))
    return _normalize_http_url(absolute)


def _normalize_http_url(url: str) -> str:
    cleaned = html.unescape((url or "").strip()).rstrip(").,;]}>")
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    if "duckduckgo.com" in host or "bing.com" in host or "microsoft.com" in host:
        return ""
    return cleaned


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()
