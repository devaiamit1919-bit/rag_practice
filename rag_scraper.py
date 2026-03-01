"""Scraping/sitemap stage for the RAG playground."""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rag_config import SourceConfig, RunPaths, append_jsonl, log_message, now_iso, sha1_hex


@dataclass
class ScrapedPage:
    url: str
    final_url: str
    status_code: int
    title: str
    text: str
    raw_file: Path
    error: str = ""


def load_processed_urls(ledger_path: Path) -> Dict[str, str]:
    processed: Dict[str, str] = {}
    if not ledger_path.exists():
        return processed

    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            url = row.get("url")
            status = row.get("status", "")
            if url:
                processed[url] = status
        except json.JSONDecodeError:
            continue
    return processed


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: List[str] = []
        self.skip_depth = 0
        self.heading_level = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: ANN001
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "header", "footer", "nav"}:
            self.skip_depth += 1
            return

        if self.skip_depth > 0:
            return

        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.heading_level = int(tag[1])
        elif tag in {"p", "li", "div", "section", "article", "br"}:
            self.parts.append("\\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "header", "footer", "nav"}:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth > 0:
            return
        if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.heading_level = 0

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self.heading_level > 0:
            self.parts.append(f"\\n{'#' * self.heading_level} {text}")
        else:
            self.parts.append(f" {text}")

    def extracted_text(self) -> str:
        full = "".join(self.parts)
        full = re.sub(r"\\n{3,}", "\\n\\n", full)
        full = re.sub(r"\\s+", " ", full)
        return full.strip()


def parse_sitemap_urls(xml_text: str) -> List[str]:
    root = ET.fromstring(xml_text)
    urls = [el.text.strip() for el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc") if el.text]
    if not urls:
        urls = [el.text.strip() for el in root.findall(".//loc") if el.text]
    if not urls:
        raise RuntimeError("No <loc> tags found in sitemap")

    out = []
    seen = set()
    for u in urls:
        u = normalize_url(u)
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0].strip()


def fetch_url(url: str, timeout: int, max_retries: int) -> tuple[int, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "rag-practice-sandbox/0.1"})
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return int(resp.getcode() or 200), resp.geturl(), body
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt >= max_retries:
                break
            backoff = min(5.0, 2 ** attempt)
            time.sleep(backoff * random.uniform(0.9, 1.1))
    raise RuntimeError(last_error or "unknown fetch error")


def fetch_sitemap_xml(url: str, timeout: int) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch sitemap '{url}': {exc}")


def extract_text(html_text: str) -> str:
    parser = TextExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.extracted_text()


def scrape_pages(
    cfg: SourceConfig,
    paths: RunPaths,
    limit: Optional[int] = None,
    dry_run: bool = False,
    same_host_only: bool = False,
    resume: bool = False,
    retry_failed: bool = False,
) -> Dict[str, Any]:
    manifest = {
        "source": cfg.source,
        "dataset": cfg.dataset,
        "sitemap_url": cfg.sitemap_url,
        "urls": [],
        "digest": "",
        "status_counts": {"ok": 0, "skip": 0, "error": 0},
    }

    log_message(paths.ingest_log, f"starting scrape for source={cfg.source} dataset={cfg.dataset} sitemap={cfg.sitemap_url}")
    sitemap_xml = fetch_sitemap_xml(cfg.sitemap_url, cfg.timeout_seconds)
    manifest["digest"] = sha1_hex(sitemap_xml)
    urls = parse_sitemap_urls(sitemap_xml)

    if same_host_only:
        allowed_host = urlparse(cfg.sitemap_url).netloc.lower()
        urls = [u for u in urls if urlparse(u).netloc.lower() == allowed_host]
        log_message(paths.ingest_log, f"same_host_only enabled; filtered urls to host={allowed_host}, count={len(urls)}")

    if limit is not None and limit > 0:
        urls = urls[:limit]
        log_message(paths.ingest_log, f"applied limit={limit}; urls now={len(urls)}")

    log_message(paths.ingest_log, f"sitemap parsed with {len(urls)} urls")

    manifest["url_count"] = len(urls)
    if dry_run:
        manifest["urls"] = urls
        return manifest

    if paths.urls_ledger.exists():
        processed = load_processed_urls(paths.urls_ledger) if resume else {}
    else:
        processed = {}

    if paths.urls_ledger.exists() and not resume:
        paths.urls_ledger.unlink()

    pages: List[ScrapedPage] = []
    skipped_existing = 0
    total_urls = len(urls)
    for i, url in enumerate(urls):
        position = i + 1
        log_message(paths.ingest_log, f"progress {position}/{total_urls}: fetching {url}")
        previous_status = processed.get(url)
        if previous_status in {"fetched", "fetched_empty"} or (previous_status == "failed" and not retry_failed):
            skipped_existing += 1
            if previous_status == "failed":
                manifest["status_counts"]["error"] += 1
                log_message(paths.ingest_log, f"progress {position}/{total_urls}: skipped already-failed {url}")
            else:
                manifest["status_counts"]["ok"] += 1
                log_message(paths.ingest_log, f"progress {position}/{total_urls}: skipped already-done {url}")
            continue

        if i > 0:
            time.sleep(max(0.0, cfg.delay_seconds))

        url_hash = sha1_hex(url)
        raw_file = paths.raw_dir / f"{url_hash}.json"
        try:
            status, final_url, html_body = fetch_url(url, cfg.timeout_seconds, cfg.max_retries)
            text = extract_text(html_body)
            title = ""
            match = re.search(r"<title>(.*?)</title>", html_body, flags=re.IGNORECASE | re.DOTALL)
            if match:
                title = re.sub(r"\\s+", " ", match.group(1)).strip()

            raw_payload = {
                "url": url,
                "fetched_at": now_iso(),
                "status": "fetched",
                "status_code": status,
                "title": title,
                "final_url": final_url,
                "text_content": text,
            }
            raw_file.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            manifest["status_counts"]["ok"] += 1
            if not text:
                manifest["status_counts"]["skip"] += 1
            else:
                pages.append(ScrapedPage(url=url, final_url=final_url, status_code=status, title=title, text=text, raw_file=raw_file))

            append_jsonl(
                paths.urls_ledger,
                {
                    "url": url,
                    "status": "fetched" if text else "fetched_empty",
                    "fetched_at": now_iso(),
                    "http_status": status,
                    "error": "",
                    "raw_page_file": str(raw_file.relative_to(paths.root.parent.parent.parent)),
                    "chunk_file": "",
                },
            )
            log_message(paths.ingest_log, f"fetched ok: {url}")
        except Exception as exc:  # noqa: BLE001
            manifest["status_counts"]["error"] += 1
            raw_payload = {
                "url": url,
                "fetched_at": now_iso(),
                "status": "failed",
                "status_code": 0,
                "title": "",
                "final_url": url,
                "text_content": "",
            }
            raw_file.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            append_jsonl(
                paths.urls_ledger,
                {
                    "url": url,
                    "status": "failed",
                    "fetched_at": now_iso(),
                    "http_status": 0,
                    "error": str(exc),
                    "raw_page_file": str(raw_file.relative_to(paths.root.parent.parent.parent)),
                    "chunk_file": "",
                },
            )
            log_message(paths.ingest_log, f"fetched error: {url} :: {exc}")

    log_message(paths.ingest_log, (
        f"scrape complete for source={cfg.source}: "
        f"ok={manifest['status_counts']['ok']} "
        f"skip={manifest['status_counts']['skip']} "
        f"error={manifest['status_counts']['error']} "
        f"total={len(urls)}"
    ))

    manifest["urls"] = urls
    manifest["skipped_existing"] = skipped_existing
    return {
        **manifest,
        "pages": pages,
        "sitemap_xml": sitemap_xml,
        "digest": manifest["digest"],
    }
