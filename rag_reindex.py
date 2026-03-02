"""Build an index from already scraped raw page files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rag_config import DEFAULT_CATALOG, parse_catalog, resolve_dataset_config, resolve_paths, now_iso
from rag_embedding import build_index as build_index_tf
from rag_embedding_st import build_index as build_index_st
from rag_scraper import ScrapedPage


def load_pages_from_raw(raw_dir: Path) -> List[ScrapedPage]:
    pages: List[ScrapedPage] = []
    for raw_file in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        url = payload.get("url", "")
        if not url:
            continue
        if payload.get("status") == "failed":
            continue
        pages.append(
            ScrapedPage(
                url=url,
                final_url=payload.get("final_url", url),
                status_code=int(payload.get("status_code", 0) or 0),
                title=payload.get("title", ""),
                text=payload.get("text_content", ""),
                raw_file=raw_file,
                error="",
            )
        )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild RAG index from scraped raw pages.")
    parser.add_argument("--source", default="selfstudys")
    parser.add_argument("--dataset", default="jee")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    catalog = parse_catalog(Path(args.catalog))
    config = resolve_dataset_config(catalog, args.source, args.dataset, ask_delay=None)
    paths = resolve_paths(config.source, config.dataset)

    run_id = args.run_id or now_iso().replace(":", "").replace("-", "").replace("T", "_")
    paths.root.mkdir(parents=True, exist_ok=True)
    index_path = paths.index_path(run_id)
    if index_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Index already exists for run_id={run_id}. Use --overwrite to replace it."
        )

    pages = load_pages_from_raw(paths.raw_dir)
    if not pages:
        raise RuntimeError("No valid raw pages found. Run scraper first.")

    if config.embedding_provider in {"st", "sentence_transformers", "sentence-transformer", "sentence-transformers"}:
        build_index_st(config, paths, run_id, pages)
    else:
        build_index_tf(config, paths, run_id, pages)

    print(f"Rebuild completed: {config.source}/{config.dataset}")
    print(f"run_id: {run_id}")
    print(f"index: {paths.index_path(run_id)}")


if __name__ == "__main__":
    main()
