"""RAG playground CLI.

Commands:
- `python3 rag_playground.py ingest`
- `python3 rag_playground.py ask "question"`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_config import DEFAULT_CATALOG, parse_catalog, resolve_dataset_config, resolve_paths, ensure_dirs, now_iso
from rag_query import answer
from rag_scraper import scrape_pages
from rag_embedding import build_index


def run_ingest(
    source: str | None = None,
    dataset: str | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
    run_id: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    ask_delay: float | None = None,
    same_host_only: bool = False,
) -> None:
    catalog = parse_catalog(catalog_path)
    config = resolve_dataset_config(catalog, source=source, dataset=dataset, ask_delay=ask_delay)
    paths = resolve_paths(config.source, config.dataset)
    ensure_dirs(paths)

    if not run_id:
        run_id = now_iso().replace(":", "").replace("-", "").replace("T", "_")

    if paths.index_path(run_id).exists() and not overwrite:
        raise RuntimeError(
            f"Index already exists for run_id={run_id}. Use --overwrite or a different --run-id."
        )

    scrape_result = scrape_pages(config, paths, limit=limit, dry_run=dry_run, same_host_only=same_host_only)
    print(f"Parsed {scrape_result['url_count']} urls from sitemap")

    if dry_run:
        for u in scrape_result.get("urls", []):
            print(u)
        return

    pages = scrape_result["pages"]
    if not pages:
        manifest = {
            "source": config.source,
            "dataset": config.dataset,
            "sitemap_url": config.sitemap_url,
            "retrieved_at": now_iso(),
            "run_id": run_id,
            "url_count": scrape_result["url_count"],
            "status_counts": scrape_result["status_counts"],
            "digest": scrape_result["digest"],
            "urls_file": str(paths.urls_ledger.relative_to(paths.root.parent.parent)),
            "index_file": str(paths.index_path(run_id).relative_to(paths.root.parent.parent)),
            "chunk_file": str(paths.chunk_path(run_id).relative_to(paths.root.parent.parent)),
        }
        paths.sitemap_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("No chunks created; all fetched pages were empty or failed.")

    build_index(config, paths, run_id, pages)
    manifest = {
        "source": config.source,
        "dataset": config.dataset,
        "sitemap_url": config.sitemap_url,
        "retrieved_at": now_iso(),
        "run_id": run_id,
        "url_count": scrape_result["url_count"],
        "status_counts": scrape_result["status_counts"],
        "digest": scrape_result["digest"],
        "urls_file": str(paths.urls_ledger.relative_to(paths.root.parent.parent)),
        "index_file": str(paths.index_path(run_id).relative_to(paths.root.parent.parent)),
        "chunk_file": str(paths.chunk_path(run_id).relative_to(paths.root.parent.parent)),
    }
    paths.sitemap_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Ingest completed for {config.source}/{config.dataset}")
    print(f"run_id: {run_id}")
    print(f"index: {paths.index_path(run_id)}")


def run_ask(
    question: str,
    source: str | None = None,
    dataset: str | None = None,
    top_k: int | None = None,
    run_id: str | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
) -> None:
    catalog = parse_catalog(catalog_path)
    source_name = source or catalog["default_source"]
    dataset_name = dataset or catalog["default_dataset"]
    config = resolve_dataset_config(catalog, source_name, dataset_name, ask_delay=None)
    selected_top_k = top_k if top_k is not None else config.top_k

    answer(
        question=question,
        source=source_name,
        dataset=dataset_name,
        top_k=selected_top_k,
        catalog_path=catalog_path,
        run_id=run_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument("--source")
    ingest_parser.add_argument("--dataset")
    ingest_parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ingest_parser.add_argument("--run-id")
    ingest_parser.add_argument("--limit", type=int)
    ingest_parser.add_argument("--dry-run", action="store_true")
    ingest_parser.add_argument("--overwrite", action="store_true")
    ingest_parser.add_argument("--ask-delay", type=float)
    ingest_parser.add_argument("--same-host-only", action="store_true")

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("question", nargs="?")
    ask_parser.add_argument("--source")
    ask_parser.add_argument("--dataset")
    ask_parser.add_argument("--top-k", type=int)
    ask_parser.add_argument("--run-id")
    ask_parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))

    args = parser.parse_args()

    if args.cmd == "ingest":
        run_ingest(
            source=args.source,
            dataset=args.dataset,
            catalog_path=Path(args.catalog),
            run_id=args.run_id,
            limit=args.limit,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            ask_delay=args.ask_delay,
            same_host_only=args.same_host_only,
        )
    elif args.cmd == "ask":
        question = args.question
        if not question:
            question = input("Enter your question: ").strip()
        if not question:
            raise SystemExit("Empty question.")
        run_ask(
            question=question,
            source=args.source,
            dataset=args.dataset,
            top_k=args.top_k,
            run_id=args.run_id,
            catalog_path=Path(args.catalog),
        )


if __name__ == "__main__":
    main()
