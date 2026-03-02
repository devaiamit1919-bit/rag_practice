"""Shared config, schema, and filesystem helpers for the RAG playground."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_CATALOG = DEFAULT_DATA_DIR / "catalog.json"


@dataclass
class SourceConfig:
    source: str
    dataset: str
    sitemap_url: str
    delay_seconds: float
    timeout_seconds: int
    max_retries: int
    chunk_chars: int
    overlap: int
    top_k: int
    embedding_provider: str
    embedding_model: str


@dataclass
class RunPaths:
    root: Path
    manifest_dir: Path
    raw_dir: Path
    chunks_dir: Path
    index_dir: Path
    logs_dir: Path

    @property
    def sitemap_manifest(self) -> Path:
        return self.manifest_dir / "sitemap.json"

    @property
    def urls_ledger(self) -> Path:
        return self.manifest_dir / "urls.jsonl"

    @property
    def ingest_log(self) -> Path:
        return self.logs_dir / "ingest.log"

    def chunk_path(self, run_id: str) -> Path:
        return self.chunks_dir / f"{run_id}_chunks.jsonl"

    def index_path(self, run_id: str) -> Path:
        return self.index_dir / f"{run_id}_index.json"


def resolve_paths(source: str, dataset: str) -> RunPaths:
    root = DEFAULT_DATA_DIR / "sources" / source / dataset
    return RunPaths(
        root=root,
        manifest_dir=root / "manifest",
        raw_dir=root / "raw_pages" / "by_url",
        chunks_dir=root / "chunks",
        index_dir=root / "index",
        logs_dir=root / "logs",
    )


def ensure_dirs(paths: RunPaths) -> None:
    for p in (paths.root, paths.manifest_dir, paths.raw_dir, paths.chunks_dir, paths.index_dir, paths.logs_dir):
        p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_catalog(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Catalog not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    default_source = raw.get("default_source")
    default_dataset = raw.get("default_dataset")
    if not default_source or not default_dataset:
        raise RuntimeError("catalog.json must include default_source and default_dataset")

    sources = raw.get("sources", {})
    if default_source not in sources:
        raise RuntimeError(f"Default source '{default_source}' missing in catalog")

    dataset_cfg = sources[default_source].get("datasets", {}).get(default_dataset)
    if not dataset_cfg:
        raise RuntimeError(f"Default dataset '{default_dataset}' missing for source '{default_source}'")

    fetch_cfg = dataset_cfg.get("fetch", {})
    chunk_cfg = dataset_cfg.get("chunking", {})
    embedding_cfg = dataset_cfg.get("embedding", {})

    return {
        "default_source": default_source,
        "default_dataset": default_dataset,
        "all": raw,
        "source_cfg": SourceConfig(
            source=default_source,
            dataset=default_dataset,
            sitemap_url=dataset_cfg.get("sitemap_url", ""),
            delay_seconds=float(fetch_cfg.get("delay_seconds", 1.0)),
            timeout_seconds=int(fetch_cfg.get("timeout_seconds", 20)),
            max_retries=int(fetch_cfg.get("max_retries", 2)),
            chunk_chars=int(chunk_cfg.get("chunk_chars", 450)),
            overlap=int(chunk_cfg.get("overlap", 70)),
            top_k=int(dataset_cfg.get("top_k", 3)),
            embedding_provider=str(embedding_cfg.get("provider", "tf")),
            embedding_model=str(embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        ),
    }


def resolve_dataset_config(
    catalog: Dict[str, Any],
    source: Optional[str],
    dataset: Optional[str],
    ask_delay: Optional[float] = None,
) -> SourceConfig:
    sources = catalog["all"].get("sources", {})
    source_name = source or catalog["default_source"]
    if source_name not in sources:
        raise RuntimeError(f"Source '{source_name}' not found in catalog")

    source_data = sources[source_name]
    dataset_name = dataset or catalog["default_dataset"]
    dataset_data = source_data.get("datasets", {}).get(dataset_name)
    if not dataset_data:
        raise RuntimeError(f"Dataset '{dataset_name}' not found for source '{source_name}'")

    fetch_cfg = dataset_data.get("fetch", {})
    chunk_cfg = dataset_data.get("chunking", {})
    embedding_cfg = dataset_data.get("embedding", {})
    resolved_delay = ask_delay if ask_delay is not None else float(fetch_cfg.get("delay_seconds", 1.0))

    return SourceConfig(
        source=source_name,
        dataset=dataset_name,
        sitemap_url=dataset_data.get("sitemap_url", ""),
        delay_seconds=resolved_delay,
        timeout_seconds=int(fetch_cfg.get("timeout_seconds", 20)),
        max_retries=int(fetch_cfg.get("max_retries", 2)),
        chunk_chars=int(chunk_cfg.get("chunk_chars", 450)),
        overlap=int(chunk_cfg.get("overlap", 70)),
        top_k=int(dataset_data.get("top_k", catalog["source_cfg"].top_k)),
        embedding_provider=str(embedding_cfg.get("provider", "tf")),
        embedding_model=str(embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
    )


def log_message(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{now_iso()} {message}\\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\\n")
