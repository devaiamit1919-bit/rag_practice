"""Sentence-Transformers based chunking/vector index for the RAG playground."""
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rag_config import RunPaths, SourceConfig, now_iso, sha1_hex
from rag_embedding import chunk_text
from rag_scraper import ScrapedPage


@dataclass
class ChunkRecord:
    source: str
    dataset: str
    run_id: str
    url: str
    url_hash: str
    chunk_index: int
    text: str
    title: str
    vector: List[float]

    def to_dict(self) -> Dict:
        return {
            "chunk_id": f"{self.url_hash}:{self.chunk_index}",
            "source": self.source,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "url": self.url,
            "url_hash": self.url_hash,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "text": self.text,
            "chars": len(self.text),
            "vector": self.vector,
        }


def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers torch"
        ) from exc

    return SentenceTransformer(model_name)


def vectorize_texts(model, texts: List[str]) -> List[List[float]]:
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in vectors]


def vectorize_query(model, text: str) -> List[float]:
    vectors = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    first = vectors[0]
    return first.tolist() if hasattr(first, "tolist") else list(first)


def build_index(
    cfg: SourceConfig,
    paths: RunPaths,
    run_id: str,
    pages: List[Any],
) -> Dict[str, Any]:
    chunks: List[ChunkRecord] = []
    for page in pages:
        for idx, ch in enumerate(chunk_text(page.text, cfg.chunk_chars, cfg.overlap)):
            if not ch:
                continue
            chunks.append(
                ChunkRecord(
                    source=cfg.source,
                    dataset=cfg.dataset,
                    run_id=run_id,
                    url=page.url,
                    url_hash=sha1_hex(page.url),
                    chunk_index=idx,
                    text=ch,
                    title=page.title,
                    vector=[],
                )
            )

    if not chunks:
        return {"items": [], "vocab": {}, "embedding_provider": "sentence_transformers", "embedding_model": cfg.embedding_model}

    model = _load_model(cfg.embedding_model)
    texts = [c.text for c in chunks]
    vectors = vectorize_texts(model, texts)

    for chunk_record, vec in zip(chunks, vectors):
        chunk_record.vector = vec

    chunk_records = [c.to_dict() for c in chunks]
    chunk_path = paths.chunk_path(run_id)
    with chunk_path.open("w", encoding="utf-8") as f:
        for item in chunk_records:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")

    dim = len(vectors[0]) if vectors else 0
    index_payload = {
        "source": cfg.source,
        "dataset": cfg.dataset,
        "run_id": run_id,
        "created_at": now_iso(),
        "sitemap_url": cfg.sitemap_url,
        "item_count": len(chunk_records),
        "vocab": {},
        "embedding_provider": "sentence_transformers",
        "embedding_model": cfg.embedding_model,
        "embedding_dim": dim,
        "items": chunk_records,
    }
    index_path = paths.index_path(run_id)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "items": chunk_records,
        "vocab": {},
        "embedding_provider": "sentence_transformers",
        "embedding_model": cfg.embedding_model,
        "index_path": str(index_path.relative_to(index_path.parent.parent)),
        "chunk_path": str(chunk_path.relative_to(chunk_path.parent.parent)),
    }


def run_from_scraped(
    source: str = "selfstudys",
    dataset: str = "jee",
    catalog_path: str = "data/catalog.json",
    run_id: str | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    from rag_config import parse_catalog, resolve_dataset_config, resolve_paths

    catalog = parse_catalog(Path(catalog_path))
    cfg = resolve_dataset_config(catalog, source, dataset, ask_delay=None)
    paths = resolve_paths(cfg.source, cfg.dataset)
    run_id = run_id or now_iso().replace(":", "").replace("-", "").replace("T", "_")

    paths.index_dir.mkdir(parents=True, exist_ok=True)
    index_path = paths.index_path(run_id)
    if index_path.exists() and not overwrite:
        raise RuntimeError(f"Index already exists for run_id={run_id}. Use --overwrite.")

    pages: List[ScrapedPage] = []
    for raw_file in sorted(paths.raw_dir.glob("*.json")):
        try:
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "failed":
            continue
        url = payload.get("url", "")
        if not url:
            continue
        pages.append(
            ScrapedPage(
                url=url,
                final_url=payload.get("final_url", url),
                status_code=int(payload.get("status_code", 0) or 0),
                title=payload.get("title", ""),
                text=payload.get("text_content", ""),
                raw_file=raw_file,
                error=payload.get("error", ""),
            )
        )

    if not pages:
        raise RuntimeError("No valid scraped pages found. Run rag_scraper.py first.")

    return build_index(cfg, paths, run_id, pages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embedding index from already-scraped raw pages (ST embeddings)."
    )
    parser.add_argument("--source", default="selfstudys")
    parser.add_argument("--dataset", default="jee")
    parser.add_argument("--catalog", default="data/catalog.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = run_from_scraped(
        source=args.source,
        dataset=args.dataset,
        catalog_path=args.catalog,
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    print(f"index_path: {result.get('index_path')}")
    print(f"chunk_path: {result.get('chunk_path')}")
    print(f"items: {len(result.get('items', []))}")


if __name__ == "__main__":
    main()
