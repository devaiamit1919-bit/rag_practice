"""Embedding/chunking/index stage for the RAG playground."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rag_config import RunPaths, SourceConfig, now_iso, sha1_hex
from rag_scraper import ScrapedPage


def tokenize(text: str) -> List[str]:
    return [tok for tok in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if len(tok) > 1]


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    text = " ".join(text.split()).strip()
    if not text:
        return []
    step = max(1, chunk_chars - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_chars])
    return chunks


def build_vocab(chunks: List[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for chunk in chunks:
        for tok in set(tokenize(chunk)):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def vectorize(text: str, vocab: Dict[str, int]) -> List[float]:
    counts = Counter(tokenize(text))
    vec = [0.0] * len(vocab)
    for term, count in counts.items():
        if term in vocab:
            vec[vocab[term]] = count
    return vec


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


def build_index(
    cfg: SourceConfig,
    paths: RunPaths,
    run_id: str,
    pages: List[ScrapedPage],
) -> Dict[str, Any]:
    all_chunks: List[ChunkRecord] = []
    for page in pages:
        chunks = chunk_text(page.text, cfg.chunk_chars, cfg.overlap)
        for idx, ch in enumerate(chunks):
            if not ch:
                continue
            all_chunks.append(
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

    if not all_chunks:
        return {"items": [], "vocab": {}}

    vocab = build_vocab([c.text for c in all_chunks])
    for item in all_chunks:
        item.vector = vectorize(item.text, vocab)

    chunk_records = [c.to_dict() for c in all_chunks]
    chunk_path = paths.chunk_path(run_id)
    with chunk_path.open("w", encoding="utf-8") as f:
        for item in chunk_records:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\\n")

    index_payload = {
        "source": cfg.source,
        "dataset": cfg.dataset,
        "run_id": run_id,
        "created_at": now_iso(),
        "sitemap_url": cfg.sitemap_url,
        "item_count": len(chunk_records),
        "vocab": vocab,
        "items": chunk_records,
    }
    index_path = paths.index_path(run_id)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "items": chunk_records,
        "vocab": vocab,
        "index_path": str(index_path.relative_to(index_path.parent.parent)),
        "chunk_path": str(chunk_path.relative_to(chunk_path.parent.parent)),
    }
