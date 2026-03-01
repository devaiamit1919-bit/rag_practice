"""Query/retrieval stage for the RAG playground."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rag_embedding import tokenize


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def vectorize(text: str, vocab: Dict[str, int]) -> List[float]:
    from collections import Counter

    counts = Counter(tokenize(text))
    vec = [0.0] * len(vocab)
    for term, count in counts.items():
        if term in vocab:
            vec[vocab[term]] = count
    return vec


def resolve_latest_run_id(index_dir: Path) -> Optional[str]:
    if not index_dir.exists():
        return None
    candidates = [p.stem[:-6] for p in index_dir.glob("*_index.json") if p.stem.endswith("_index")]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def load_index(source: str, dataset: str, run_id: Optional[str], catalog_path: Path) -> Tuple[Dict, List[Dict[str, Any]], str]:
    # late import to avoid circular dependency
    from rag_config import parse_catalog, resolve_paths

    catalog = parse_catalog(catalog_path)
    sources = catalog["all"].get("sources", {})
    if source not in sources:
        raise RuntimeError(f"Source '{source}' not found")
    if dataset not in sources[source].get("datasets", {}):
        raise RuntimeError(f"Dataset '{dataset}' not found for source '{source}'")

    paths = resolve_paths(source, dataset)
    selected_run = run_id or resolve_latest_run_id(paths.index_dir)
    if not selected_run:
        raise RuntimeError(f"No index found for {source}/{dataset}")

    index_path = paths.index_path(selected_run)
    if not index_path.exists():
        raise RuntimeError(f"Index file not found: {index_path}")

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    return raw.get("vocab", {}), raw.get("items", []), selected_run


def retrieve(
    question: str,
    source: str,
    dataset: str,
    top_k: int,
    catalog_path: Path,
    run_id: Optional[str] = None,
) -> Tuple[List[Tuple[Dict[str, Any], float]], str]:
    vocab, items, selected_run = load_index(source, dataset, run_id, catalog_path)
    qvec = vectorize(question, vocab)
    scored = []
    for item in items:
        score = cosine_similarity(qvec, item.get("vector", []))
        if score > 0:
            scored.append((item, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], (run_id or selected_run)


def answer(
    question: str,
    source: str,
    dataset: str,
    top_k: int,
    catalog_path: Path,
    run_id: Optional[str] = None,
) -> None:
    matches, selected = retrieve(question, source, dataset, top_k, catalog_path, run_id)

    print(f"Query: {question}")
    print(f"Source: {source}/{dataset}")
    print(f"Run: {selected}")
    print(f"Top-K: {top_k}\n")

    if not matches:
        print("No relevant context found. Add more data or ask a different question.")
        return

    for i, (item, score) in enumerate(matches, start=1):
        snippet = item["text"][:250]
        if len(item["text"]) > 250:
            snippet += "..."
        print(f"{i}. score={score:.4f}")
        print(f"   source={item['source']}/{item['dataset']} run={item['run_id']}")
        print(f"   url={item['url']}")
        print(f"   {snippet}\n")
