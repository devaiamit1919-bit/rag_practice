"""Query/retrieval CLI for the RAG playground.

Usage:
  python3 rag_query.py "your question" [--source S] [--dataset D] [--top-k K]
  python3 rag_query.py --question "..." --ollama-model qwen3.5:9b
"""
from __future__ import annotations

import json
import math
import urllib.request
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rag_embedding import tokenize


DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 180
DEFAULT_MAX_CONTEXT_CHARS = 12000


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


def load_index(source: str, dataset: str, run_id: Optional[str], catalog_path: Path) -> Tuple[
    Dict,
    Dict[str, Any],
    List[Dict[str, Any]],
    str,
]:
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
    metadata = {
        "embedding_provider": raw.get("embedding_provider", "tf"),
        "embedding_model": raw.get("embedding_model", ""),
        "vocab": raw.get("vocab", {}),
        "created_at": raw.get("created_at"),
    }
    return metadata["vocab"], metadata, raw.get("items", []), selected_run


def retrieve(
    question: str,
    source: str,
    dataset: str,
    top_k: int,
    catalog_path: Path,
    run_id: Optional[str] = None,
) -> Tuple[List[Tuple[Dict[str, Any], float]], str]:
    vocab, metadata, items, selected_run = load_index(source, dataset, run_id, catalog_path)
    provider = metadata.get("embedding_provider", "tf")
    if provider == "sentence_transformers":
        from rag_embedding_st import vectorize_query
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        qvec = vectorize_query(model, question)
    else:
        qvec = vectorize(question, vocab)
    scored = []
    for item in items:
        score = cosine_similarity(qvec, item.get("vector", []))
        if score > 0:
            scored.append((item, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], (run_id or selected_run)


def ask_ollama(
    question: str,
    context_snippets: List[Tuple[Dict[str, Any], float]],
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    print_prompt: bool = False,
) -> str:
    if not context_snippets:
        context = "No retrieved context available."
        references: List[str] = []
    else:
        references = []
        lines: List[str] = []
        for idx, (item, score) in enumerate(context_snippets, start=1):
            url = item.get("url", "").strip()
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            if url and url not in references:
                references.append(url)
            heading = f"[{idx}] {title}" if title else f"[{idx}]"
            lines.append(f"{heading}\n{url}\n{text}")
        context = "\n\n".join(lines)[:max_context_chars]

    system_prompt = (
        "You are a careful research-style assistant. "
        "Use only the provided context. "
        "If context is insufficient, say so clearly."
    )
    user_prompt = (
        "Write the answer as a short article.\n"
        "Use clear markdown sections: Title, Overview, Key Details, Conclusion.\n"
        "After the article, add a section 'References' that lists source links used.\n"
        "Only use links that appear in the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "If context is weak or irrelevant, do not invent information."
    )

    if print_prompt:
        print("==== Ollama request payload ====")
        print(f"model: {model}")
        print(f"endpoint: {endpoint}")
        print("system_prompt:")
        print(system_prompt)
        print("user_prompt:")
        print(user_prompt)
        print("options: temperature=0.2, num_predict=768")
        print("====")

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 768},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url=f"{endpoint}/api/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    message = raw.get("message", {}).get("content")
    if message:
        article = str(message).strip()
        if references:
            if "References" not in article:
                article += "\n\n### References\n"
            for i, ref in enumerate(references, start=1):
                article += f"- [{i}] {ref}\n"
        return article
    raise RuntimeError("ollama response missing message.content")


def answer(
    question: str,
    source: str,
    dataset: str,
    top_k: int,
    catalog_path: Path,
    run_id: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    print_prompt: bool = False,
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

    try:
        final_answer = ask_ollama(
            question,
            matches,
            model=ollama_model,
            timeout_seconds=ollama_timeout,
            max_context_chars=max_context_chars,
            print_prompt=print_prompt,
        )
        print("Answer:")
        print(final_answer)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not generate final answer from Ollama: {exc}")
        print("Showing retrieval snippets only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local RAG index")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--source", default="selfstudys")
    parser.add_argument("--dataset", default="jee")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--catalog", default=str(Path(__file__).resolve().parent / "data" / "catalog.json"))
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("Empty question.")

    answer(
        question=question,
        source=args.source,
        dataset=args.dataset,
        top_k=args.top_k,
        catalog_path=Path(args.catalog),
        run_id=args.run_id,
        ollama_model=args.ollama_model,
        ollama_timeout=args.ollama_timeout,
        max_context_chars=args.max_context_chars,
        print_prompt=args.print_prompt,
    )


if __name__ == "__main__":
    main()
