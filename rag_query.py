"""Query/retrieval CLI for the RAG playground.

Usage:
  python3 rag_query.py "your question" [--source S] [--dataset D] [--top-k K]
  python3 rag_query.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError

from rag_embedding import tokenize


DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 180
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_DISABLE_THINKING = True


def _extract_block_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        if value.get("type") == "text" and isinstance(value.get("content"), str):
            return value.get("content")
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            piece = _extract_block_text(item).strip()
            if piece:
                parts.append(piece)
        return "\n".join(parts)
    return ""


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


def load_index(
    source: str,
    dataset: str,
    run_id: Optional[str],
    catalog_path: Path,
) -> Tuple[Dict, Dict[str, Any], List[Dict[str, Any]], str]:
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

    scored: List[Tuple[Dict[str, Any], float]] = []
    for item in items:
        score = cosine_similarity(qvec, item.get("vector", []))
        if score > 0:
            scored.append((item, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], (run_id or selected_run)


def _extract_ollama_content(response: Dict[str, Any], allow_reasoning: bool) -> str | None:
    message = response.get("message")
    if isinstance(message, dict):
        content_text = _extract_block_text(message.get("content"))
        if content_text:
            return content_text
        if isinstance(message.get("text"), str):
            return message.get("text").strip()
        if isinstance(message.get("response"), str):
            return message.get("response").strip()
        if allow_reasoning and isinstance(message.get("thinking"), str):
            thinking = message.get("thinking").strip()
            if thinking:
                return thinking
        if allow_reasoning and isinstance(message.get("reasoning"), str):
            return message.get("reasoning")
    if isinstance(response.get("response"), str):
        return response.get("response").strip()
    if isinstance(response.get("text"), str):
        return response.get("text").strip()
    if allow_reasoning and isinstance(response.get("thinking"), str):
        thinking = response.get("thinking").strip()
        if thinking:
            return thinking
    if isinstance(message, str):
        return message
    if isinstance(response.get("choices"), list) and response["choices"]:
        for choice in response["choices"]:
            if not isinstance(choice, dict):
                continue
            message_choice = choice.get("message", {})
            if isinstance(message_choice, dict):
                extracted = _extract_block_text(message_choice.get("content"))
                if extracted:
                    return extracted
    return None


def _strip_thinking(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        left = text.find("<think>")
        right = text.find("</think>")
        if left != -1 and right != -1 and right > left:
            remaining = (text[:left] + text[right + len("</think>"):]).strip()
            if remaining:
                return remaining

    for marker in ("Final Answer:", "Final Output:", "Answer:"):
        idx = text.rfind(marker)
        if idx != -1:
            candidate = text[idx + len(marker):].strip()
            if candidate:
                return candidate

    marker = "Thinking Process:"
    idx = text.find(marker)
    if idx == -1:
        return text
    after = text[idx + len(marker):].strip()
    if "\n\n" in after:
        after = after.split("\n\n", 1)[1].strip()
    return after or text.replace(marker, "").strip()


def ask_ollama(
    question: str,
    context_snippets: List[Tuple[Dict[str, Any], float]],
    model: str = DEFAULT_OLLAMA_MODEL,
    endpoint: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    print_prompt: bool = False,
    disable_thinking: bool = DEFAULT_DISABLE_THINKING,
    num_predict: int = 128,
) -> str:
    if not context_snippets:
        context = "No retrieved context available."
        references: List[str] = []
    else:
        references = []
        lines: List[str] = []
        for idx, (item, _score) in enumerate(context_snippets, start=1):
            url = item.get("url", "").strip()
            title = item.get("title", "").strip()
            text = item.get("text", "").strip()
            if url and url not in references:
                references.append(url)
            heading = f"[{idx}] {title}" if title else f"[{idx}]"
            lines.append(f"{heading}\n{url}\n{text}")
        context = "\n\n".join(lines)[:max_context_chars]

    system_prompt = (
        "You are a careful research-style assistant. Use only the provided context. "
        "If context is insufficient, say so clearly."
    )
    if disable_thinking:
        system_prompt += " Do not show reasoning or chain-of-thought. Return only the final answer."

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
        print(f"timeout: {timeout_seconds}")
        print(f"num_predict: {num_predict}")
        print("system_prompt:")
        print(system_prompt)
        print("user_prompt:")
        print(user_prompt)
        print("====")

    options = {"temperature": 0.2, "num_predict": num_predict}
    chat_payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
    }
    if disable_thinking:
        chat_payload["think"] = False
    payload = json.dumps(chat_payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{endpoint}/api/chat",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    raw: Dict[str, Any] = {}
    content: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        content = _extract_ollama_content(raw, allow_reasoning=not disable_thinking)
        if not content and disable_thinking:
            # Some reasoning models return text only in `thinking`; recover and post-strip it.
            content = _extract_ollama_content(raw, allow_reasoning=True)
    except (socket.timeout, URLError):
        content = None

    if not content:
        generate_body: Dict[str, Any] = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": options,
        }
        if disable_thinking:
            generate_body["think"] = False
        generate_payload = json.dumps(generate_body).encode("utf-8")
        gen_req = urllib.request.Request(
            url=f"{endpoint}/api/generate",
            data=generate_payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(gen_req, timeout=timeout_seconds) as resp:
                gen_raw = json.loads(resp.read().decode("utf-8"))
            content = gen_raw.get("response") or gen_raw.get("text") or _extract_ollama_content(gen_raw, allow_reasoning=True)
            raw = gen_raw
        except (socket.timeout, URLError) as exc:
            raise RuntimeError(f"cannot reach Ollama at {endpoint}: {exc}") from exc

    if not content:
        error_msg = raw.get("error") if isinstance(raw, dict) else None
        if error_msg:
            raise RuntimeError(f"ollama error: {error_msg}")
        raise RuntimeError(f"ollama response missing content fields: keys={list(raw.keys()) if isinstance(raw, dict) else 'unknown'}")

    article = _strip_thinking(str(content).strip()) if disable_thinking else str(content).strip()
    if references:
        if "References" not in article:
            article += "\n\n### References\n"
        for i, ref in enumerate(references, start=1):
            article += f"- [{i}] {ref}\n"
    return article


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
    disable_thinking: bool = DEFAULT_DISABLE_THINKING,
    num_predict: int = 128,
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
            disable_thinking=disable_thinking,
            num_predict=num_predict,
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
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=DEFAULT_DISABLE_THINKING,
        help="Disable reasoning/thinking in model output (default: enabled).",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow model reasoning/thinking text in output.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny built-in smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Skip Ollama call (smoke test only)")
    args = parser.parse_args()

    if args.smoke_test:
        test_snippets: List[Tuple[Dict[str, Any], float]] = [
            (
                {
                    "source": "selfstudys",
                    "dataset": "jee",
                    "run_id": "smoke",
                    "url": "https://example.com/jee-basics",
                    "title": "JEE Overview",
                    "text": "JEE has two stages: JEE Main and JEE Advanced. It is an engineering entrance exam in India.",
                },
                1.0,
            )
        ]
        if args.dry_run:
            print("Smoke test: skipping Ollama call (dry-run)")
            print("model:", args.ollama_model)
            print("timeout:", args.ollama_timeout)
            print("num_predict:", args.num_predict)
            return
        print("Smoke test: calling Ollama with synthetic context")
        print("answer:")
        print(
            ask_ollama(
                "What is JEE?",
                test_snippets,
                model=args.ollama_model,
                timeout_seconds=args.ollama_timeout,
                max_context_chars=args.max_context_chars,
                print_prompt=args.print_prompt,
                disable_thinking=args.disable_thinking,
                num_predict=args.num_predict,
            )
        )
        return

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
        disable_thinking=args.disable_thinking,
        num_predict=args.num_predict,
    )


if __name__ == "__main__":
    main()
