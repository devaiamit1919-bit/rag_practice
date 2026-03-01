# RAG Practice Playground

A source-aware local RAG ingest/playground for learning and experiments.

## What you get
- `rag_practice/rag_playground.py`
  - `ingest` command for sitemap crawling and indexing
  - `ask` command for retrieval against indexed chunks
- `rag_practice/rag_scraper.py` for sitemap/page scraping
- `rag_practice/rag_embedding.py` for chunking and index/vector building
- `rag_practice/rag_query.py` for retrieval/score computation
- `rag_practice/data/catalog.json`
  - source and dataset configuration
- Structured artifact directories for future datasets:
  - `data/sources/<source>/<dataset>/manifest/sitemap.json`
  - `data/sources/<source>/<dataset>/manifest/urls.jsonl`
  - `data/sources/<source>/<dataset>/raw_pages/by_url/<url_hash>.json`
  - `data/sources/<source>/<dataset>/chunks/<run_id>_chunks.jsonl`
  - `data/sources/<source>/<dataset>/index/<run_id>_index.json`
  - `data/sources/<source>/<dataset>/logs/ingest.log`

## Quick start

From this folder:

```bash
cd /Users/amit/programming/python_projects/llm-practice/rag_practice
```

1. Ingest from default source+dataset:

```bash
python3 rag_playground.py ingest
```

2. Ingest with options:

```bash
python3 rag_playground.py ingest --source selfstudys --dataset jee --limit 20 --ask-delay 1.0
python3 rag_playground.py ingest --source selfstudys --dataset jee --same-host-only --ask-delay 1.0
python3 rag_playground.py ingest --dry-run --source selfstudys --dataset jee
python3 rag_playground.py ingest --overwrite --run-id baseline_20260301
```

3. Ask against default source+dataset:

```bash
python3 rag_playground.py ask "What are the key parts of a RAG system?"
```

4. Ask against a specific run/dataset:

```bash
python3 rag_playground.py ask "What is RAG?" --source selfstudys --dataset jee --run-id baseline_20260301 --top-k 5
```

## Adding a new sitemap source later

Edit `data/catalog.json` and add a new source/dataset entry.
No Python code changes are required if CLI flags are used (`--source`, `--dataset`).

## Politeness defaults

- `delay_seconds`: `1.0`
- `timeout_seconds`: `20`
- `max_retries`: `2`
- retry backoff: `1s`, `2s`, `4s` (cap 5s)
- single-threaded, sequential fetching

## Notes

- Retrieval is TF+cosine for learning and can be swapped later for embedding models.
- If a specific run file exists already, use `--overwrite` to replace it.
