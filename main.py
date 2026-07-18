"""
main.py — CLI entry point for the semantic search research system.

Supports three modes:
    gui        — launch the desktop Tkinter GUI (default)
    index      — build a FAISS index from a PDF folder (headless)
    search     — run a single query against an existing index
    evaluate   — run a full evaluation against a labelled query set
    compare    — evaluate all three models and write a comparison CSV

Usage examples
--------------
    python main.py gui
    python main.py index  --folder ./data  --model minilm
    python main.py search --model minilm   --query "deep learning"
    python main.py evaluate --model mpnet  --queries evaluation/queries.json
    python main.py compare  --queries evaluation/queries.json
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from extraction.extractor       import DocumentRegistry, HybridExtractor
from preprocessing.preprocessor import TextPreprocessor
from embeddings.embedder        import EmbeddingEngine
from indexing.faiss_index       import FAISSIndex, ChunkMeta
from search.semantic_search     import SemanticSearch
from search.keyword_search      import KeywordSearch
from evaluation.evaluator       import SearchEvaluator
from evaluation.logger          import ExperimentLogger


# ─────────────────────────────────────────────────────────────────────────────
# BUILD — dataset ingestion + extraction + embedding + FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    folder:        Path,
    model_key:     str,
    use_chunking:  bool = None,
    chunk_size:    int  = None,
    chunk_overlap: int  = None,
) -> dict:
    """
    Full pipeline: scan PDFs → extract → preprocess → embed → index.

    Returns a dict with build statistics.
    """
    folder        = Path(folder)
    use_chunking  = config.USE_CHUNKING  if use_chunking  is None else use_chunking
    chunk_size    = config.CHUNK_SIZE    if chunk_size    is None else chunk_size
    chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    logger.info("=" * 60)
    logger.info("BUILD INDEX — model=%s  folder=%s", model_key, folder)
    logger.info("=" * 60)

    # ── 1. Registry ───────────────────────────────────────────────────────────
    registry = DocumentRegistry(config.REGISTRY_FILE)
    registry.scan_folder(folder, reset=True)
    if not registry.documents:
        logger.error("No PDFs found in %s", folder)
        sys.exit(1)
    logger.info("Registered %d PDF(s)", len(registry))

    # ── 2. Extraction + preprocessing ─────────────────────────────────────────
    extractor    = HybridExtractor(
        min_text_length  = config.OCR_MIN_TEXT_LENGTH,
        min_text_quality = config.OCR_MIN_TEXT_QUALITY,
        tesseract_cmd    = config.TESSERACT_CMD,
        ocr_dpi          = config.OCR_DPI,
        ocr_language     = config.OCR_LANGUAGE,
    )
    preprocessor = TextPreprocessor(
        use_chunking  = use_chunking,
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    all_chunks:  list = []
    chunk_metas: list = []
    doc_texts:   list = []
    total_extract_time = 0.0

    for doc_entry in registry:
        doc_id = doc_entry["doc_id"]
        cached_text = registry.get_extracted_text(doc_id)
        if cached_text:
            logger.info("Retrieved cached text for: %s", doc_entry["filename"])
            ext_result = {
                "full_text": cached_text,
                "extraction_time": 0.0
            }
        else:
            logger.info("Extracting: %s", doc_entry["filename"])
            try:
                ext_result = extractor.extract_document(doc_entry["filepath"])
                registry.update_extracted_text(doc_id, ext_result["full_text"])
                total_extract_time += ext_result["extraction_time"]
            except Exception as exc:
                logger.error("Skipping %s: %s", doc_entry["filename"], exc)
                continue

        chunks = preprocessor.process(ext_result["full_text"], doc_entry["doc_id"])
        for chunk in chunks:
            all_chunks.append(chunk.text)
            chunk_metas.append(ChunkMeta(
                faiss_id    = len(chunk_metas),
                doc_id      = doc_entry["doc_id"],
                chunk_index = chunk.chunk_index,
                filename    = doc_entry["filename"],
                filepath    = doc_entry["filepath"],
                text        = chunk.text,
            ))

        doc_texts.append({
            "doc_id":   doc_entry["doc_id"],
            "filename": doc_entry["filename"],
            "filepath": doc_entry["filepath"],
            "text":     ext_result["full_text"],
        })

    logger.info(
        "Extracted %d document(s) → %d chunk(s) in %.1fs",
        len(doc_texts), len(all_chunks), total_extract_time,
    )

    if not all_chunks:
        logger.error("No text could be extracted.")
        sys.exit(1)

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    embedder = EmbeddingEngine(model_key=model_key, models_dir=config.MODELS_DIR)
    logger.info("Generating embeddings (%s)…", model_key)
    t_embed    = time.perf_counter()
    embeddings = embedder.encode(all_chunks)
    embed_time = time.perf_counter() - t_embed
    logger.info("Embeddings: shape=%s  time=%.2fs", embeddings.shape, embed_time)

    # ── 4. FAISS index ────────────────────────────────────────────────────────
    faiss_idx = FAISSIndex(model_key=model_key, indexes_dir=config.INDEXES_DIR)
    faiss_idx.build(embeddings, chunk_metas)
    faiss_idx.save()
    logger.info(
        "FAISS index saved: %d vectors  build_time=%.3fs",
        faiss_idx.total_vectors, faiss_idx.build_time,
    )

    # ── 5. Log experiment ─────────────────────────────────────────────────────
    exp_logger = ExperimentLogger(config.LOGS_DIR)
    run_id     = ExperimentLogger.make_run_id(model_key)
    exp_logger.log_experiment(
        run_id             = run_id,
        model_key          = model_key,
        model_name         = config.EMBEDDING_MODELS[model_key]["name"],
        search_type        = "semantic",
        num_documents      = len(registry),
        num_chunks         = len(all_chunks),
        embedding_dim      = embedder.dimension,
        indexing_time_s    = faiss_idx.build_time,
        embedding_time_s   = embed_time,
        use_chunking       = use_chunking,
        chunk_size         = chunk_size,
        chunk_overlap      = chunk_overlap,
        index_size_vectors = faiss_idx.total_vectors,
    )

    stats = {
        "run_id":        run_id,
        "model_key":     model_key,
        "num_documents": len(registry),
        "num_chunks":    len(all_chunks),
        "embed_time":    embed_time,
        "build_time":    faiss_idx.build_time,
        "dimension":     embedder.dimension,
        "doc_texts":     doc_texts,   # pass-through for keyword index builds
    }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH — single query
# ─────────────────────────────────────────────────────────────────────────────

def run_search(model_key: str, query: str, top_k: int = 10):
    engine = SemanticSearch(
        model_key   = model_key,
        indexes_dir = config.INDEXES_DIR,
        models_dir  = config.MODELS_DIR,
    )
    if not engine.load_index():
        logger.error(
            "No index found for model=%s. Run 'python main.py index' first.",
            model_key,
        )
        sys.exit(1)

    result = engine.search(query, top_k=top_k)
    _print_results(result)


def _print_results(result: dict):
    print(f"\nQuery : {result['query']}")
    print(f"Model : {result.get('model_key', 'keyword')}")
    print(f"Time  : {result['query_time_ms']} ms")
    print(f"{'-' * 80}")
    for hit in result["results"]:
        print(
            f"  [{hit['rank']:>2}]  score={hit['score']:.4f}  "
            f"file={hit['filename']}"
        )
        if hit.get("snippet"):
            snippet = hit["snippet"][:200].replace("\n", " ")
            print(f"         {snippet}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE — one model against a query set
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate(
    model_key:     str,
    queries_file:  Path,
    top_k:         int = 10,
    search_type:   str = "semantic",
    doc_texts:     list = None,
) -> dict:
    evaluator = SearchEvaluator(
        eval_ks     = config.EVAL_TOP_K_VALUES,
        results_dir = config.EVAL_RESULTS_DIR,
    )
    queries = evaluator.load_queries(queries_file)
    if not queries:
        logger.error("No queries found in %s", queries_file)
        sys.exit(1)

    if search_type == "semantic":
        engine = SemanticSearch(
            model_key   = model_key,
            indexes_dir = config.INDEXES_DIR,
            models_dir  = config.MODELS_DIR,
        )
        if not engine.load_index():
            logger.error(
                "No index for model=%s. Run 'main.py index' first.", model_key
            )
            sys.exit(1)
        search_fn = engine.search
    else:
        if not doc_texts:
            logger.error("doc_texts required for keyword evaluation.")
            sys.exit(1)
        kw = KeywordSearch()
        kw.build(doc_texts)
        search_fn = kw.search

    eval_output = evaluator.evaluate(
        queries     = queries,
        search_fn   = search_fn,
        top_k       = top_k,
        model_label = model_key,
        search_type = search_type,
    )

    # Print summary
    agg = eval_output.get("aggregate", {})
    print(f"\n{'-'*60}")
    print(f"  Model: {model_key}   Type: {search_type}   Top-K: {top_k}")
    print(f"{'-'*60}")
    for metric, val in agg.items():
        if metric not in ("model", "search_type", "top_k", "num_queries"):
            print(f"  {metric:<25} {val}")
    print()

    # Save per-query CSV
    ts_label = ExperimentLogger.make_run_id(model_key)
    csv_name = f"eval_{model_key}_{search_type}_{ts_label}.csv"
    out_path = evaluator.save_results_csv(eval_output, filename=csv_name)
    print(f"  Per-query results → {out_path}")

    # Log summary
    exp_logger = ExperimentLogger(config.LOGS_DIR)
    exp_logger.log_eval_summary(
        run_id    = ts_label,
        aggregate = agg,
    )

    return eval_output


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE — all three models side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def run_compare(folder: Path, queries_file: Path, top_k: int = 10):
    """Build indexes for all three models and produce a comparison table."""
    rows = []
    for model_key in config.EMBEDDING_MODELS:
        logger.info("\n%s Building index for %s %s", "="*20, model_key, "="*20)
        stats = build_index(folder=folder, model_key=model_key)

        for stype in ("semantic", "keyword"):
            dt = stats.get("doc_texts", []) if stype == "keyword" else None
            ev = run_evaluate(
                model_key    = model_key,
                queries_file = queries_file,
                top_k        = top_k,
                search_type  = stype,
                doc_texts    = dt,
            )
            agg = ev.get("aggregate", {})
            agg["model_key"]   = model_key
            agg["search_type"] = stype
            rows.append(agg)

    # Write comparison CSV
    if rows:
        comp_path = config.RESULTS_DIR / "model_comparison.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nComparison table → {comp_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Semantic PDF Search — Research CLI",
    )
    sub = p.add_subparsers(dest="command", help="sub-commands")

    # gui
    sub.add_parser("gui", help="Launch the desktop GUI (default)")

    # index
    pi = sub.add_parser("index", help="Build FAISS index from a PDF folder")
    pi.add_argument("--folder",   required=True,               help="PDF folder path")
    pi.add_argument("--model",    default=config.DEFAULT_MODEL, help="minilm|mpnet|scibert")
    pi.add_argument("--no-chunk", action="store_true",         help="Disable chunking")
    pi.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    pi.add_argument("--overlap",    type=int, default=config.CHUNK_OVERLAP)

    # search
    ps = sub.add_parser("search", help="Search with an existing index")
    ps.add_argument("--model",  default=config.DEFAULT_MODEL)
    ps.add_argument("--query",  required=True)
    ps.add_argument("--top-k",  type=int, default=config.DEFAULT_TOP_K)
    ps.add_argument("--type",   choices=["semantic", "keyword"], default="semantic")

    # evaluate
    pe = sub.add_parser("evaluate", help="Evaluate a model against a query set")
    pe.add_argument("--model",   default=config.DEFAULT_MODEL)
    pe.add_argument("--queries", default=str(config.EVAL_QUERIES_FILE))
    pe.add_argument("--top-k",   type=int, default=config.DEFAULT_TOP_K)
    pe.add_argument("--type",    choices=["semantic", "keyword"], default="semantic")
    pe.add_argument("--folder",  default=None,
                    help="PDF folder (needed for keyword evaluation)")

    # compare
    pc = sub.add_parser("compare", help="Build + evaluate all three models")
    pc.add_argument("--folder",  required=True)
    pc.add_argument("--queries", default=str(config.EVAL_QUERIES_FILE))
    pc.add_argument("--top-k",   type=int, default=config.DEFAULT_TOP_K)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None or args.command == "gui":
        from gui.app import SemanticSearchApp
        SemanticSearchApp.run()

    elif args.command == "index":
        build_index(
            folder        = Path(args.folder),
            model_key     = args.model,
            use_chunking  = not args.no_chunk,
            chunk_size    = args.chunk_size,
            chunk_overlap = args.overlap,
        )

    elif args.command == "search":
        run_search(
            model_key = args.model,
            query     = args.query,
            top_k     = args.top_k,
        )

    elif args.command == "evaluate":
        doc_texts = None
        if args.type == "keyword" and args.folder:
            # Need to rebuild keyword index from documents
            stats     = build_index(Path(args.folder), args.model)
            doc_texts = stats.get("doc_texts")
        run_evaluate(
            model_key    = args.model,
            queries_file = Path(args.queries),
            top_k        = args.top_k,
            search_type  = args.type,
            doc_texts    = doc_texts,
        )

    elif args.command == "compare":
        run_compare(
            folder       = Path(args.folder),
            queries_file = Path(args.queries),
            top_k        = args.top_k,
        )


if __name__ == "__main__":
    main()
