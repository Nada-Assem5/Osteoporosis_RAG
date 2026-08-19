"""
Clinical Practice Guidelines RAG Pipeline - Main Entrypoint (main.py).

Executes the 6-stage RAG pipeline end-to-end:
1. Ingest.py              : PDF Ingestion & Structural Noise Filtering
2. Chunk.py               : Section-Aware 400-Token Chunking & Metadata Enrichment
3. Embeddings.py          : Dense Sentence-Transformer Vector Encoding
4. Vector_db.py           : Persistent ChromaDB Store & Hybrid Index Construction
5. Retrieval.py           : Multi-Mode Retrieval, Precision@K Benchmark & Evidence Panel
6. Grounded_Generation.py : Evidence Synthesizer, 4-Tier Confidence Gating & Claim Verification
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import stages directly from scripts/ package
from scripts import (
    ingest,
    chunk,
    embeddings,
    vector_db,
    retrieval,
    grounded_generation
)


def run_full_pipeline(
    sample_query: str = "When should a DXA bone density scan be offered according to NICE guidelines?",
    top_k: int = 3,
    mode: str = "hybrid"
):
    """
    Executes all pipeline stages sequentially end-to-end.
    """
    print("\n" + "#" * 92)
    print("  CLINICAL PRACTICE GUIDELINE RAG PIPELINE: END-TO-END EXECUTION")
    print("#" * 92)

    # 1. Ingestion
    print("\n>>> STAGE 1: INGESTION")
    elements = ingest.run()

    # 2. Chunking
    print("\n>>> STAGE 2: SEMANTIC CHUNKING")
    chunks = chunk.run()

    # 3. Embeddings
    print("\n>>> STAGE 3: EMBEDDING GENERATION")
    vectors = embeddings.run()

    # 4. Vector Database
    print("\n>>> STAGE 4: PERSISTENT VECTOR DB & INDEXING")
    db = vector_db.run()

    # 5. Retrieval & Evidence Panel
    print(f"\n>>> STAGE 5: RETRIEVAL & EVIDENCE PANEL (Query: \"{sample_query}\")")
    # FIX: retrieve_evidence() returns a 3-tuple (results, evidence_panel,
    # confidence_assessment), not 2 - unpacking into only (results, panel)
    # raised "ValueError: too many values to unpack" and crashed the full
    # pipeline before Stage 6 ever ran. Also surface the Retrieval Confidence
    # Thresholds guardrail (Safety Workflow step 2) here instead of silently
    # dropping it, since it decides whether generation should be trusted.
    results, panel, confidence = retrieval.retrieve_evidence(sample_query, top_k=top_k, mode=mode)
    print(panel)
    if confidence.blocked:
        print(f"\n  [GUARDRAIL] Retrieval confidence check blocked generation: {confidence.block_reason}\n")

    # 6. Day 3 Grounded Generation
    print("\n>>> STAGE 6: GROUNDED CLINICAL GENERATION & CLAIM VERIFICATION")
    response = grounded_generation.run(sample_query, top_k=top_k, mode=mode)
    print(response.get("output_text", ""))

    print("\n" + "#" * 92)
    print("  [SUCCESS] FULL PIPELINE EXECUTION COMPLETED")
    print("#" * 92 + "\n")
    return response


def handle_chat(top_k: int = 3, mode: str = "hybrid"):
    """Interactive multi-turn clinical chat loop."""
    print("\n" + "=" * 80)
    print("  CLINICAL PRACTICE GUIDELINES RAG: INTERACTIVE ASSISTANT")
    print(f"  Search Mode: {mode.upper()} | Top-K: {top_k}")
    print("  Type 'quit', 'exit', or 'q' to end the session.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n[Clinician Query] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Ending clinical assistant session. Goodbye.")
                break

            resp = grounded_generation.run(user_input, top_k=top_k, mode=mode)
            print(resp.get("output_text", ""))
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Practice Guidelines RAG Pipeline (Single Source of Truth: scripts/)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline Stage Subcommands")

    # Command: run (or default)
    subparsers.add_parser("run", help="Run full pipeline end-to-end")

    # Command: clean / ingest
    subparsers.add_parser("clean", help="Stage 1: PDF Document Ingestion & Text Normalization")
    subparsers.add_parser("ingest", help="Stage 1: PDF Document Ingestion & Text Normalization")

    # Command: chunk
    subparsers.add_parser("chunk", help="Stage 2: Section-Aware 400-Token Chunking")

    # Command: embed
    subparsers.add_parser("embed", help="Stage 3: Dense Semantic Vector Generation")

    # Command: build
    subparsers.add_parser("build", help="Stage 2-4: Chunking, Embeddings, and ChromaDB Vector Indexing")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Stage 6: Query Knowledge Base with Evidence Synthesis")
    ask_parser.add_argument("query", type=str, help="Clinical question to synthesize")
    ask_parser.add_argument("--prompt", "-p", "--custom-prompt", dest="prompt", default=None, help="Custom generation prompt / instruction")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Top-K evidence passages (default: 3)")
    ask_parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")

    # Command: evaluate / benchmark
    eval_parser = subparsers.add_parser("evaluate", help="Stage 5: Evaluate Retrieval Benchmark (Precision@K, MRR, NDCG)")
    eval_parser.add_argument("--compare", action="store_true", help="Run multi-configuration comparison grid")
    eval_parser.add_argument("--top-k", type=int, default=3, help="Top-K evidence passages (default: 3)")
    eval_parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")
    eval_parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid RRF alpha weight (default: 0.5)")

    bench_parser = subparsers.add_parser("benchmark", help="Stage 5: Evaluate Retrieval Benchmark (Precision@K, MRR, NDCG)")
    bench_parser.add_argument("--compare", action="store_true", help="Run multi-configuration comparison grid")
    bench_parser.add_argument("--top-k", type=int, default=3, help="Top-K evidence passages (default: 3)")
    bench_parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")
    bench_parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid RRF alpha weight (default: 0.5)")

    # Command: compare
    compare_parser = subparsers.add_parser("compare", help="Stage 5: Comprehensive Multi-Configuration Retrieval Comparison")
    compare_parser.add_argument("--output-dir", type=str, default="data/eval_results", help="Directory to save Markdown and JSON comparison reports")

    # Command: experiment
    exp_parser = subparsers.add_parser("experiment", help="Multi-Dimensional Grid Evaluation (Chunk Size × Overlap × Model × Search × Top-K)")
    exp_parser.add_argument("--quick", action="store_true", help="Run focused subset of key configurations")
    exp_parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None, help="Chunk sizes in tokens")
    exp_parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=None, help="Chunk overlaps in tokens")
    exp_parser.add_argument("--models", type=str, nargs="+", default=None, help="Embedding models")
    exp_parser.add_argument("--search-types", type=str, nargs="+", choices=["keyword", "semantic", "hybrid"], default=None, help="Search modes")
    exp_parser.add_argument("--top-k", type=int, nargs="+", default=None, help="Top-K values")
    exp_parser.add_argument("--hybrid-alphas", type=float, nargs="+", default=None, help="Hybrid RRF alpha weights")
    exp_parser.add_argument("--output-dir", type=str, default="data/eval_results", help="Output directory")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Interactive Clinical Chat Assistant")
    chat_parser.add_argument("--top-k", type=int, default=3, help="Top-K evidence passages")
    chat_parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")

    args = parser.parse_args()

    if args.command in {None, "run"}:
        run_full_pipeline()
    elif args.command in {"clean", "ingest"}:
        ingest.run()
    elif args.command == "chunk":
        chunk.run()
    elif args.command == "embed":
        embeddings.run()
    elif args.command == "build":
        chunk.run()
        embeddings.run()
        vector_db.run()
    elif args.command == "ask":
        resp = grounded_generation.run(args.query, custom_prompt=args.prompt, top_k=args.top_k, mode=args.mode)
        print(resp.get("output_text", ""))
    elif args.command in {"evaluate", "benchmark"}:
        if getattr(args, "compare", False):
            retrieval.run_retrieval_comparison()
        else:
            retrieval.run_retrieval_benchmark(top_k=args.top_k, mode=args.mode, alpha=getattr(args, "alpha", 0.5))
    elif args.command == "compare":
        retrieval.run_retrieval_comparison(output_dir=Path(args.output_dir))
    elif args.command == "experiment":
        from scripts.evaluate_experiments import run_experiments
        run_experiments(
            quick=args.quick,
            chunk_sizes=args.chunk_sizes,
            chunk_overlaps=args.chunk_overlaps,
            models=args.models,
            search_types=args.search_types,
            top_k_values=args.top_k,
            hybrid_alphas=args.hybrid_alphas,
            output_dir=Path(args.output_dir)
        )
    elif args.command == "chat":
        handle_chat(top_k=args.top_k, mode=args.mode)


if __name__ == "__main__":
    main()