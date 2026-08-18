"""
Main CLI Entrypoint & Pipeline Dispatcher for Osteoporosis RAG.

Dispatches commands to individual pipeline steps:
  - clean    : Text cleaning & extraction (src.parsing -> src.clean)
  - build    : Chunking & Indexing (src.chunking -> src.embedded)
  - ask      : Retrieval & Clinical Evidence Synthesis (src.embedded)
  - evaluate : Precision@K Benchmark Suite (src.evaluation)
  - chat     : Interactive Clinical Q&A loop
"""

import sys
import json
import argparse
from pathlib import Path

from typing import Optional, List, Dict, Any

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_EVAL_QUESTIONS_PATH,
    DEFAULT_GEMINI_MODEL
)
from src.clean import clean_all_guidelines
from src.embedded import (
    VectorStore,
    build_vector_index,
    check_scope_guardrail,
    classify_query_risk
)
from src.synthesis import ClinicalSynthesizer
from src.evaluation import run_full_evaluation


def handle_clean(input_dir: str, output_dir: str):
    """Execute text cleaning pipeline."""
    clean_all_guidelines(input_dir=input_dir, output_dir=output_dir)


def handle_build(input_dir: str, index_path: str, chunk_size: int, overlap: int):
    """Execute index building pipeline."""
    build_vector_index(input_dir=input_dir, index_path=index_path, chunk_size=chunk_size, overlap=overlap)


def handle_ask(
    query: str,
    index_path: str,
    top_k: int,
    mode: str,
    output_json: bool,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None
):
    """Execute clinical query retrieval and evidence synthesis."""
    target_index = Path(index_path)
    if not target_index.exists():
        print(f"[!] Vector index not found at '{target_index.resolve()}'. Run 'python main.py build' first.")
        return

    tier, guardrail_msg = classify_query_risk(query)
    if tier == "refuse_redirect":
        if output_json:
            print(json.dumps({"query": query, "risk_tier": tier, "in_scope": False, "message": guardrail_msg}))
        else:
            print(f"\n[SAFETY & GUARDRAIL REFUSAL]\n  {guardrail_msg}\n")
        return

    store = VectorStore.load(target_index)
    results = store.search(query=query, top_k=top_k, mode=mode)
    synthesizer = ClinicalSynthesizer(api_key=api_key, provider=provider, model_name=model)
    synthesis_resp = synthesizer.synthesize(query=query, retrieved_passages=results)

    # Append caution notice if patient-specific
    if tier == "needs_caution" and guardrail_msg not in synthesis_resp.clinical_caveats:
        synthesis_resp.clinical_caveats.insert(0, guardrail_msg)

    if output_json:
        evidence = [
            {
                "rank": i,
                "chunk_id": c.chunk_id,
                "document_name": c.document_name,
                "section": c.section_title,
                "page_number": c.page_number,
                "source_url": c.source_url,
                "score": round(s, 4),
                "text": c.text
            }
            for i, (c, s) in enumerate(results, 1)
        ]
        print(json.dumps({
            "query": query,
            "risk_tier": tier,
            "mode": mode,
            "evidence": evidence,
            "synthesis": synthesis_resp.to_dict()
        }, indent=2))
        return

    print("=" * 88)
    print(f"  CLINICAL RAG QUERY: \"{query}\" [Mode: {mode.upper()}]")
    print("=" * 88)
    if tier == "needs_caution":
        print(f"\n[PATIENT-SPECIFIC CAUTION ADVISORY]\n  {guardrail_msg}")
    else:
        print(f"\n[GUARDRAIL APPROVED] ({guardrail_msg})")
    print(f"[{mode.upper()} RETRIEVAL] Found {len(results)} ranked guideline passages\n")
    print("-" * 88 + "\n  EVIDENCE PANEL\n" + "-" * 88)

    for i, (chunk, score) in enumerate(results, start=1):
        print(f"\n[Source #{i}] Document: {chunk.document_name}")
        print(f"           Section : {chunk.section_title}")
        print(f"           Page    : {chunk.page_number}")
        print(f"           Chunk ID: {chunk.chunk_id}")
        if chunk.source_url:
            print(f"           URL     : {chunk.source_url}")
        print(f"           Score   : {score:.4f}")
        for line in chunk.text.splitlines()[:5]:
            print(f"             {line}")
        if len(chunk.text.splitlines()) > 5:
            print("             ...")

    print("\n" + synthesis_resp.format_markdown() + "\n")


def handle_chat(
    index_path: str,
    top_k: int,
    mode: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    api_key: Optional[str] = None
):
    """Interactive clinical Q&A session."""
    print("=" * 88)
    print(f"  CLINICAL RAG ASSISTANT [Mode: {mode.upper()}] (Type 'exit' to quit)")
    print("=" * 88)
    while True:
        try:
            q = input("\nEnter clinical question > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting session.")
                break
            tier, msg = classify_query_risk(q)
            if tier == "refuse_redirect":
                print(f"\n[SAFETY & GUARDRAIL REFUSAL]\n  {msg}\n")
                continue
            if tier == "needs_caution":
                print(f"\n[PATIENT-SPECIFIC CAUTION ADVISORY]\n  {msg}")
            handle_ask(
                query=q,
                index_path=index_path,
                top_k=top_k,
                mode=mode,
                output_json=False,
                provider=provider,
                model=model,
                api_key=api_key
            )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI subcommands."""
    parser = argparse.ArgumentParser(prog="python main.py", description="Clinical Guidelines RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Clean
    clean_p = subparsers.add_parser("clean", help="Ingest guideline PDFs & clean layout artifacts")
    clean_p.add_argument("--academic", nargs="?", const=True, default=True, help="Apply academic cleaning rules")
    clean_p.add_argument("--input-dir", default=str(DEFAULT_GUIDELINES_DIR), help="Input guidelines directory")
    clean_p.add_argument("--output-dir", default=str(DEFAULT_CLEANED_DIR), help="Output cleaned directory")
    clean_p.add_argument("--strategy", default=DEFAULT_PARTITION_STRATEGY, choices=["fast", "hi_res"])

    # 2. Build
    build_p = subparsers.add_parser("build", help="Chunk guidelines & build vector index")
    build_p.add_argument("--input-dir", default=str(DEFAULT_CLEANED_DIR), help="Cleaned text directory")
    build_p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Target index JSON path")
    build_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_CHARS, help="Target chunk size")
    build_p.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS, help="Chunk overlap size")

    # 3. Ask
    ask_p = subparsers.add_parser("ask", help="Ask clinical questions against indexed guidelines")
    ask_p.add_argument("query", type=str, help="Clinical question")
    ask_p.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default=DEFAULT_RETRIEVAL_MODE, help="Retrieval mode")
    ask_p.add_argument("--provider", choices=["gemini", "auto", "openai", "anthropic", "ollama", "fallback"], default="gemini", help="LLM synthesis provider (default: gemini)")
    ask_p.add_argument("--model", type=str, default=None, help="Gemini or LLM model identifier (default: gemini-1.5-flash)")
    ask_p.add_argument("--api-key", "--gemini-api-key", dest="api_key", type=str, default=None, help="Google Gemini API key (or set GEMINI_API_KEY)")
    ask_p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to vector index")
    ask_p.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="Number of passages to retrieve")
    ask_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 4. Evaluate
    eval_p = subparsers.add_parser("evaluate", aliases=["eval"], help="Run Precision@K benchmark across retrieval modes")
    eval_p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to vector index")
    eval_p.add_argument("--questions", default=str(DEFAULT_EVAL_QUESTIONS_PATH), help="Evaluation questions JSON")
    eval_p.add_argument("--cleaned-dir", default=str(DEFAULT_CLEANED_DIR), help="Cleaned text directory for ablation")

    # 5. Chat
    chat_p = subparsers.add_parser("chat", help="Start interactive assistant terminal session")
    chat_p.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default=DEFAULT_RETRIEVAL_MODE)
    chat_p.add_argument("--provider", choices=["gemini", "auto", "openai", "anthropic", "ollama", "fallback"], default="gemini", help="LLM synthesis provider")
    chat_p.add_argument("--model", type=str, default=None, help="Gemini model identifier (default: gemini-1.5-flash)")
    chat_p.add_argument("--api-key", "--gemini-api-key", dest="api_key", type=str, default=None, help="Google Gemini API key")
    chat_p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    chat_p.add_argument("--top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)

    # 6. Full Pipeline (Clean -> Chunk -> Embed)
    pipe_p = subparsers.add_parser("pipeline", aliases=["run-all"], help="Run full pipeline: Clean PDFs -> Chunk -> Build Vector Index")
    pipe_p.add_argument("--academic", nargs="?", const=True, default=True, help="Apply academic cleaning rules")
    pipe_p.add_argument("--strategy", default=DEFAULT_PARTITION_STRATEGY, choices=["fast", "hi_res"])
    pipe_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_CHARS, help="Target chunk size")
    pipe_p.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP_CHARS, help="Chunk overlap size")

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "clean":
        clean_all_guidelines(academic=args.academic, input_dir=args.input_dir, output_dir=args.output_dir, strategy=args.strategy)
    elif args.command == "build":
        build_vector_index(input_dir=args.input_dir, index_path=args.index_path, chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.command in ("pipeline", "run-all"):
        print("\n" + "=" * 84)
        print("  STEP 1 & 2: PDF PARSING & SMART TEXT CLEANING")
        print("=" * 84)
        clean_all_guidelines(academic=args.academic, strategy=args.strategy)
        print("\n" + "=" * 84)
        print("  STEP 3 & 4: SEMANTIC CHUNKING & VECTOR STORE EMBEDDING")
        print("=" * 84)
        build_vector_index(chunk_size=args.chunk_size, overlap=args.overlap)
        print("\n[OK] Complete end-to-end pipeline finished successfully!\n")
    elif args.command == "ask":
        handle_ask(
            query=args.query,
            index_path=args.index_path,
            top_k=args.top_k,
            mode=args.mode,
            output_json=args.json,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key
        )
    elif args.command in ("evaluate", "eval"):
        run_full_evaluation(index_path=args.index_path, questions_path=args.questions, cleaned_dir=args.cleaned_dir)
    elif args.command == "chat":
        handle_chat(
            index_path=args.index_path,
            top_k=args.top_k,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
