"""
Main Entrypoint for the Medical Guideline RAG Pipeline CLI.
"""

import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_PARTITION_STRATEGY,
    CLINICAL_KEYWORDS
)
from src.ingestion import clean_all_guidelines
from src.chunking import chunk_document
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# Build Command Execution Logic
# =====================================================================

def execute_build(
    input_dir: str = str(DEFAULT_CLEANED_DIR),
    index_path: str = str(DEFAULT_INDEX_PATH),
    chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_CHARS
) -> Dict[str, Any]:
    """Read cleaned guideline text files, chunk them, build the vector store, and persist to disk."""
    input_path = Path(input_dir)
    target_index_path = Path(index_path)

    if not input_path.exists():
        print(f"[!] Cleaned data directory '{input_path.resolve()}' not found. Run 'python main.py clean' first.")
        return {}

    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"[!] No cleaned text files found in '{input_path.resolve()}'. Run 'python main.py clean' first.")
        return {}

    print("=" * 80)
    print(f"  RAG PIPELINE: BUILDING VECTOR INDEX FROM '{input_path}'")
    print("=" * 80)

    store = VectorStore()
    all_chunks = []
    doc_stats = []

    for file_path in txt_files:
        doc_id = file_path.stem
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as exc:
            logger.error(f"Failed to read file '{file_path}': {exc}")
            continue

        doc_chunks = chunk_document(
            document_text=content,
            document_id=doc_id,
            target_chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        all_chunks.extend(doc_chunks)
        doc_stats.append((doc_id, len(content), len(doc_chunks)))
        print(f"  -> {doc_id:<45} | {len(content):>6} chars | {len(doc_chunks):>3} chunks")

    store.add_chunks(all_chunks)
    store.save(target_index_path)

    total_chars = sum(s[1] for s in doc_stats)
    print("-" * 80)
    print(f"  Indexed {len(txt_files)} documents into {len(all_chunks)} semantic chunks ({len(store.vectors)} vectors).")
    print(f"  Index saved to: '{target_index_path.resolve()}'")
    print("=" * 80)
    print("[OK] Vector index build complete.\n")

    return {
        "documents": len(txt_files),
        "total_chars": total_chars,
        "total_chunks": len(all_chunks),
        "total_vectors": len(store.vectors),
        "index_path": str(target_index_path)
    }


# =====================================================================
# Ask / Query Command Execution Logic
# =====================================================================

def check_scope_guardrail(query: str) -> Tuple[bool, str]:
    """Evaluate whether a user query is clinically relevant to osteoporosis guidelines."""
    tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', query.lower()))
    matches = tokens.intersection(CLINICAL_KEYWORDS)
    
    if not matches:
        return False, (
            "[GUARDRAIL NOTICE] Query is OUT OF SCOPE. This clinical RAG system specializes in "
            "osteoporosis risk assessment, screening, bone mineral density (DXA), and fracture prevention guidelines. "
            "Please submit a clinical or bone health question."
        )
    return True, f"In-scope query matching keywords: {', '.join(sorted(matches)[:4])}"


def execute_ask(
    query: str,
    index_path: str = str(DEFAULT_INDEX_PATH),
    top_k: int = DEFAULT_RETRIEVAL_TOP_K
) -> Dict[str, Any]:
    """Retrieve guideline passages for a clinical query and format the Evidence Panel."""
    target_index = Path(index_path)
    if not target_index.exists():
        print(f"[!] Vector index not found at '{target_index.resolve()}'. Run 'python main.py build' first.")
        return {"error": f"Index not found at {target_index}"}

    is_in_scope, guardrail_msg = check_scope_guardrail(query)

    print("=" * 80)
    print(f"  CLINICAL RAG QUERY: \"{query}\"")
    print("=" * 80)

    if not is_in_scope:
        print(f"\n[GUARDRAIL REJECTED]")
        print(f"  {guardrail_msg}\n")
        print("=" * 80)
        return {
            "query": query,
            "in_scope": False,
            "message": guardrail_msg,
            "evidence": []
        }

    store = VectorStore.load(target_index)
    results = store.search(query, top_k=top_k)

    print(f"\n[GUARDRAIL APPROVED] ({guardrail_msg})")
    print(f"[RETRIEVAL] Found {len(results)} relevant guideline passages\n")

    print("-" * 80)
    print("  EVIDENCE PANEL")
    print("-" * 80)

    evidence_records: List[Dict[str, Any]] = []
    for i, (chunk, score) in enumerate(results, start=1):
        print(f"\n[Source #{i}] Document: {chunk.document_id}")
        print(f"           Section : {chunk.section_title}")
        print(f"           Score   : {score:.4f}")
        print(f"           Excerpt :")
        for line in chunk.text.splitlines()[:6]:
            print(f"             {line}")
        if len(chunk.text.splitlines()) > 6:
            print("             ...")

        evidence_records.append({
            "rank": i,
            "document_id": chunk.document_id,
            "section": chunk.section_title,
            "score": round(score, 4),
            "text": chunk.text
        })

    print("\n" + "=" * 80)
    print("  CLINICAL GUIDELINE SYNTHESIS")
    print("=" * 80)

    if results:
        top_chunk = results[0][0]
        print(f"\nBased on {top_chunk.document_id} ({top_chunk.section_title}):\n")
        print(top_chunk.text[:500] + ("..." if len(top_chunk.text) > 500 else ""))
    else:
        print("No matching clinical recommendations found.")

    print("\n" + "=" * 80 + "\n")

    return {
        "query": query,
        "in_scope": True,
        "results_count": len(results),
        "evidence": evidence_records
    }


# =====================================================================
# CLI Parser & Main Dispatcher
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Clinical Guidelines RAG Preprocessing, Indexing & Query Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline subcommands")

    # 1. Clean sub-command
    clean_parser = subparsers.add_parser("clean", help="Ingest guideline PDFs and clean text using layout parsing")
    clean_parser.add_argument(
        "--academic",
        nargs="?",
        const=True,
        default=True,
        help="Apply academic article cleaning rules. Optionally pass comma-separated filenames."
    )
    clean_parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_GUIDELINES_DIR),
        help="Directory containing source PDF files (default: data/guidelines)"
    )
    clean_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_CLEANED_DIR),
        help="Directory where cleaned text files are saved (default: data/cleaned)"
    )
    clean_parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_PARTITION_STRATEGY,
        choices=["fast", "hi_res"],
        help="Unstructured partitioning strategy: 'fast' or 'hi_res' (default: fast)"
    )

    # 2. Build sub-command (Vector index builder)
    build_parser = subparsers.add_parser("build", help="Build vector embeddings index from cleaned guidelines")
    build_parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_CLEANED_DIR),
        help="Directory containing cleaned text files (default: data/cleaned)"
    )
    build_parser.add_argument(
        "--index-path",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help="Target index storage path (default: data/vector_store/index.json)"
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_CHARS,
        help=f"Target chunk size in characters (default: {DEFAULT_CHUNK_SIZE_CHARS})"
    )
    build_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_CHARS,
        help=f"Chunk overlap size (default: {DEFAULT_CHUNK_OVERLAP_CHARS})"
    )

    # 3. Ask sub-command (RAG Query Agent)
    ask_parser = subparsers.add_parser("ask", help="Ask clinical questions against the guideline vector store")
    ask_parser.add_argument(
        "query",
        type=str,
        help="The clinical question to evaluate"
    )
    ask_parser.add_argument(
        "--index-path",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help="Path to the vector index (default: data/vector_store/index.json)"
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of evidence passages to retrieve (default: {DEFAULT_RETRIEVAL_TOP_K})"
    )

    return parser


def main():
    parser = build_parser()

    if len(sys.argv) == 1:
        print("\n[i] No subcommand provided. Showing help:\n")
        parser.print_help()
        print("\nQuickstart:")
        print("  python main.py clean             # Ingest and clean all guidelines")
        print("  python main.py build             # Chunk guidelines & build vector index")
        print("  python main.py ask \"<question>\"   # Query guidelines with evidence panel\n")
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "clean":
        clean_all_guidelines(
            academic=args.academic,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            strategy=args.strategy
        )
    elif args.command == "build":
        execute_build(
            input_dir=args.input_dir,
            index_path=args.index_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    elif args.command == "ask":
        execute_ask(
            query=args.query,
            index_path=args.index_path,
            top_k=args.top_k
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
