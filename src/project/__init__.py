"""Command-line entry point and end-to-end RAG pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import RAGPipeline


def main() -> None:
	"""Run the document indexing or question-answering command."""
	parser = argparse.ArgumentParser(description="Index documents and ask questions")
	parser.add_argument("command", choices=("index", "ask"))
	parser.add_argument("value", help="Document path/directory for index, or question for ask")
	parser.add_argument("--db", default="chroma_db", help="Chroma persistence directory")
	parser.add_argument("--collection", default="pdf_documents")
	parser.add_argument("--recursive", action="store_true", help="Include nested documents when indexing")
	parser.add_argument("--reset", action="store_true", help="Rebuild the index; use when source documents or processing change")
	parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
	pipeline = RAGPipeline(persist_directory=args.db, collection_name=args.collection)
	if args.command == "index":
		count = pipeline.index(args.value, recursive=args.recursive, reset=args.reset)
		print(f"Indexed {count} chunk(s).")
	else:
		print(pipeline.ask(args.value, k=args.k))


__all__ = ["RAGPipeline", "main"]
