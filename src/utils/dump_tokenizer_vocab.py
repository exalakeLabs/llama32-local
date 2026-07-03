#!/usr/bin/env python3

"""
Dump a tokenizer vocabulary for quick inspection.

Examples:

python src/dump_tokenizer_vocab.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output vocab.jsonl

python src/dump_tokenizer_vocab.py \
    --backend tiktoken \
    --encoding o200k_base \
    --output o200k_vocab.jsonl

python src/dump_tokenizer_vocab.py --limit 50 --format tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_str

DEFAULT_MODEL = env_str("DEFAULT_MODEL", env_str("BASE_MODEL", env_str("GENERATOR_MODEL")))


def hf_token_text(tokenizer, token_id: int) -> str:
    return tokenizer.convert_tokens_to_string(
        [tokenizer.convert_ids_to_tokens(token_id)]
    )


def hf_vocab_rows(tokenizer, sort_by: str):
    vocab = tokenizer.get_vocab()
    rows = [
        {
            "id": token_id,
            "token": token,
            "text": hf_token_text(tokenizer, token_id),
        }
        for token, token_id in vocab.items()
    ]

    if sort_by == "token":
        return sorted(rows, key=lambda row: row["token"])
    return sorted(rows, key=lambda row: row["id"])


def load_hf_tokenizer(model_name: str, no_fast: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=not no_fast,
    )


def load_tiktoken_encoding(encoding: str, openai_model: str):
    import tiktoken

    if openai_model:
        return tiktoken.encoding_for_model(openai_model)
    return tiktoken.get_encoding(encoding)


def tiktoken_vocab_rows(encoding, sort_by: str):
    mergeable_ranks = getattr(encoding, "_mergeable_ranks", {})
    special_tokens = getattr(encoding, "_special_tokens", {})
    rows = []

    for token_bytes, token_id in mergeable_ranks.items():
        text = token_bytes.decode("utf-8", errors="replace")
        rows.append(
            {
                "id": token_id,
                "token": text,
                "text": text,
                "token_bytes_hex": token_bytes.hex(),
                "special": False,
            }
        )

    for token, token_id in special_tokens.items():
        rows.append(
            {
                "id": token_id,
                "token": token,
                "text": token,
                "token_bytes_hex": "",
                "special": True,
            }
        )

    if sort_by == "token":
        return sorted(rows, key=lambda row: row["token"])
    return sorted(rows, key=lambda row: row["id"])


def write_jsonl(rows, output):
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(rows, output):
    output.write("id\ttoken\ttext\tspecial\ttoken_bytes_hex\n")
    for row in rows:
        token = row["token"].replace("\t", "\\t").replace("\n", "\\n")
        text = row["text"].replace("\t", "\\t").replace("\n", "\\n")
        special = row.get("special", "")
        token_bytes_hex = row.get("token_bytes_hex", "")
        output.write(f"{row['id']}\t{token}\t{text}\t{special}\t{token_bytes_hex}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a tokenizer vocabulary as JSONL or TSV."
    )
    parser.add_argument(
        "--backend",
        choices=("hf", "tiktoken"),
        default="hf",
        help="Tokenizer backend. Use hf for Hugging Face models, tiktoken for OpenAI encodings.",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
        help=f"Hugging Face tokenizer/model name or path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help='tiktoken encoding name. Default: "o200k_base".',
    )
    parser.add_argument(
        "--openai-model",
        default="",
        help='Use tiktoken.encoding_for_model, e.g. "gpt-4o". Overrides --encoding.',
    )
    parser.add_argument(
        "--output",
        default="-",
        help='Output path, or "-" for stdout. Default: stdout.',
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "tsv"),
        default="jsonl",
        help="Dump format. Default: jsonl.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("id", "token"),
        default="id",
        help="Vocabulary sort order. Default: id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only dump the first N rows after sorting. 0 means all rows.",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Use the slow Hugging Face tokenizer implementation if available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.backend == "hf" and not args.model_name:
        raise SystemExit("--model_name is required, or set DEFAULT_MODEL/BASE_MODEL.")
    if args.limit < 0:
        raise SystemExit("--limit must be 0 or greater")

    if args.backend == "tiktoken":
        encoding = load_tiktoken_encoding(args.encoding, args.openai_model)
        rows = tiktoken_vocab_rows(encoding, sort_by=args.sort_by)
    else:
        tokenizer = load_hf_tokenizer(args.model_name, no_fast=args.no_fast)
        rows = hf_vocab_rows(tokenizer, sort_by=args.sort_by)
    if args.limit:
        rows = rows[: args.limit]

    if args.output == "-":
        output = sys.stdout
        close_output = False
    else:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        output = path.open("w", encoding="utf-8")
        close_output = True

    try:
        if args.format == "tsv":
            write_tsv(rows, output)
        else:
            write_jsonl(rows, output)
    finally:
        if close_output:
            output.close()

    if args.output != "-":
        print(f"Wrote {len(rows)} vocab rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
