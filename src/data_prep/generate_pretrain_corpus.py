#!/usr/bin/env python3

"""
generate_pretrain_corpus.py

Build a packed token corpus for continued pretraining.

Reads raw .txt files from prepared/ by default, tokenizes them with the target
model tokenizer, packs the token stream into fixed-length examples, and writes:

- corpus/train.jsonl
- corpus/eval.jsonl

Each JSONL row contains an input/label pair:

{"input_ids": [...], "labels": [...]}

Example:

python src/generate_pretrain_corpus.py \
    --text_dir ./prepared \
    --corpus_dir ./corpus \
    --model_name Qwen/Qwen2.5-3B \
    --seq_len 2048 \
    --num_proc 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_float, env_int, env_str

DEFAULT_CORPUS_DIR = env_str("DEFAULT_CORPUS_DIR")
DEFAULT_DATASET_NUM_PROC = env_int("DEFAULT_DATASET_NUM_PROC", 1)
DEFAULT_EVAL_RATIO = env_float("DEFAULT_EVAL_RATIO", 0.01)
DEFAULT_MIN_CHARS = env_int("DEFAULT_MIN_CHARS", 1000)
DEFAULT_MODEL = env_str("DEFAULT_MODEL")
DEFAULT_SEED = env_int("DEFAULT_SEED", 42)
DEFAULT_SEQ_LEN = env_int("DEFAULT_SEQ_LEN", 2048)
DEFAULT_TEXT_DIR = env_str("DEFAULT_TEXT_DIR")
DEFAULT_TOKENIZE_BATCH_SIZE = env_int("DEFAULT_TOKENIZE_BATCH_SIZE", 128)
RAG_DIR = env_str("RAG_DIR")

CHECKPOINT_VERSION = 1


def resolve_num_proc(dataset_num_proc: int) -> int:
    if dataset_num_proc > 0:
        return dataset_num_proc
    return max((os.cpu_count() or 2) // 2, 1)


def print_runtime_note() -> None:
    print(
        "\nRuntime note: generate_pretrain_corpus.py is a CPU tokenizer/dataset "
        "packing step. GPU utilization is not expected here."
    )
    print(
        "Tune throughput with --num_proc and --tokenize_batch_size. GPU usage "
        "shows up in RAG embedding, model training, and chat/inference.\n"
    )


def warn_if_output_overlaps_rag(corpus_dir: Path) -> None:
    if not RAG_DIR:
        return

    try:
        same_dir = corpus_dir.resolve() == Path(RAG_DIR).expanduser().resolve()
    except OSError:
        same_dir = False

    if same_dir:
        print(
            "Warning: --corpus_dir matches RAG_DIR. This script writes "
            "continued-pretraining files such as train.jsonl and eval.jsonl; "
            "the RAG index should usually live in a separate directory."
        )


def load_text_files(text_dir: Path, glob_pattern: str, min_chars: int) -> Dataset:
    texts = []
    paths = sorted(text_dir.rglob(glob_pattern))

    print(f"\nFound {len(paths)} text files under {text_dir}\n")

    for path in paths:
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if len(content) < min_chars:
                continue

            texts.append({"text": content})
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not texts:
        raise SystemExit(f"No usable text files found under {text_dir}")

    return Dataset.from_list(texts)


def source_fingerprint(text_dir: Path, glob_pattern: str, min_chars: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(text_dir.resolve()).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(glob_pattern.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(min_chars).encode("ascii"))

    for path in sorted(text_dir.rglob(glob_pattern)):
        try:
            stat = path.stat()
        except OSError:
            continue

        relative = path.relative_to(text_dir)
        digest.update(b"\0")
        digest.update(str(relative).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))

    return digest.hexdigest()


def checkpoint_config(
    args: argparse.Namespace,
    text_dir: Path,
    source_hash: str,
) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "model_name": args.model_name,
        "text_dir": str(text_dir.resolve()),
        "glob": args.glob,
        "seq_len": args.seq_len,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "min_chars": args.min_chars,
        "source_fingerprint": source_hash,
    }


def checkpoint_stage_dir(checkpoint_dir: Path, stage: str) -> Path:
    return checkpoint_dir / stage


def checkpoint_manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "checkpoint_manifest.json"


def load_checkpoint(
    checkpoint_dir: Optional[Path],
    stage: str,
    config: dict,
) -> Optional[Dataset]:
    if checkpoint_dir is None:
        return None

    stage_dir = checkpoint_stage_dir(checkpoint_dir, stage)
    manifest_path = checkpoint_manifest_path(stage_dir)
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable checkpoint {stage_dir}: {exc}")
        return None

    if manifest.get("config") != config:
        print(f"Ignoring stale checkpoint: {stage_dir}")
        return None

    print(f"Loading checkpoint: {stage_dir}")
    return load_from_disk(str(stage_dir))


def save_checkpoint(
    dataset: Dataset,
    checkpoint_dir: Optional[Path],
    stage: str,
    config: dict,
) -> None:
    if checkpoint_dir is None:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = checkpoint_stage_dir(checkpoint_dir, stage)
    tmp_dir = checkpoint_dir / f".{stage}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    dataset.save_to_disk(str(tmp_dir))
    checkpoint_manifest_path(tmp_dir).write_text(
        json.dumps(
            {
                "stage": stage,
                "config": config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    tmp_dir.rename(stage_dir)


def get_or_create_checkpoint(
    checkpoint_dir: Optional[Path],
    stage: str,
    config: dict,
    build,
) -> Dataset:
    dataset = load_checkpoint(checkpoint_dir, stage, config)
    if dataset is not None:
        return dataset

    dataset = build()
    save_checkpoint(dataset, checkpoint_dir, stage, config)
    return dataset


def tokenize_and_pack_dataset(
    dataset: Dataset,
    tokenizer,
    seq_len: int,
    dataset_num_proc: int,
    tokenize_batch_size: int,
) -> Dataset:
    print("\nTokenizing dataset...\n")
    num_proc = resolve_num_proc(dataset_num_proc)
    print(f"Using num_proc={num_proc}, tokenize_batch_size={tokenize_batch_size}\n")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            return_attention_mask=False,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=["text"],
        num_proc=num_proc,
    )

    print("\nPacking sequences...\n")

    def group_texts(examples):
        input_ids = sum(examples["input_ids"], [])
        total_length = len(input_ids)
        total_length = (total_length // seq_len) * seq_len

        packed_input_ids = [
            input_ids[i : i + seq_len]
            for i in range(0, total_length, seq_len)
        ]

        return {
            "input_ids": packed_input_ids,
            "labels": [tokens.copy() for tokens in packed_input_ids],
        }

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
    )


def split_documents(dataset: Dataset, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return dataset, None

    if len(dataset) < 2:
        print(
            "Only one source document available; validation will be split from packed tokens."
        )
        return dataset, None

    split = dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
    )
    return split["train"], split["test"]


def split_packed_tokens(dataset: Dataset, eval_ratio: float, seed: int):
    if eval_ratio <= 0:
        return dataset, None

    if len(dataset) < 2:
        raise SystemExit(
            "Need at least two packed sequences to create train/eval JSONL files. "
            "Add more text or reduce --seq_len."
        )

    split = dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
    )
    return split["train"], split["test"]


def split_empty_eval_from_train(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    eval_ratio: float,
    seed: int,
):
    if len(eval_dataset) > 0:
        return train_dataset, eval_dataset

    print(
        "Eval document split produced no packed tokens; validation will be split from train tokens."
    )
    return split_packed_tokens(
        train_dataset,
        eval_ratio=eval_ratio,
        seed=seed,
    )


def write_jsonl(path: Path, dataset: Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate packed token JSONL files for continued pretraining."
    )

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--text_dir",
        default=DEFAULT_TEXT_DIR,
        help=f"Directory containing prepared .txt files. Defaults to {DEFAULT_TEXT_DIR}.",
    )

    parser.add_argument(
        "--corpus_dir",
        default=DEFAULT_CORPUS_DIR,
        help=f"Output directory for token JSONL files. Defaults to {DEFAULT_CORPUS_DIR}.",
    )

    parser.add_argument(
        "--train_file",
        default="train.jsonl",
        help="Train JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--eval_file",
        default="eval.jsonl",
        help="Eval JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--glob",
        default="*.txt",
        help='Glob pattern under --text_dir. Defaults to "*.txt".',
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
    )

    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help="Validation split ratio. Defaults to 0.01.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--min_chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Skip text files shorter than this many characters.",
    )

    parser.add_argument(
        "--dataset_num_proc",
        "--num_proc",
        dest="dataset_num_proc",
        type=int,
        default=DEFAULT_DATASET_NUM_PROC,
        help="Tokenization/packing process count. 0 means half of available CPUs.",
    )

    parser.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=DEFAULT_TOKENIZE_BATCH_SIZE,
        help="Documents per tokenizer batch. Lower this if tokenization is killed for memory.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help=(
            "Directory for resumable dataset checkpoints. Defaults to "
            "<corpus_dir>/.generate_pretrain_corpus_checkpoints."
        ),
    )

    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable resumable dataset checkpoints.",
    )

    return parser.parse_args()


def resolve_output_file(corpus_dir: Path, file_arg: str) -> Path:
    path = Path(file_arg).expanduser()
    if not path.is_absolute():
        path = corpus_dir / path
    return path


def resolve_checkpoint_dir(corpus_dir: Path, checkpoint_dir: Optional[str]) -> Path:
    if checkpoint_dir is None:
        return corpus_dir / ".generate_pretrain_corpus_checkpoints"

    path = Path(checkpoint_dir).expanduser()
    if not path.is_absolute():
        path = corpus_dir / path
    return path


def main() -> int:
    args = parse_args()

    text_dir = Path(args.text_dir).expanduser()
    corpus_dir = Path(args.corpus_dir).expanduser()
    train_file = resolve_output_file(corpus_dir, args.train_file)
    eval_file = resolve_output_file(corpus_dir, args.eval_file)
    checkpoint_dir = None
    if not args.no_checkpoints:
        checkpoint_dir = resolve_checkpoint_dir(corpus_dir, args.checkpoint_dir)

    if args.seq_len <= 0:
        raise SystemExit("--seq_len must be greater than 0")
    if args.eval_ratio <= 0 or args.eval_ratio >= 1:
        raise SystemExit("--eval_ratio must be > 0 and < 1")
    if args.tokenize_batch_size <= 0:
        raise SystemExit("--tokenize_batch_size must be greater than 0")

    print_runtime_note()
    warn_if_output_overlaps_rag(corpus_dir)

    source_hash = source_fingerprint(
        text_dir=text_dir,
        glob_pattern=args.glob,
        min_chars=args.min_chars,
    )
    checkpoint_config_data = checkpoint_config(
        args=args,
        text_dir=text_dir,
        source_hash=source_hash,
    )

    if checkpoint_dir is None:
        print("\nDataset checkpoints disabled.\n")
    else:
        print(f"\nDataset checkpoint directory: {checkpoint_dir}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_text_files(
        text_dir=text_dir,
        glob_pattern=args.glob,
        min_chars=args.min_chars,
    )

    train_docs, eval_docs = split_documents(
        raw_dataset,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    train_dataset = get_or_create_checkpoint(
        checkpoint_dir,
        "train_packed",
        checkpoint_config_data,
        lambda: tokenize_and_pack_dataset(
            train_docs,
            tokenizer,
            seq_len=args.seq_len,
            dataset_num_proc=args.dataset_num_proc,
            tokenize_batch_size=args.tokenize_batch_size,
        ),
    )

    if eval_docs is None:
        train_dataset, eval_dataset = split_packed_tokens(
            train_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )
    else:
        eval_dataset = get_or_create_checkpoint(
            checkpoint_dir,
            "eval_packed",
            checkpoint_config_data,
            lambda: tokenize_and_pack_dataset(
                eval_docs,
                tokenizer,
                seq_len=args.seq_len,
                dataset_num_proc=args.dataset_num_proc,
                tokenize_batch_size=args.tokenize_batch_size,
            ),
        )
        train_dataset, eval_dataset = split_empty_eval_from_train(
            train_dataset,
            eval_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )

    if eval_dataset is None:
        raise SystemExit("An eval split is required for continued_pretrain_partial.py.")

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise SystemExit(
            "Token corpus is empty after packing. Add more text or reduce --seq_len."
        )

    write_jsonl(train_file, train_dataset)
    write_jsonl(eval_file, eval_dataset)

    print()
    print(f"Train examples: {len(train_dataset)} -> {train_file}")
    print(f"Eval examples:  {len(eval_dataset)} -> {eval_file}")
    print("\nDONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
