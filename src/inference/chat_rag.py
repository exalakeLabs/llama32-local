#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import torch

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from inference.model_runtime import generate_text, load_generation_model
from rag.rag_model_config import validate_embedding_model, validate_generator_model
from utils.runtime_env import env_file_text, env_int, env_path, env_str

ADAPTER_DIR = env_path("ADAPTER_DIR", "output/lora/final")
BASE_MODEL = env_str("BASE_MODEL")
EMBED_MODEL = env_str("EMBED_MODEL")
EMBED_DEVICE = env_str("RAG_EMBED_DEVICE", "auto")
GENERATOR_MODEL = env_str("GENERATOR_MODEL", BASE_MODEL)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 0)
RAG_DIR = env_path("RAG_DIR", "rag")
RETRIEVE_K = env_int("RETRIEVE_K", 24)
SYSTEM_PROMPT = env_file_text("SYSTEM_PROMPT_FILE", env_str("SYSTEM_PROMPT"))
REQUIRE_ACCELERATOR = env_str("CHAT_REQUIRE_ACCELERATOR")


def bytes_to_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def normalize_device_key(device) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return f"cuda:{0 if device.index is None else device.index}"
        return device.type

    value = str(device).strip().lower()
    if value == "cuda":
        return "cuda:0"
    if value.startswith("cuda:"):
        return value
    if value in {"cpu", "disk", "meta", "mps", "xpu"}:
        return value
    if value.startswith("xpu:"):
        return value
    return value or "unknown"


def device_label(device_key: str) -> str:
    if device_key.startswith("cuda"):
        index = 0
        if ":" in device_key:
            try:
                index = int(device_key.split(":", 1)[1])
            except ValueError:
                index = 0
        backend = "ROCm/HIP" if torch.version.hip is not None else "CUDA"
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            name = torch.cuda.get_device_name(index)
            return f"{device_key} ({backend} GPU: {name})"
        return f"{device_key} ({backend} GPU)"
    if device_key == "disk":
        return "disk offload"
    if device_key == "meta":
        return "meta/offloaded"
    return device_key


def is_gpu_device(device_key: str) -> bool:
    return device_key.startswith("cuda") or device_key.startswith("xpu") or device_key == "mps"


def normalize_accelerator(value: str | None) -> str:
    requested = (value or "").strip().lower()
    if requested in {"", "0", "false", "none", "off", "cpu"}:
        return ""
    if requested in {"gpu", "accelerator"}:
        return "gpu"
    if requested in {"rocm", "hip", "amd", "radeon"}:
        return "rocm"
    if requested in {"cuda", "nvidia"}:
        return "cuda"
    if requested in {"mps", "xpu"}:
        return requested
    raise SystemExit("Unknown required accelerator. Use rocm, cuda, gpu, mps, xpu, or none.")


def device_index(device_key: str) -> int:
    if ":" not in device_key:
        return 0
    try:
        return int(device_key.split(":", 1)[1])
    except ValueError:
        return 0


def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_embed_device(device: str) -> str:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested.startswith(("rocm", "hip", "cuda")):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"Embedding device {requested!r} was requested, but no CUDA/ROCm GPU is visible.\n"
                f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, torch.version.hip={torch.version.hip}\n"
                "If chat.zsh printed 'CUDA visible devices: <none>', the GPU was hidden from Python. "
                "Unset LOW_VRAM_HIDE_GPU or use LOW_VRAM_ROCM_RUNTIME=rocm RAG_EMBED_DEVICE=rocm ./chat.zsh."
            )
        if requested.startswith(("rocm", "hip")) and torch.version.hip is None:
            raise SystemExit(
                f"Embedding device {requested!r} was requested, but this PyTorch build is not ROCm/HIP."
            )
        torch_device = requested.replace("rocm", "cuda", 1).replace("hip", "cuda", 1)
        index = device_index(torch_device)
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"Embedding device {requested!r} was requested, but only "
                f"{torch.cuda.device_count()} CUDA/ROCm device(s) are visible."
            )
        return torch_device

    if requested == "mps":
        if not mps_available():
            raise SystemExit("Embedding device 'mps' was requested, but torch.backends.mps is not available.")
        return "mps"

    if requested.startswith("xpu"):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise SystemExit("Embedding device 'xpu' was requested, but torch.xpu is not available.")
        return requested

    raise SystemExit("Unknown embedding device. Use auto, rocm, rocm:<id>, cuda, cuda:<id>, mps, xpu, or cpu.")


def print_torch_runtime_report() -> None:
    print("\n--- RUNTIME DEVICE CHECK ---")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        value = os.environ.get("CUDA_VISIBLE_DEVICES") or "<empty>"
        print(f"CUDA_VISIBLE_DEVICES: {value}")
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION') or '<empty>'}")
    if torch.cuda.is_available():
        backend = "ROCm/HIP" if torch.version.hip is not None else "CUDA"
        print(f"PyTorch GPU backend: {backend}")
        if torch.version.hip is not None:
            print(f"ROCm HIP version: {torch.version.hip}")
            print("PyTorch exposes ROCm GPUs as cuda:* devices.")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            arch = getattr(props, "gcnArchName", "")
            capability = torch.cuda.get_device_capability(index)
            arch_text = f", arch={arch}" if arch else f", compute capability={capability}"
            print(f"GPU {index}: {props.name}, total VRAM={bytes_to_gib(props.total_memory)}{arch_text}")
        return

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("PyTorch GPU backend: XPU")
        print(f"Visible XPU count: {torch.xpu.device_count()}")
        return

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("PyTorch GPU backend: MPS")
        return

    print("PyTorch GPU backend: none visible")
    print("Runtime placement: CPU fallback")


def validate_required_accelerator(required: str) -> str:
    required = normalize_accelerator(required)
    if not required:
        return ""

    if required == "rocm":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ROCm was required, but PyTorch does not see a CUDA/ROCm device.\n"
                f"torch={torch.__version__}, torch.version.hip={torch.version.hip}\n"
                "Check ROCm, HSA_OVERRIDE_GFX_VERSION, and LOW_VRAM_HIDE_GPU."
            )
        if torch.version.hip is None:
            raise SystemExit(
                "ROCm was required, but this PyTorch build is CUDA/NVIDIA or CPU-only, not HIP/ROCm."
            )
        print("Required accelerator: ROCm/HIP satisfied by cuda:0")
        return required

    if required == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was required, but PyTorch does not see a CUDA GPU.")
        if torch.version.hip is not None:
            raise SystemExit("CUDA/NVIDIA was required, but this PyTorch build is ROCm/HIP.")
        print("Required accelerator: CUDA satisfied by cuda:0")
        return required

    if required == "gpu":
        if torch.cuda.is_available():
            backend = "ROCm/HIP" if torch.version.hip is not None else "CUDA"
            print(f"Required accelerator: GPU satisfied by {backend} cuda:0")
            return required
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("Required accelerator: GPU satisfied by XPU")
            return required
        if mps_available():
            print("Required accelerator: GPU satisfied by MPS")
            return required
        raise SystemExit("A GPU accelerator was required, but PyTorch does not see one.")

    if required == "mps":
        if not mps_available():
            raise SystemExit("MPS was required, but torch.backends.mps is not available.")
        print("Required accelerator: MPS satisfied")
        return required

    if required == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise SystemExit("XPU was required, but torch.xpu is not available.")
        print("Required accelerator: XPU satisfied")
        return required

    return required


def sentence_transformer_device(embedder) -> str:
    for attr in ("device", "_target_device"):
        device = getattr(embedder, attr, None)
        if device is not None:
            return normalize_device_key(device)
    try:
        return normalize_device_key(next(embedder.parameters()).device)
    except StopIteration:
        return "unknown"


def print_embedder_device_report(embedder) -> None:
    device_key = sentence_transformer_device(embedder)
    status = "GPU active" if is_gpu_device(device_key) else "CPU fallback"
    print(f"RAG embedder actual device: {device_label(device_key)} ({status})")


def find_hf_device_map(model):
    seen = set()

    def visit(obj, depth: int):
        if obj is None or depth > 4:
            return None
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        device_map = getattr(obj, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return device_map

        for attr in ("base_model", "model", "module"):
            nested = getattr(obj, attr, None)
            found = visit(nested, depth + 1)
            if found:
                return found
        return None

    return visit(model, 0)


def summarize_parameter_devices(model) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for param in model.parameters():
        key = normalize_device_key(param.device)
        entry = summary.setdefault(key, {"tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["bytes"] += param.numel() * param.element_size()
    return summary


def print_generator_device_report(model) -> None:
    print("\n--- GENERATOR DEVICE PLACEMENT ---")
    device_map = find_hf_device_map(model)
    if device_map:
        counts: dict[str, int] = {}
        for device in device_map.values():
            key = normalize_device_key(device)
            counts[key] = counts.get(key, 0) + 1
        parts = [f"{device_label(key)}: {count} modules" for key, count in sorted(counts.items())]
        print(f"Generator hf_device_map: {', '.join(parts)}")
        if any(is_gpu_device(key) for key in counts):
            print("Generator GPU usage: GPU modules are assigned")
        else:
            print("Generator GPU usage: no GPU modules assigned; CPU/offload fallback")
    else:
        print("Generator hf_device_map: <not reported by model>")

    param_summary = summarize_parameter_devices(model)
    if not param_summary:
        print("Generator parameter devices: <no parameters reported>")
        return

    parts = []
    for key, entry in sorted(param_summary.items()):
        parts.append(f"{device_label(key)}: {entry['tensors']} tensors, {bytes_to_gib(entry['bytes'])}")
    print(f"Generator parameter devices: {', '.join(parts)}")
    if any(is_gpu_device(key) for key in param_summary):
        print("Generator parameter placement: GPU active")
    else:
        print("Generator parameter placement: CPU/offload fallback")


def generator_device_keys(model) -> set[str]:
    keys: set[str] = set()
    device_map = find_hf_device_map(model)
    if device_map:
        keys.update(normalize_device_key(device) for device in device_map.values())
    keys.update(summarize_parameter_devices(model).keys())
    return keys


def validate_generator_accelerator(model, required: str) -> None:
    required = normalize_accelerator(required)
    if not required:
        return

    keys = generator_device_keys(model)
    uses_torch_cuda_device = any(key.startswith("cuda") for key in keys)
    uses_any_gpu = any(is_gpu_device(key) for key in keys)

    if required == "rocm":
        if torch.version.hip is None or not uses_torch_cuda_device:
            placement = ", ".join(sorted(keys)) or "<none>"
            raise SystemExit(
                "Generator was required to use ROCm, but it was not placed on the Radeon GPU.\n"
                f"Observed generator devices: {placement}\n"
                "Use LOW_VRAM_ROCM_RUNTIME=rocm and a generator model that fits the RX 7600."
            )
        return

    if required == "cuda":
        if torch.version.hip is not None or not uses_torch_cuda_device:
            placement = ", ".join(sorted(keys)) or "<none>"
            raise SystemExit(
                "Generator was required to use CUDA/NVIDIA, but it was not placed on a CUDA GPU.\n"
                f"Observed generator devices: {placement}"
            )
        return

    if required == "gpu" and not uses_any_gpu:
        placement = ", ".join(sorted(keys)) or "<none>"
        raise SystemExit(
            "Generator was required to use a GPU accelerator, but it fell back to CPU/offload.\n"
            f"Observed generator devices: {placement}"
        )


def load_chunks(chunks_file: Path):
    rows = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_index_config(rag_dir: Path) -> dict:
    config_file = rag_dir / "index_config.json"
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid RAG index config at {config_file}: {exc}") from exc


def resolve_embed_model(index_config: dict, requested: str) -> str:
    indexed = index_config.get("embed_model")
    if not indexed:
        return requested
    if requested and requested != indexed:
        print(f"RAG index was built with {indexed}; using it instead of requested {requested}.")
    return indexed


def validate_embed_dimension(index, embedder, embed_model: str) -> None:
    index_dim = getattr(index, "d", None)
    if hasattr(embedder, "get_embedding_dimension"):
        embed_dim = embedder.get_embedding_dimension()
    else:
        embed_dim = embedder.get_sentence_embedding_dimension()
    if index_dim is not None and embed_dim is not None and int(index_dim) != int(embed_dim):
        raise SystemExit(
            "\n".join(
                [
                    "RAG index and embedder dimensions do not match.",
                    f"Index dimension: {index_dim}",
                    f"Embedder dimension for {embed_model}: {embed_dim}",
                    "Rebuild the RAG index with ./pipeline.zsh rag, or use the embedder recorded in index_config.json.",
                ]
            )
        )


def retrieve(query, embedder, index, rows, top_k=RETRIEVE_K):
    import numpy as np

    q = embedder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, top_k)

    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = rows[int(idx)]
        row = dict(row)
        row["score"] = float(score)
        hits.append(row)
    return hits


def hit_text(hit: dict) -> str:
    return str(hit.get("text") or hit.get("content") or hit.get("chunk") or "")


def hit_source(hit: dict) -> str:
    metadata = hit.get("metadata")
    if isinstance(metadata, dict):
        nested_source = (
            metadata.get("source")
            or metadata.get("source_relpath")
            or metadata.get("source_path")
            or metadata.get("path")
            or metadata.get("file")
            or metadata.get("title")
        )
        if nested_source:
            return str(nested_source)

    return str(
        hit.get("source")
        or hit.get("source_relpath")
        or hit.get("source_path")
        or hit.get("path")
        or hit.get("file")
        or hit.get("document")
        or hit.get("title")
        or "<unknown source>"
    )


def hit_chunk_id(hit: dict, fallback_index: int) -> str:
    return str(
        hit.get("chunk_id")
        or hit.get("chunk_index")
        or hit.get("id")
        or fallback_index
    )


def format_context(hits, max_chars: int = MAX_CONTEXT_CHARS):
    parts = []
    remaining = max_chars
    for i, hit in enumerate(hits, start=1):
        text = hit_text(hit)
        if max_chars > 0:
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "\n[truncated]"
            remaining -= len(text)
        parts.append(
            f"[Context {i} | source={hit_source(hit)} | chunk={hit_chunk_id(hit, i)} | score={hit['score']:.4f}]\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def trim_history(history: list[dict[str, str]], memory_turns: int) -> list[dict[str, str]]:
    if memory_turns <= 0:
        return []
    return history[-(memory_turns * 2) :]


def retrieval_query_with_history(
    query: str,
    history: list[dict[str, str]],
    memory_turns: int,
) -> str:
    if memory_turns <= 0 or not history:
        return query

    recent = trim_history(history, memory_turns)
    parts = []
    for message in recent:
        role = message["role"].title()
        parts.append(f"{role}: {message['content']}")
    parts.append(f"Current question: {query}")
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a local FAISS RAG index.")
    parser.add_argument("--rag-dir", default=str(RAG_DIR))
    parser.add_argument("--no-rag", action="store_true", help="Run plain chat without loading a RAG index.")
    parser.add_argument("--base-model", "--generator-model", dest="generator_model", default=GENERATOR_MODEL)
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--embed-device", default=EMBED_DEVICE)
    parser.add_argument("--top-k", type=int, default=RETRIEVE_K)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--require-accelerator", default=REQUIRE_ACCELERATOR)
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=4,
        help=(
            "Number of prior user/assistant turns to include in each prompt. "
            "Use 0 to disable chat memory."
        ),
    )
    args = parser.parse_args()

    if args.memory_turns < 0:
        raise SystemExit("--memory-turns must be 0 or greater")

    print_torch_runtime_report()
    required_accelerator = validate_required_accelerator(args.require_accelerator)

    adapter_dir = Path(args.adapter_dir).expanduser()

    validate_generator_model(args.generator_model)
    rows = None
    index = None
    embedder = None

    if args.no_rag:
        print("RAG: disabled")
    else:
        import faiss
        from sentence_transformers import SentenceTransformer

        rag_dir = Path(args.rag_dir).expanduser()
        chunks_file = rag_dir / "chunks.jsonl"
        index_file = rag_dir / "index.faiss"
        index_config = load_index_config(rag_dir)
        rows = load_chunks(chunks_file)
        index = faiss.read_index(str(index_file))
        embed_model = resolve_embed_model(index_config, args.embed_model)
        validate_embedding_model(embed_model)
        embed_device = resolve_embed_device(args.embed_device)
        if embed_device != args.embed_device:
            print(f"Resolved RAG embedder device: requested {args.embed_device!r} -> {embed_device!r}")
        print(f"Loading embedder: {embed_model} on {embed_device}")
        embedder = SentenceTransformer(embed_model, device=embed_device)
        print_embedder_device_report(embedder)
        validate_embed_dimension(index, embedder, embed_model)

    tokenizer, model = load_generation_model(
        base_model=args.generator_model,
        adapter_path=adapter_dir,
        use_adapter=not args.no_adapter and adapter_dir.exists(),
    )
    print_generator_device_report(model)
    validate_generator_accelerator(model, required_accelerator)

    history: list[dict[str, str]] = []
    if args.memory_turns > 0:
        print(f"Chat memory: last {args.memory_turns} turn(s)")
    else:
        print("Chat memory: disabled")

    while True:
        query = input("\nPrompt> ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break

        if args.no_rag:
            user_prompt = query
        else:
            retrieval_query = retrieval_query_with_history(
                query,
                history,
                memory_turns=args.memory_turns,
            )
            hits = retrieve(retrieval_query, embedder, index, rows, top_k=args.top_k)
            context = format_context(hits, max_chars=args.max_context_chars)

            print("\n--- RETRIEVED CONTEXT ---\n")
            for i, hit in enumerate(hits, start=1):
                print(f"{hit_source(hit)} [chunk {hit_chunk_id(hit, i)}] score={hit['score']:.4f}")
            if args.max_context_chars > 0:
                print(f"Context character budget: {args.max_context_chars}")

            user_prompt = (
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Answer using only the context above."
            )

        messages = [
            {"role": "system", "content": args.system_prompt},
            *trim_history(history, args.memory_turns),
            {"role": "user", "content": user_prompt},
        ]

        answer = generate_text(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

        print("\n--- ANSWER ---\n")
        print(answer)
        history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
        )
        history = trim_history(history, args.memory_turns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
