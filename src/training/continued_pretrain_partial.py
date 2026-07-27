#!/usr/bin/env python3

"""
continued_pretrain_partial.py

Optimized continued pretraining script for:
- NVIDIA GPUs ranging from smaller RTX cards to A100-class accelerators
- BF16 tensor-core training
- Partial-layer continued pretraining
- Scientific / technical corpora
- Proper sequence packing
- Validation perplexity tracking
- FlashAttention2
- Fused AdamW
- Tensor-core friendly settings

Recommended install:

pip install \
  torch \
  transformers \
  datasets \
  accelerate \
  sentencepiece \
  evaluate

Optional CUDA acceleration:

pip install flash-attn --no-build-isolation

Step 1, generate packed token JSONL:

python src/generate_pretrain_corpus.py \
    --text_dir ./prepared \
    --corpus_dir ./corpus \
    --seq_len 2048 \
    --num_proc 1

Step 2, train from the packed corpus:

python src/continued_pretrain_partial.py \
    --corpus_dir ./corpus \
    --output_dir ./output_science \
    --eval_prompts ./eval_prompts.txt

Smaller GPU starting point:

python src/continued_pretrain_partial.py \
    --corpus_dir ./corpus \
    --eval_prompts ./eval_prompts.txt \
    --train_last_n_layers 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --attention eager \
    --dataloader_num_workers 2
"""

import argparse
import inspect
import math
from pathlib import Path

import torch

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
    default_data_collator,
    __version__ as TRANSFORMERS_VERSION,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from inference.model_runtime import is_rocm, patch_rocm_grouped_mm, patch_rocm_isin
from utils.runtime_env import env_float, env_int, env_str

DEFAULT_ATTENTION = env_str("DEFAULT_ATTENTION", "auto")
DEFAULT_CORPUS_DIR = env_str("DEFAULT_CORPUS_DIR")
DEFAULT_DATALOADER_NUM_WORKERS = env_int("DEFAULT_DATALOADER_NUM_WORKERS", 8)
DEFAULT_DEVICE_MAP = env_str("DEFAULT_DEVICE_MAP", "auto")
DEFAULT_DTYPE = env_str("DEFAULT_DTYPE", "auto")
DEFAULT_EVAL_FILE = env_str("DEFAULT_EVAL_FILE", "eval.jsonl")
DEFAULT_FLOAT32_MATMUL_PRECISION = env_str("DEFAULT_FLOAT32_MATMUL_PRECISION", "high")
DEFAULT_GRADIENT_ACCUMULATION_STEPS = env_int("DEFAULT_GRADIENT_ACCUMULATION_STEPS", 128)
DEFAULT_LEARNING_RATE = env_float("DEFAULT_LEARNING_RATE", 2e-6)
DEFAULT_LOGGING_STEPS = env_int("DEFAULT_LOGGING_STEPS", 1)
DEFAULT_LR_SCHEDULER_TYPE = env_str("DEFAULT_LR_SCHEDULER_TYPE", "cosine")
DEFAULT_MAX_GRAD_NORM = env_float("DEFAULT_MAX_GRAD_NORM", 1.0)
DEFAULT_MAX_MEMORY = env_str("DEFAULT_MAX_MEMORY", "")
DEFAULT_MAX_TRAIN_TOKENS = env_int("DEFAULT_MAX_TRAIN_TOKENS", 0)
DEFAULT_MODEL = env_str("DEFAULT_MODEL")
DEFAULT_MXFP4_DEQUANTIZE = env_str("DEFAULT_MXFP4_DEQUANTIZE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_NUM_TRAIN_EPOCHS = env_float("DEFAULT_NUM_TRAIN_EPOCHS", 1.0)
DEFAULT_OPTIM = env_str("DEFAULT_OPTIM", "adamw_torch_fused")
DEFAULT_OUTPUT_DIR = env_str("DEFAULT_OUTPUT_DIR")
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = env_int("DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE", 1)
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = env_int("DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE", 8)
DEFAULT_PROMPT_EVAL = env_str("DEFAULT_PROMPT_EVAL", "auto").lower()
DEFAULT_SAVE_STEPS = env_int("DEFAULT_SAVE_STEPS", 250)
DEFAULT_SAVE_TOTAL_LIMIT = env_int("DEFAULT_SAVE_TOTAL_LIMIT", 2)
DEFAULT_TRAIN_FILE = env_str("DEFAULT_TRAIN_FILE", "train.jsonl")
DEFAULT_TRAIN_LAST_N_LAYERS = env_int("DEFAULT_TRAIN_LAST_N_LAYERS", 8)
DEFAULT_TRAIN_LM_HEAD = env_str("DEFAULT_TRAIN_LM_HEAD", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_WARMUP_RATIO = env_float("DEFAULT_WARMUP_RATIO", 0.03)
DEFAULT_WEIGHT_DECAY = env_float("DEFAULT_WEIGHT_DECAY", 0.01)
DEFAULT_CPU_MEMORY = env_str("DEFAULT_CPU_MEMORY", "64GiB")
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 500)

# Good alternatives:
# "mistralai/Mistral-7B-v0.1"
# "meta-llama/Llama-3.2-3B"


def transformers_dtype_kwarg():
    try:
        version = tuple(int(part) for part in TRANSFORMERS_VERSION.split(".")[:2])
    except ValueError:
        return "dtype"

    return "dtype" if version >= (4, 51) else "torch_dtype"


def rocm_supports_bf16(device_id=0):
    if not is_rocm():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(device_id), "gcnArchName", "")
        return any(
            arch.startswith(prefix)
            for prefix in ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101", "gfx1102")
        )
    except Exception:
        return False


def resolve_torch_dtype(dtype_name):
    if dtype_name == "auto":
        if rocm_supports_bf16():
            return torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtypes[dtype_name]


def gpu_supports_tf32(device_id=0):
    if not torch.cuda.is_available() or is_rocm():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device_id)
        return major >= 8
    except Exception:
        return False


def resolve_tf32(requested_tf32, device_id=0):
    if not requested_tf32:
        return False
    if gpu_supports_tf32(device_id):
        return True

    backend = "ROCm/HIP" if is_rocm() else "CUDA"
    print(f"TF32 requested but unsupported on this {backend} device; disabling TF32.")
    return False


def configure_torch_backend(tf32, float32_matmul_precision):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision(float32_matmul_precision)
    print(f"TF32 enabled: {tf32}")
    print(f"float32 matmul precision: {float32_matmul_precision}")


def build_trainable_tail_device_map(model_name, train_last_n_layers, train_lm_head):
    if not torch.cuda.is_available():
        return "auto"

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model_type = getattr(config, "model_type", "")
    total_layers = getattr(config, "num_hidden_layers", None)

    supported_model_types = {"gpt_oss", "llama", "mistral", "qwen2"}
    if model_type not in supported_model_types or total_layers is None:
        print(
            f"trainable device map is not specialized for model_type={model_type!r}; "
            "falling back to device_map=auto."
        )
        return "auto"

    train_last_n_layers = min(max(train_last_n_layers, 0), total_layers)
    freeze_until = total_layers - train_last_n_layers
    device_map = {
        "model.embed_tokens": "cpu",
        "model.norm": 0,
        # Keep logits/loss on GPU; leave the parameters frozen unless requested.
        "lm_head": 0,
    }
    if model_type == "gpt_oss":
        device_map["model.rotary_emb"] = "cpu"

    for idx in range(total_layers):
        device_map[f"model.layers.{idx}"] = 0 if idx >= freeze_until else "cpu"

    print(
        "Using trainable-tail device map: "
        f"{train_last_n_layers} upper layer(s), final norm, and lm_head on GPU; "
        "frozen lower layers on CPU."
    )
    if train_lm_head:
        print("LM head is placed on GPU and will be trainable.")
    else:
        print("LM head is placed on GPU for loss computation but remains frozen.")

    return device_map


def resolve_device_map(device_map, model_name, train_last_n_layers, train_lm_head):
    if device_map == "none":
        return None
    if device_map == "single":
        return {"": 0}
    if device_map == "trainable":
        return build_trainable_tail_device_map(
            model_name,
            train_last_n_layers,
            train_lm_head,
        )
    return device_map


def resolve_max_memory(max_memory, cpu_memory):
    if not max_memory:
        return None
    return {
        0: max_memory,
        "cpu": cpu_memory,
    }


def resolve_corpus_file(corpus_dir, file_arg):
    path = Path(file_arg).expanduser()
    if not path.is_absolute():
        path = corpus_dir / path
    return path


def load_token_corpus(train_file, eval_file):
    train_file = Path(train_file)
    eval_file = Path(eval_file)

    if not train_file.exists():
        raise SystemExit(f"Train corpus file not found: {train_file}")
    if not eval_file.exists():
        raise SystemExit(f"Eval corpus file not found: {eval_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "eval": str(eval_file),
        },
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    if len(train_dataset) == 0:
        raise SystemExit(f"Train corpus is empty: {train_file}")
    if len(eval_dataset) == 0:
        raise SystemExit(f"Eval corpus is empty: {eval_file}")

    validate_token_dataset(train_dataset, "train", train_file)
    validate_token_dataset(eval_dataset, "eval", eval_file)

    return train_dataset, eval_dataset


def validate_token_dataset(dataset, split_name, path):
    example = dataset[0]
    missing = [
        field
        for field in ("input_ids", "labels")
        if field not in example
    ]
    if missing:
        missing_fields = ", ".join(missing)
        raise SystemExit(
            f"{split_name} corpus at {path} is missing required field(s): {missing_fields}"
        )

    if len(example["input_ids"]) != len(example["labels"]):
        raise SystemExit(
            f"{split_name} corpus at {path} has mismatched input_ids and labels lengths."
        )


def sequence_length(dataset):
    if len(dataset) == 0:
        return 0
    return len(dataset[0]["input_ids"])


class TruncatingDataCollator:
    def __init__(self, max_length=0):
        self.max_length = max(0, max_length)

    def __call__(self, features):
        if not self.max_length:
            return default_data_collator(features)

        truncated = []
        for feature in features:
            item = dict(feature)
            for key in ("input_ids", "labels", "attention_mask"):
                value = item.get(key)
                if value is not None and len(value) > self.max_length:
                    item[key] = value[: self.max_length]
            truncated.append(item)

        return default_data_collator(truncated)

# ============================================================
# Model loading
# ============================================================

def load_with_attention(common, attention, allow_eager_fallback=False):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            attn_implementation=attention,
        )
        return model, attention
    except ValueError as e:
        message = str(e)
        unsupported_attention = (
            "does not support an attention implementation" in message
            or (
                "does not support" in message
                and "attn_implementation" in message
            )
        )
        if (
            allow_eager_fallback
            and attention in {"flash_attention_2", "sdpa"}
            and unsupported_attention
        ):
            print(f"{attention} attention is not supported for this model; falling back to eager.")
            model = AutoModelForCausalLM.from_pretrained(
                **common,
                attn_implementation="eager",
            )
            return model, "eager"
        raise


def load_causal_lm(
    model_name,
    dtype,
    attention=DEFAULT_ATTENTION,
    device_map=DEFAULT_DEVICE_MAP,
    max_memory=DEFAULT_MAX_MEMORY,
    cpu_memory=DEFAULT_CPU_MEMORY,
    train_last_n_layers=DEFAULT_TRAIN_LAST_N_LAYERS,
    train_lm_head=DEFAULT_TRAIN_LM_HEAD,
    mxfp4_dequantize=DEFAULT_MXFP4_DEQUANTIZE,
    low_cpu_mem_usage=True,
):

    common = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    common[transformers_dtype_kwarg()] = dtype
    if mxfp4_dequantize:
        common["quantization_config"] = Mxfp4Config(dequantize=True)

    resolved_device_map = resolve_device_map(
        device_map,
        model_name,
        train_last_n_layers,
        train_lm_head,
    )
    if resolved_device_map is not None:
        common["device_map"] = resolved_device_map
        if device_map == "auto":
            resolved_max_memory = resolve_max_memory(max_memory, cpu_memory)
            if resolved_max_memory is not None:
                common["max_memory"] = resolved_max_memory

    if attention == "default":
        print("Using default attention implementation.")
        return AutoModelForCausalLM.from_pretrained(**common)

    if attention == "eager":
        model, _ = load_with_attention(common, "eager")
        print("eager attention enabled.")
        return model

    if attention != "auto":
        model, resolved_attention = load_with_attention(
            common,
            attention,
            allow_eager_fallback=True,
        )
        print(f"{resolved_attention} attention enabled.")
        return model

    try:
        model, _ = load_with_attention(common, "flash_attention_2")
        print("Flash Attention 2 enabled.")
        return model
    except (ImportError, TypeError, ValueError) as e:
        print(f"Flash Attention 2 not available; falling back to SDPA. ({e})")

    try:
        model, _ = load_with_attention(common, "sdpa")
        print("SDPA attention enabled.")
        return model
    except (ImportError, TypeError, ValueError) as e:
        print(f"SDPA not available; falling back to eager attention. ({e})")

    return AutoModelForCausalLM.from_pretrained(
        **common,
        attn_implementation="eager",
    )


# ============================================================
# Freeze lower layers
# ============================================================

def freeze_lower_layers(model, train_last_n_layers=8, train_lm_head=True):

    layers = model.model.layers

    total_layers = len(layers)

    train_last_n_layers = min(max(train_last_n_layers, 0), total_layers)
    freeze_until = total_layers - train_last_n_layers

    print("\n======================================")
    print(f"Total transformer layers: {total_layers}")
    print(f"Training last {train_last_n_layers} layers")
    print(f"Freezing first {freeze_until} layers")
    print("======================================\n")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Train upper transformer blocks
    for idx in range(freeze_until, total_layers):

        for param in layers[idx].parameters():
            param.requires_grad = True

    # Train final norm
    if hasattr(model.model, "norm"):

        for param in model.model.norm.parameters():
            param.requires_grad = True

    if hasattr(model, "lm_head") and train_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True

    # Explicitly keep embeddings frozen
    if hasattr(model.model, "embed_tokens"):

        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {(100 * trainable / total):.2f}%\n")


def print_trainable_device_summary(model):
    by_device = {}
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        count = param.numel()
        total += count
        device = str(param.device)
        by_device[device] = by_device.get(device, 0) + count

    print("Trainable parameter devices:")
    if not by_device:
        print("  <none>")
    else:
        for device, count in sorted(by_device.items()):
            pct = 100 * count / total
            print(f"  {device}: {count:,} ({pct:.1f}%)")

    return by_device


def validate_trainable_device_placement(model):
    by_device = print_trainable_device_summary(model)
    if not by_device:
        raise SystemExit("No trainable parameters remain after layer freezing.")

    has_gpu_trainable = any(device.startswith("cuda") for device in by_device)
    if torch.cuda.is_available() and not has_gpu_trainable:
        raise SystemExit(
            "No trainable parameters are on the GPU. With device_map=auto and a low "
            "max_memory cap, Accelerate can place the frozen lower layers on GPU and "
            "the trainable upper layers on CPU, which breaks AMP optimizer stepping. "
            "For GPT-OSS partial pretraining, use DEFAULT_DEVICE_MAP=trainable. "
            "Otherwise increase DEFAULT_MAX_MEMORY enough to keep trainable layers "
            "on GPU, or reduce trainable state with DEFAULT_TRAIN_LAST_N_LAYERS=1 "
            "and DEFAULT_TRAIN_LM_HEAD=0."
        )

# ============================================================
# Evaluation
# ============================================================

def preferred_input_device(model):
    for param in model.parameters():
        if param.device.type == "cuda":
            return param.device

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def has_meta_parameters(model):
    return any(param.device.type == "meta" for param in model.parameters())


def should_run_prompt_eval(model, prompt_eval_mode, device_map):
    mode = prompt_eval_mode.lower()
    if mode == "on":
        return True

    if mode == "off":
        return False

    if device_map == "trainable":
        print(
            "Prompt generation evaluation skipped: device_map=trainable uses "
            "CPU/offload meta tensors. Trainer eval loss still runs."
        )
        return False

    if has_meta_parameters(model):
        print(
            "Prompt generation evaluation skipped: model has meta/offloaded "
            "parameters. Use --prompt_eval on to force it."
        )
        return False

    return True


@torch.no_grad()
def run_eval(model, tokenizer, prompts, max_new_tokens=MAX_NEW_TOKENS):

    model.eval()
    input_device = preferred_input_device(model)

    print("\n================ EVAL ================\n")

    for prompt in prompts:

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(input_device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        print("=" * 80)
        print(f"PROMPT:\n{prompt}\n")
        print(f"OUTPUT:\n{text}\n")

# ============================================================
# Metrics
# ============================================================

def compute_metrics(eval_pred):

    loss = eval_pred.loss

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "perplexity": perplexity,
    }


# ============================================================
# Training arguments
# ============================================================

def estimate_training_steps(
    num_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
):
    if num_examples <= 0:
        return 0

    effective_batch_size = max(
        1,
        per_device_train_batch_size * gradient_accumulation_steps,
    )
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch_size))
    return max(1, math.ceil(steps_per_epoch * num_train_epochs))


def estimate_warmup_steps(
    num_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
):
    if num_examples <= 0 or warmup_ratio <= 0:
        return 0

    total_steps = estimate_training_steps(
        num_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return max(1, int(total_steps * warmup_ratio))


def make_training_arguments(
    output_dir,
    train_examples,
    num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    lr_scheduler_type=DEFAULT_LR_SCHEDULER_TYPE,
    per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    requested_save_steps=DEFAULT_SAVE_STEPS,
    eval_steps=None,
    save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
    logging_steps=DEFAULT_LOGGING_STEPS,
    dtype=torch.bfloat16,
    tf32=False,
    gradient_checkpointing=True,
    gradient_checkpointing_use_reentrant=False,
    optim=DEFAULT_OPTIM,
    dataloader_num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    max_grad_norm=1.0,
):

    valid = inspect.signature(TrainingArguments.__init__).parameters

    training_steps = estimate_training_steps(
        train_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    warmup_steps = estimate_warmup_steps(
        train_examples,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
    )
    save_steps = max(1, min(requested_save_steps, training_steps))
    eval_steps = save_steps if eval_steps is None else max(1, min(eval_steps, training_steps))
    if optim == "adafactor" and max_grad_norm > 0:
        print("Note: disabling Trainer gradient clipping for Adafactor.")
        max_grad_norm = 0.0
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = False
    if dtype == torch.float16:
        print(
            "Note: model dtype is fp16, so Trainer fp16 AMP is disabled. "
            "Using AMP GradScaler with fp16-loaded parameters raises "
            "'Attempting to unscale FP16 gradients'. Prefer DEFAULT_DTYPE=bf16 "
            "on RDNA3/ROCm."
        )
    dataloader_persistent_workers = (
        dataloader_persistent_workers and dataloader_num_workers > 0
    )

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=tf32,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        report_to="none",
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": gradient_checkpointing_use_reentrant,
        },
        optim=optim,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        max_grad_norm=max_grad_norm,
        remove_unused_columns=False,
    )

    if "overwrite_output_dir" in valid:
        kwargs["overwrite_output_dir"] = False

    if "eval_strategy" in valid:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = "steps"
    else:
        print("Note: this transformers version does not expose an eval strategy argument.")

    if "save_strategy" in valid:
        kwargs["save_strategy"] = "steps"

    unsupported = [
        key
        for key in sorted(kwargs)
        if key not in valid
    ]
    for key in unsupported:
        print(f"Note: this transformers version does not support {key!r}; skipping.")
        del kwargs[key]

    return TrainingArguments(**kwargs)

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--corpus_dir",
        default=DEFAULT_CORPUS_DIR,
        help=f"Directory containing packed token JSONL files. Defaults to {DEFAULT_CORPUS_DIR}.",
    )

    parser.add_argument(
        "--train_file",
        default=DEFAULT_TRAIN_FILE,
        help="Train JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--eval_file",
        default=DEFAULT_EVAL_FILE,
        help="Eval JSONL file name, relative to --corpus_dir unless absolute.",
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Checkpoint and final model output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )

    parser.add_argument(
        "--eval_prompts",
        required=True,
    )

    parser.add_argument(
        "--train_last_n_layers",
        type=int,
        default=DEFAULT_TRAIN_LAST_N_LAYERS,
    )
    parser.add_argument(
        "--train_lm_head",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAIN_LM_HEAD,
        help="Train the LM head. Disable on low-VRAM cards to avoid large optimizer state.",
    )

    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=DEFAULT_DTYPE,
        help=(
            "Model and training dtype. BF16 is preferred on RDNA3/ROCm; "
            "FP16 runs without Trainer AMP scaling."
        ),
    )

    parser.add_argument(
        "--attention",
        choices=("auto", "flash_attention_2", "sdpa", "eager", "default"),
        default=DEFAULT_ATTENTION,
        help="Attention backend. Use eager for GPT-OSS on this Transformers build.",
    )

    parser.add_argument(
        "--device_map",
        choices=("auto", "single", "trainable", "none"),
        default=DEFAULT_DEVICE_MAP,
        help=(
            "'auto' lets Transformers place layers, 'single' forces GPU 0, "
            "'trainable' keeps the GPT-OSS trainable tail on GPU."
        ),
    )
    parser.add_argument(
        "--max_memory",
        default=DEFAULT_MAX_MEMORY,
        help="Optional GPU memory cap for device_map=auto, e.g. 6GiB.",
    )
    parser.add_argument(
        "--cpu_memory",
        default=DEFAULT_CPU_MEMORY,
        help="CPU memory cap paired with --max_memory, e.g. 64GiB.",
    )
    parser.add_argument(
        "--mxfp4_dequantize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MXFP4_DEQUANTIZE,
        help=(
            "Dequantize MXFP4 weights to bf16 while loading. Uses more CPU RAM "
            "but avoids MXFP4 GPU conversion."
        ),
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 on NVIDIA Ampere+ GPUs. Disabled automatically on ROCm/AMD.",
    )

    parser.add_argument(
        "--float32_matmul_precision",
        choices=("highest", "high", "medium"),
        default=DEFAULT_FLOAT32_MATMUL_PRECISION,
    )

    parser.add_argument(
        "--compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
    )

    parser.add_argument(
        "--lr_scheduler_type",
        default=DEFAULT_LR_SCHEDULER_TYPE,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help=(
            "Checkpoint interval in optimizer steps. Small runs are capped to "
            "save at least one checkpoint."
        ),
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation interval in optimizer steps. Defaults to save_steps.",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
    )

    parser.add_argument(
        "--optim",
        default=DEFAULT_OPTIM,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
    )

    parser.add_argument(
        "--dataloader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--dataloader_persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
    )

    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=DEFAULT_MAX_TRAIN_TOKENS,
        help="Runtime token cap for packed train/eval examples. 0 uses full corpus sequences.",
    )

    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--prompt_eval",
        choices=("auto", "on", "off"),
        default=DEFAULT_PROMPT_EVAL,
        help=(
            "Run prompt-generation eval before/after training. 'auto' skips "
            "offloaded trainable-tail models and still keeps Trainer eval loss."
        ),
    )

    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir).expanduser()
    train_file = resolve_corpus_file(corpus_dir, args.train_file)
    eval_file = resolve_corpus_file(corpus_dir, args.eval_file)
    eval_prompts_path = Path(args.eval_prompts).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_torch_dtype(args.dtype)
    tf32 = resolve_tf32(args.tf32)

    configure_torch_backend(
        tf32=tf32,
        float32_matmul_precision=args.float32_matmul_precision,
    )
    patch_rocm_isin()
    patch_rocm_grouped_mm()

    # ========================================================
    # Load packed token corpus
    # ========================================================

    print("\nLoading token corpus...\n")

    train_dataset, eval_dataset = load_token_corpus(
        train_file=train_file,
        eval_file=eval_file,
    )

    print(f"Train corpus: {train_file} ({len(train_dataset)} examples)")
    print(f"Eval corpus:  {eval_file} ({len(eval_dataset)} examples)")
    train_seq_len = sequence_length(train_dataset)
    eval_seq_len = sequence_length(eval_dataset)
    print(f"Train sequence length: {train_seq_len}")
    print(f"Eval sequence length:  {eval_seq_len}")
    if args.max_train_tokens > 0:
        print(f"Runtime train/eval token cap: {args.max_train_tokens}")
        if train_seq_len > args.max_train_tokens or eval_seq_len > args.max_train_tokens:
            print(
                "Packed examples will be truncated by the data collator. "
                "Rebuild with DEFAULT_SEQ_LEN set to this value for cleaner packing."
            )

    # ========================================================
    # Load tokenizer
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # Load model
    # ========================================================

    print("\nLoading model...\n")

    print(f"Requested dtype: {args.dtype}")
    print(f"Resolved dtype: {dtype}")
    print(f"Attention: {args.attention}")
    print(f"Device map: {args.device_map}")
    if args.max_memory:
        if args.device_map == "auto":
            print(f"Max memory: GPU 0={args.max_memory}, CPU={args.cpu_memory}")
        else:
            print(
                f"Max memory: {args.max_memory} "
                f"(ignored with device_map={args.device_map}; only auto uses it)"
            )
    print(f"MXFP4 dequantize: {args.mxfp4_dequantize}")

    model = load_causal_lm(
        args.model_name,
        dtype=dtype,
        attention=args.attention,
        device_map=args.device_map,
        max_memory=args.max_memory,
        cpu_memory=args.cpu_memory,
        train_last_n_layers=args.train_last_n_layers,
        train_lm_head=args.train_lm_head,
        mxfp4_dequantize=args.mxfp4_dequantize,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    if args.gradient_checkpointing:
        # Required with gradient checkpointing
        model.config.use_cache = False

    if args.compile_model:
        print("\nCompiling model...\n")

        model = torch.compile(model)

    # ========================================================
    # Freeze lower layers
    # ========================================================

    freeze_lower_layers(
        model,
        train_last_n_layers=args.train_last_n_layers,
        train_lm_head=args.train_lm_head,
    )
    print(f"LM head trainable: {args.train_lm_head}")
    validate_trainable_device_placement(model)

    # ========================================================
    # Eval prompts
    # ========================================================

    prompts = eval_prompts_path.read_text().splitlines()

    prompts = [
        p.strip()
        for p in prompts
        if p.strip()
    ]
    run_prompt_eval = should_run_prompt_eval(
        model,
        args.prompt_eval,
        args.device_map,
    )

    # ========================================================
    # Baseline evaluation
    # ========================================================

    if run_prompt_eval:
        print("\nRUNNING BASELINE EVALUATION\n")

        run_eval(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.eval_max_new_tokens,
        )
    else:
        print("\nSKIPPING BASELINE PROMPT EVALUATION\n")

    # ========================================================
    # Training arguments
    # ========================================================

    training_args = make_training_arguments(
        output_dir,
        train_examples=len(train_dataset),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        requested_save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        dtype=dtype,
        tf32=tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_use_reentrant=args.gradient_checkpointing_use_reentrant,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        max_grad_norm=args.max_grad_norm,
    )

    print(f"\nCheckpoint/output directory: {output_dir}\n")
    print(f"Corpus directory: {corpus_dir}")
    print(f"Train batch size per device: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Trainable upper layers: {args.train_last_n_layers}")
    print(f"Train LM head: {args.train_lm_head}")
    print(f"Max train tokens: {args.max_train_tokens or '<full sequence>'}")
    print(f"Optimizer: {args.optim}")
    print(f"Max grad norm: {training_args.max_grad_norm}")

    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        data_collator=TruncatingDataCollator(args.max_train_tokens),
    )

    # ========================================================
    # Train
    # ========================================================

    print("\nSTARTING CONTINUED PRETRAINING\n")

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No valid checkpoint found. Starting fresh.")
        trainer.train()

    # ========================================================
    # Final evaluation
    # ========================================================

    if run_prompt_eval:
        print("\nRUNNING FINAL EVALUATION\n")

        run_eval(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.eval_max_new_tokens,
        )
    else:
        print("\nSKIPPING FINAL PROMPT EVALUATION\n")

    # ========================================================
    # Save model
    # ========================================================

    print("\nSaving model...\n")

    trainer.save_model(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))

    print("\nDONE\n")

if __name__ == "__main__":
    main()
