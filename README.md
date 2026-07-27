# DIY GPU Local

Local data preparation, RAG indexing, continued pretraining, LoRA training, and
chat tooling for Hugging Face causal language models.

The project is tuned for experimenting on local hardware, including consumer
AMD/ROCm and NVIDIA GPUs. The current defaults target a midrange local GPU such
as a 16 GB RTX card, while leaving low-VRAM and larger-GPU knobs exposed.

![Bolt Graphics Zeus](https://raw.githubusercontent.com/exalakeLabs/res/main/images/Bolt-Graphics-Zeus-1456x819-2949710606.png)

## What This Project Does

This repo lets you build a local model workspace from source documents:

1. Download or import raw text.
2. Clean the text into a prepared corpus.
3. Build packed token corpora for continued pretraining.
4. Build instruction/training-pair JSONL files for LoRA/SFT flows.
5. Build a FAISS RAG index over the prepared text.
6. Run continued pretraining or LoRA training.
7. Chat with the base model, trained adapter, RAG index, or adapter plus RAG.

The top-level Zsh launchers are the preferred entrypoints:

```text
install.zsh             Create/update .venv, install backend-specific PyTorch.
pipeline.zsh            Compatibility wrapper for run-train-pipeline.zsh.
run-train-pipeline.zsh  Main workflow runner for corpus, RAG, pretraining, LoRA.
chat.zsh                Runtime launcher with low/high-VRAM GPU profiles.
```

The Python files under `src/*.py` are compatibility wrappers around the package
modules in `src/data_prep`, `src/rag`, `src/training`, and `src/inference`.

## Layout

```text
project-root/
  .env.default       Current environment template used by install.zsh.
  .env.example       Older sample env file, kept for reference.
  eval_prompts.txt   Prompts used before/after continued pretraining.
  prompt_engineer.txt
  install.zsh
  pipeline.zsh      Compatibility wrapper around run-train-pipeline.zsh.
  run-train-pipeline.zsh
  chat.zsh
  src/
    data_prep/       Download, clean, pack token corpus, make training pairs.
    rag/             Chunk text, embed, write FAISS index metadata.
    training/        LoRA/SFT and partial continued-pretraining entrypoints.
    inference/       Chat, RAG, adapter, and runtime helpers.
    utils/           Env, HTTP, PDF/OCR/text helpers.
```

## Quick Start

Install the Python environment for your accelerator. For an NVIDIA card:

```bash
./install.zsh --backend cuda
```

Other supported install backends:

```bash
./install.zsh --backend rocm
./install.zsh --backend mps
```

If you are setting up manually, copy the current template and edit paths/models:

```bash
cp .env.default .env
```

Then run one stage at a time:

```bash
./pipeline.zsh corpus --jobs 4
./pipeline.zsh rag
./pipeline.zsh pretrain
./chat.zsh
```

Or run the main batch flow:

```bash
./pipeline.zsh all --jobs 4
```

`all` expands to `corpus`, `rag`, and `pretrain`.

## Configuration

`pipeline.zsh` is a compatibility wrapper around `run-train-pipeline.zsh`. The
runner loads `.runtime` when present, then loads `.env`. `install.zsh` creates
`.env` from `.env.default` when missing and can prompt for literal defaults.

Important path variables:

```bash
RAWTEXT_DIR=/datasets/raw-text
PREPARED_DIR=/datasets/model_root/prepared
CORPUS_DIR=/datasets/model_root/corpus
RAG_DIR=/datasets/model_root/rag
DEFAULT_OUTPUT_DIR=/datasets/model_root/model/output_partial
```

Important model variables:

```bash
EMBED_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
GENERATOR_MODEL=Qwen/Qwen2.5-3B-Instruct
BASE_MODEL=${GENERATOR_MODEL}
```

Keep `EMBED_MODEL` and `RERANKER_MODEL` on embedding/reranking models. Do not
point them at a generator model. Use `GENERATOR_MODEL` and `BASE_MODEL` for the
chat/training model.

Prompt configuration:

```bash
SYSTEM_PROMPT_FILE=prompt_engineer.txt
CHAT_INCLUDE_EVAL_PROMPTS=0
```

`prompt_engineer.txt` is the main chat system prompt. `SYSTEM_PROMPT` is still
supported as a fallback when no prompt file is set. `eval_prompts.txt` is used by
training and evaluation flows; interactive chat leaves it disabled by default so
normal questions are not over-constrained by retrieval-evaluation instructions.
Set `CHAT_INCLUDE_EVAL_PROMPTS=1` when you explicitly want those eval priorities
appended to chat.

## The Pipeline Runner

`pipeline.zsh` / `run-train-pipeline.zsh` is the main operator interface. It can
run complete workflows, single stages, or one stage with pass-through arguments.

```bash
./pipeline.zsh [commands] [options] [-- extra-args]
```

Commands:

| Command | What It Runs |
| --- | --- |
| `all` | `corpus`, `rag`, then `pretrain` |
| `corpus` | Download raw text, clean it, create packed pretrain corpora, create training pairs |
| `raw-text` | Only download raw text |
| `clean-text` | Only clean raw text into prepared text |
| `pretrain-corpus` | Only create packed token corpora for continued pretraining |
| `pairs` | Only create instruction/training-pair JSONL files |
| `rag` | Build the FAISS RAG index |
| `pretrain` | Run partial continued pretraining |
| `lora` | Run the LoRA training pipeline |
| `amd-monitor` | Run AMD GPU monitoring helper commands |
| `install-gh` | Install/authenticate GitHub CLI helper commands |

Convenience flags:

```bash
./pipeline.zsh --build-corpus
./pipeline.zsh --build-rag
./pipeline.zsh --pretrain
./pipeline.zsh --lora
```

Running `./pipeline.zsh` with no command starts an interactive prompt for the
major stages.

### Corpus Stage

The corpus stage is the data-prep bundle:

```bash
./pipeline.zsh corpus --jobs 4
```

It runs these substeps in order:

1. `raw-text`: download Project Gutenberg and Wikipedia plaintext sources.
2. `clean-text`: normalize raw text into `PREPARED_DIR`.
3. `pretrain-corpus`: pack token sequences into `train.jsonl` and `eval.jsonl`.
4. `pairs`: create training-pair JSONL files.

Useful corpus options:

```bash
./pipeline.zsh corpus --skip-download
./pipeline.zsh corpus --skip-clean
./pipeline.zsh corpus --skip-pretrain-corpus
./pipeline.zsh corpus --skip-pairs
./pipeline.zsh corpus --jobs 8
./pipeline.zsh pretrain-corpus --num-proc 2
```

If you change `DEFAULT_SEQ_LEN`, rebuild the packed corpus:

```bash
./pipeline.zsh pretrain-corpus
```

For quick low-VRAM tests, `DEFAULT_MAX_TRAIN_TOKENS` can cap packed examples at
training time even before you rebuild the JSONL corpus.

This stage is CPU-bound. Hugging Face tokenizers and the dataset `map`/packing
work run on CPU workers, so `nvidia-smi` will normally show little or no GPU
activity. Use `--num-proc` and `DEFAULT_TOKENIZE_BATCH_SIZE` for throughput.
GPU usage is expected during `rag`, `pretrain`, `lora`, and `chat.zsh`.

### CreateCorpusToken

`CreateCorpusToken` is an independent preprocessing step for building a packed
Hugging Face Dataset from cleaned text. It follows the packed-stream pattern:
tokenize with the target model tokenizer, append EOS between documents, split
the stream into fixed-length blocks, and save `input_ids`, `attention_mask`, and
`labels` with `Dataset.save_to_disk`.

Run it directly:

```bash
python3 src/create_corpus_token.py \
  --text_dir "$PREPARED_DIR" \
  --output_dir "$DEFAULT_CORPUS_TOKEN_DIR" \
  --model_name "$DEFAULT_MODEL" \
  --block_size 4096 \
  --num_proc 8 \
  --overwrite
```

Or through the workflow runner:

```bash
./run-train-pipeline.zsh create-corpus-token \
  --token-dir "$DEFAULT_CORPUS_TOKEN_DIR" \
  --block-size 4096 \
  --num-proc 8 \
  -- --overwrite
```

The output directory also includes `create_corpus_token_manifest.json` recording
the tokenizer/model, source directory, block size, and number of packed blocks.

### Inspect Tokenizer Vocabulary

Dump the tokenizer vocabulary used to turn text into token IDs:

```bash
python3 src/dump_tokenizer_vocab.py \
  --model_name "$DEFAULT_MODEL" \
  --output "$CORPUS_DIR/tokenizer_vocab.jsonl"
```

For a quick terminal preview:

```bash
python3 src/dump_tokenizer_vocab.py --limit 50 --format tsv
```

To inspect an OpenAI/tiktoken vocabulary:

```bash
python3 src/dump_tokenizer_vocab.py \
  --backend tiktoken \
  --encoding o200k_base \
  --limit 50 \
  --format tsv
```

Or select the encoding by OpenAI model name:

```bash
python3 src/dump_tokenizer_vocab.py \
  --backend tiktoken \
  --openai-model gpt-4o \
  --output "$CORPUS_DIR/tiktoken_o200k_vocab.jsonl"
```

`CreateCorpusToken` can also use tiktoken for experimental OpenAI-tokenizer
datasets:

```bash
python3 src/create_corpus_token.py \
  --tokenizer_backend tiktoken \
  --tiktoken_encoding o200k_base \
  --text_dir "$PREPARED_DIR" \
  --output_dir "$CORPUS_DIR/tokenized-o200k" \
  --block_size 4096 \
  --overwrite
```

Keep the default Hugging Face backend when creating token IDs for Qwen, Llama,
or other Hugging Face causal language models. Token IDs from tiktoken are from a
different vocabulary and are not interchangeable with those models.

### RAG Stage

Build a FAISS index from `PREPARED_DIR`:

```bash
./pipeline.zsh rag
```

Useful RAG options:

```bash
./pipeline.zsh rag --chunk-size 1800 --overlap 250 --batch-size 32
./pipeline.zsh rag --embed-model BAAI/bge-base-en-v1.5
```

The RAG stage writes:

```text
RAG_DIR/
  index.faiss
  chunks.jsonl
  index_config.json
```

`index_config.json` records the embedding model, so the chat runtime can detect
when the current environment differs from the built index.

### Continued Pretraining Stage

Run partial continued pretraining from the packed token corpus:

```bash
./pipeline.zsh pretrain
```

`pipeline.zsh pretrain` injects a few safer defaults when you have not supplied
the equivalent pass-through argument:

```text
--eval_prompts "$EVAL_PROMPTS"
--corpus_dir "$CORPUS_DIR"
--attention "$CONTINUED_PRETRAIN_ATTENTION"
--device_map "$CONTINUED_PRETRAIN_DEVICE_MAP"
--max_memory "$CONTINUED_PRETRAIN_MAX_MEMORY", when device_map=auto
--optim "$CONTINUED_PRETRAIN_OPTIM"
--mxfp4_dequantize, unless CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=0
```

Pass trainer-specific arguments after `--`:

```bash
./pipeline.zsh pretrain -- --num_train_epochs 0.25
./pipeline.zsh pretrain -- --train_last_n_layers 2 --no-train_lm_head
./pipeline.zsh pretrain -- --corpus_dir "$CORPUS_DIR" --eval_prompts eval_prompts.txt
```

For a 16 GB RTX card, a reasonable starting point is:

```bash
DEFAULT_SEQ_LEN=1024
DEFAULT_MAX_TRAIN_TOKENS=0
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=2
DEFAULT_GRADIENT_ACCUMULATION_STEPS=8
DEFAULT_TRAIN_LAST_N_LAYERS=4
DEFAULT_TRAIN_LM_HEAD=1
DEFAULT_DTYPE=bf16
DEFAULT_ATTENTION=eager
CONTINUED_PRETRAIN_ATTENTION=eager
DEFAULT_DEVICE_MAP=trainable
CONTINUED_PRETRAIN_DEVICE_MAP=trainable
DEFAULT_PROMPT_EVAL=auto
DEFAULT_OPTIM=adamw_torch
DEFAULT_MAX_GRAD_NORM=1.0
CONTINUED_PRETRAIN_MAX_MEMORY=12GiB
CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=1
```

For low-VRAM Radeon cards, use:

```bash
DEFAULT_SEQ_LEN=256
DEFAULT_MAX_TRAIN_TOKENS=256
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=32
DEFAULT_TRAIN_LAST_N_LAYERS=1
DEFAULT_TRAIN_LM_HEAD=0
DEFAULT_DTYPE=bf16
DEFAULT_ATTENTION=eager
CONTINUED_PRETRAIN_ATTENTION=eager
DEFAULT_DEVICE_MAP=trainable
CONTINUED_PRETRAIN_DEVICE_MAP=trainable
DEFAULT_PROMPT_EVAL=auto
DEFAULT_OPTIM=adafactor
DEFAULT_MAX_GRAD_NORM=0
CONTINUED_PRETRAIN_MAX_MEMORY=3GiB
CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=1
```

Why these defaults matter:

- `openai/gpt-oss-*` needs eager attention in this Transformers build.
- `DEFAULT_DEVICE_MAP=trainable` keeps only the GPT-OSS trainable tail on GPU
  and leaves the frozen lower layers on CPU, which is slower but avoids loading
  the full 20B model into VRAM.
- `CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=1` avoids the native MXFP4 CPU/offload
  meta-tensor failure during training.
- `DEFAULT_PROMPT_EVAL=auto` skips prompt-generation eval when the model is
  offloaded; Trainer eval loss still runs.
- BF16 avoids the FP16 AMP GradScaler failure on ROCm/RDNA3.
- TF32 is disabled automatically unless the GPU is NVIDIA Ampere or newer.
- `DEFAULT_TRAIN_LM_HEAD=0` avoids a large optimizer-state allocation.
- `adafactor` avoids Adam's two full moment buffers, which often appear after
  the first optimizer step and can push 8 GB cards over the edge.
- `DEFAULT_MAX_TRAIN_TOKENS` gives an immediate activation-memory cap even when
  the packed corpus was generated at a longer sequence length.

On a larger GPU, increase `DEFAULT_TRAIN_LAST_N_LAYERS`, use a longer
`DEFAULT_SEQ_LEN`, and consider `DEFAULT_DEVICE_MAP=single` only when the full
model fits in VRAM. `DEFAULT_MAX_MEMORY` only applies to `device_map=auto`.

### LoRA Stage

Run the LoRA training pipeline:

```bash
./pipeline.zsh lora
```

Pass LoRA trainer args after `--`:

```bash
./pipeline.zsh lora -- --num-train-epochs 1 --lora-rank 16
```

LoRA consumes the training-pair files generated by the corpus stage. It does not
require a RAG index.

## Chat And Inference

The top-level chat launcher applies runtime profiles for low/high-VRAM machines:

```bash
./chat.zsh
```

It prints the selected generator model, device map, memory cap, dtype, offload
directory, RAG embedder, retrieval count, and PyTorch allocation config before
starting chat.

Useful overrides:

```bash
GENERATOR_DEVICE_MAP=auto ./chat.zsh
GENERATOR_GPU_MEMORY=4GiB ./chat.zsh
GENERATOR_DTYPE=bf16 ./chat.zsh
LOW_VRAM_ROCM_RUNTIME=cpu ./chat.zsh
LOW_VRAM_ROCM_RUNTIME=rocm ./chat.zsh
RAG_EMBED_DEVICE=rocm ./chat.zsh
RAG_STRICT_CONTEXT=1 ./chat.zsh
CHAT_INCLUDE_EVAL_PROMPTS=1 ./chat.zsh
```

RAG chat is retrieval-assisted by default: it uses retrieved passages when they
are relevant, but can answer from general knowledge when retrieval is unrelated
or weak. Set `RAG_STRICT_CONTEXT=1` when you want answers constrained only to
the retrieved local corpus. `CHAT_INCLUDE_EVAL_PROMPTS=1` appends
`eval_prompts.txt` to the system prompt for evaluation-style runs.

Direct Python chat modes:

```bash
# Base model only
python3 src/inference/chat_rag.py --no-rag --no-adapter

# Base model + LoRA adapter
python3 src/inference/chat_rag.py --no-rag

# Base model + RAG
python3 src/inference/chat_rag.py --no-adapter

# Base model + LoRA adapter + RAG
python3 src/inference/chat_rag.py
```

For teaching-style RAG inspection:

```bash
python3 src/inference/teach_gpt_oss_rag.py --question "What does the prepared material say?"
python3 src/inference/teach_gpt_oss_rag.py --dry-run --question "What should I know?"
python3 src/inference/teach_gpt_oss_rag.py --print-teaching-prompt
```

## Guided Python Orchestrator

`src/run_pipeline.py` is a small guided wrapper. It asks which major stages to
run and then calls the same underlying scripts.

```bash
python3 src/run_pipeline.py
python3 src/run_pipeline.py --dry-run
```

Use `pipeline.zsh` for repeatable scripted runs. Use `src/run_pipeline.py` when
you want prompts.

## Direct Module Entrypoints

Most workflows can be called directly when debugging:

```bash
python3 src/data_prep/clean_text.py \
  --input-dir "$RAWTEXT_DIR" \
  --output-dir "$PREPARED_DIR"

python3 src/data_prep/generate_pretrain_corpus.py \
  --text_dir "$PREPARED_DIR" \
  --corpus_dir "$CORPUS_DIR"

python3 src/data_prep/make_training_pairs.py --text-dir "$PREPARED_DIR"
python3 src/rag/index_builder.py --input-dir "$PREPARED_DIR" --output-dir "$RAG_DIR"
python3 src/training/continued_pretrain_partial.py \
  --corpus_dir "$CORPUS_DIR" \
  --eval_prompts eval_prompts.txt

python3 src/training/train_pipeline.py
```

PDF/OCR helpers live under `src/utils`:

```bash
python3 src/utils/extract_pdfs.py --pdf-dir "$PDF_DIR" --text-dir "$RAWTEXT_DIR"
python3 src/utils/pdf_to_txt.py --pdf-dir "$PDF_DIR" --text-dir "$RAWTEXT_DIR"
```

## Troubleshooting

`GptOssForCausalLM does not support ... scaled_dot_product_attention`

Use eager attention:

```bash
export DEFAULT_ATTENTION=eager
export CONTINUED_PRETRAIN_ATTENTION=eager
./pipeline.zsh pretrain
```

CUDA out of memory while loading GPT-OSS weights

`DEFAULT_MAX_MEMORY` does not limit loading when `DEFAULT_DEVICE_MAP=single`.
Use trainable-tail placement for continued pretraining on a 16 GB card:

```bash
export DEFAULT_DEVICE_MAP=trainable
export CONTINUED_PRETRAIN_DEVICE_MAP=trainable
export CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=1
export CONTINUED_PRETRAIN_MAX_MEMORY=12GiB
export DEFAULT_TRAIN_LAST_N_LAYERS=1
export DEFAULT_TRAIN_LM_HEAD=0
./pipeline.zsh pretrain
```

`Tensor on device meta is not on the expected device cuda:0`

Native GPT-OSS MXFP4 plus CPU/offload placement can fail during training. Keep
continued pretraining on the explicit BF16-dequantized path:

```bash
export CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE=1
./pipeline.zsh pretrain
```

`Tensor.item() cannot be called on meta tensors`

This can happen during prompt-generation eval with CPU/offloaded models. Leave
prompt eval on `auto`, or disable it explicitly:

```bash
export DEFAULT_PROMPT_EVAL=off
./pipeline.zsh pretrain
```

`--tf32 requires Ampere or a newer GPU`

TF32 is NVIDIA-specific. The continued-pretraining script now resolves TF32
automatically and disables it on ROCm/AMD.

`Attempting to unscale FP16 gradients`

Use `DEFAULT_DTYPE=bf16` on RDNA3/ROCm. FP16-loaded model parameters and Trainer
FP16 AMP do not mix cleanly with GradScaler.

`No inf checks were recorded for this optimizer`

This usually means no trainable parameters were on the GPU. Keep
`DEFAULT_DEVICE_MAP=trainable` for GPT-OSS partial pretraining, or ensure
`device_map=auto` does not offload the trainable upper layers to CPU.

HIP out of memory on an 8 GB card

Try these in order:

```bash
export DEFAULT_SEQ_LEN=256
export DEFAULT_MAX_TRAIN_TOKENS=256
export DEFAULT_OPTIM=adafactor
export DEFAULT_MAX_GRAD_NORM=0
./pipeline.zsh pretrain-corpus
./pipeline.zsh pretrain
```

For a quick retry without rebuilding the corpus, the runtime cap is enough:

```bash
export DEFAULT_MAX_TRAIN_TOKENS=256
./pipeline.zsh pretrain
```

Then reduce training scope further:

```bash
export DEFAULT_TRAIN_LAST_N_LAYERS=1
export DEFAULT_TRAIN_LM_HEAD=0
```

Embedding or reranker errors mentioning `gpt-oss`

`EMBED_MODEL` and `RERANKER_MODEL` must be embedding/reranking models. Put
generator models in `GENERATOR_MODEL` and `BASE_MODEL`.

## Compatibility Notes

- `.env` is machine-local configuration. `.env.default` is the current template.
- `.runtime` is loaded by the launchers when present.
- `src/*.py` top-level files are compatibility wrappers around package modules.
- RAG indexes are reusable when the embedding model matches `index_config.json`.
- LoRA adapters are reusable only with the base model they were trained against.
