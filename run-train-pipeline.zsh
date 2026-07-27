#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

COMMANDS=()
PASSTHROUGH_ARGS=()
RAW_TEXT_OUTPUT_DIR=""
PREPARED_OUTPUT_DIR=""
CORPUS_OUTPUT_DIR=""
RAG_OUTPUT_DIR=""
CORPUS_TOKEN_OUTPUT_DIR=""
RUN_GUTENBERG=1
RUN_WIKIPEDIA=1
RUN_HTML=0
RUN_DOWNLOAD=1
RUN_CLEAN=1
RUN_PRETRAIN_CORPUS=1
RUN_PAIRS=1
RAW_TEXT_JOBS="${RAW_TEXT_JOBS:-4}"
WIKI_CRAWL_DEPTH="${WIKI_CRAWL_DEPTH:-1}"
WIKI_CRAWL_MAX_PAGES="${WIKI_CRAWL_MAX_PAGES:-240}"
WIKI_CRAWL_LINKS_PER_PAGE="${WIKI_CRAWL_LINKS_PER_PAGE:-35}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-}"
CORPUS_TOKEN_BLOCK_SIZE="${CORPUS_TOKEN_BLOCK_SIZE:-}"
TRAINING_PAIRS_TRAIN="${TRAINING_PAIRS_TRAIN:-}"
TRAINING_PAIRS_VAL="${TRAINING_PAIRS_VAL:-}"

usage() {
  cat <<'EOF'
Usage: ./pipeline.zsh [commands] [options] [-- extra-args]
       ./run-train-pipeline.zsh [commands] [options] [-- extra-args]

Commands:
  all                 Run corpus, rag, and pretrain.
  corpus              Download raw text, clean text, create pretrain corpus, and create pairs.
  raw-text            Only download raw text.
  clean-text          Only clean raw text into prepared text.
  pretrain-corpus     Only create packed token corpora for continued pretraining.
  create-corpus-token Create a packed Hugging Face token Dataset independently.
  pairs               Only create instruction/training pairs.
  rag                 Build the RAG index.
  pretrain            Run continued pretraining.
  lora                Run the LoRA train pipeline.
  install-gh          Install and authenticate the GitHub CLI helper from the old script.
  amd-monitor         Run the AMD monitoring commands from the old helper.

Convenience options:
  --all               Same as command: all.
  --build-corpus      Same as command: corpus.
  --build-rag         Same as command: rag.
  --pretrain          Same as command: pretrain.
  --lora              Same as command: lora.

Corpus/download options:
  --raw-dir DIR          Raw text directory (default: RAWTEXT_DIR or text).
  --output-dir DIR       Alias for --raw-dir, kept for old raw-text usage.
  --prepared-dir DIR     Cleaned text directory (default: PREPARED_DIR or prepared).
  --corpus-dir DIR       Corpus output directory (default: CORPUS_DIR or corpus).
  --token-dir DIR        CreateCorpusToken output directory (default: DEFAULT_CORPUS_TOKEN_DIR).
  --block-size N         CreateCorpusToken block size (default: DEFAULT_SEQ_LEN).
  --skip-download        Do not download raw text during the corpus command.
  --skip-clean           Do not clean text during the corpus command.
  --skip-pretrain-corpus Do not create packed token corpora during the corpus command.
  --skip-pairs           Do not create training pairs during the corpus command.
  --skip-gutenberg      Do not download Project Gutenberg books.
  --skip-wikipedia      Do not download Wikipedia plaintext pages.
  --run-html            Download and convert URLs listed in HTML_URLS.
  --jobs N              Run up to N downloader commands in parallel (default: RAW_TEXT_JOBS or 4).
  --wiki-crawl-depth N  Wikipedia link depth from each seed page (default: 1).
  --wiki-crawl-max N    Max crawled Wikipedia pages per topic crawl (default: 240).
  --wiki-crawl-links N  Max linked pages enqueued per crawled page (default: 35).
  --num-proc N          Worker processes for packed pretrain corpus generation.

RAG options:
  --rag-dir DIR         RAG index output directory (default: RAG_DIR or rag).
  --embed-model MODEL   Embedding model for the RAG index.
  --chunk-size N        Chunk size in characters (default: CHUNK_SIZE_CHARS or 1800).
  --overlap N           Chunk overlap in characters (default: OVERLAP_CHARS or 250).
  --batch-size N        Embedding batch size (default: BATCH_SIZE or 32).

Pass-through:
  -- extra-args         Extra args for a single pretrain or lora command.

Other:
  -h, --help            Show this help.

Interactive mode:
  Running ./pipeline.zsh or ./run-train-pipeline.zsh with no command prompts for these stages:
    1. Download content.
    2. Create/re-create corpus from cleaned text.
    3. Build the RAG index.
    4. Pre-train the model from the generated token corpus.

Environment:
  RAWTEXT_DIR           Default output directory, usually set by .env/.runtime.
  PREPARED_DIR          Default cleaned text directory.
  CORPUS_DIR            Default pretrain/pair corpus directory.
  RAG_DIR               Default RAG index directory.
  RAW_TEXT_JOBS         Default parallel downloader command count.
  PYTHON                Python executable to use after .runtime is loaded.
EOF
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  local answer suffix

  if [[ "$default" == "y" ]]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi

  while true; do
    printf "%s %s " "$prompt" "$suffix"
    if ! read -r answer; then
      answer=""
    fi
    answer="${answer:l}"

    if [[ -z "$answer" ]]; then
      answer="$default"
    fi

    case "$answer" in
      y|yes)
        return 0
        ;;
      n|no)
        return 1
        ;;
      *)
        print "Please answer y or n."
        ;;
    esac
  done
}

prompt_pipeline_commands() {
  print "Select pipeline stages to run:"
  print "  1. Download content into raw text."
  print "  2. Create/re-create corpus from cleaned text."
  print "  3. Build the RAG index."
  print "  4. Pre-train the model from generated token corpus."
  print

  if ask_yes_no "1. Download content?" y; then
    COMMANDS+=(raw-text)
  fi

  if ask_yes_no "2. Create/re-create corpus from cleaned text?" y; then
    COMMANDS+=(clean-text pretrain-corpus)
  fi

  if ask_yes_no "3. Build the RAG index?" y; then
    COMMANDS+=(rag)
  fi

  if ask_yes_no "4. Pre-train the model from generated tokens?" n; then
    COMMANDS+=(pretrain)
  fi

  if [[ "${#COMMANDS[@]}" -eq 0 ]]; then
    print "No stages selected; exiting."
    exit 0
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    all|corpus|raw-text|clean-text|pretrain-corpus|create-corpus-token|pairs|rag|pretrain|lora|install-gh|amd-monitor)
      COMMANDS+=("$1")
      shift
      ;;
    --all)
      COMMANDS+=(all)
      shift
      ;;
    --build-corpus)
      COMMANDS+=(corpus)
      shift
      ;;
    --build-rag)
      COMMANDS+=(rag)
      shift
      ;;
    --pretrain)
      COMMANDS+=(pretrain)
      shift
      ;;
    --lora)
      COMMANDS+=(lora)
      shift
      ;;
    --)
      shift
      PASSTHROUGH_ARGS=("$@")
      break
      ;;
    --raw-dir)
      RAW_TEXT_OUTPUT_DIR="${2:?missing directory for --raw-dir}"
      shift 2
      ;;
    --output-dir)
      RAW_TEXT_OUTPUT_DIR="${2:?missing directory for --output-dir}"
      shift 2
      ;;
    --prepared-dir)
      PREPARED_OUTPUT_DIR="${2:?missing directory for --prepared-dir}"
      shift 2
      ;;
    --corpus-dir)
      CORPUS_OUTPUT_DIR="${2:?missing directory for --corpus-dir}"
      shift 2
      ;;
    --token-dir)
      CORPUS_TOKEN_OUTPUT_DIR="${2:?missing directory for --token-dir}"
      shift 2
      ;;
    --block-size)
      CORPUS_TOKEN_BLOCK_SIZE="${2:?missing value for --block-size}"
      shift 2
      ;;
    --rag-dir)
      RAG_OUTPUT_DIR="${2:?missing directory for --rag-dir}"
      shift 2
      ;;
    --skip-download)
      RUN_DOWNLOAD=0
      shift
      ;;
    --skip-clean)
      RUN_CLEAN=0
      shift
      ;;
    --skip-pretrain-corpus)
      RUN_PRETRAIN_CORPUS=0
      shift
      ;;
    --skip-pairs)
      RUN_PAIRS=0
      shift
      ;;
    --skip-gutenberg)
      RUN_GUTENBERG=0
      shift
      ;;
    --skip-wikipedia)
      RUN_WIKIPEDIA=0
      shift
      ;;
    --run-html)
      RUN_HTML=1
      shift
      ;;
    --jobs)
      RAW_TEXT_JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --wiki-crawl-depth)
      WIKI_CRAWL_DEPTH="${2:?missing value for --wiki-crawl-depth}"
      shift 2
      ;;
    --wiki-crawl-max)
      WIKI_CRAWL_MAX_PAGES="${2:?missing value for --wiki-crawl-max}"
      shift 2
      ;;
    --wiki-crawl-links)
      WIKI_CRAWL_LINKS_PER_PAGE="${2:?missing value for --wiki-crawl-links}"
      shift 2
      ;;
    --num-proc)
      DATASET_NUM_PROC="${2:?missing value for --num-proc}"
      shift 2
      ;;
    --embed-model)
      export EMBED_MODEL="${2:?missing value for --embed-model}"
      shift 2
      ;;
    --chunk-size)
      export CHUNK_SIZE_CHARS="${2:?missing value for --chunk-size}"
      shift 2
      ;;
    --overlap)
      export OVERLAP_CHARS="${2:?missing value for --overlap}"
      shift 2
      ;;
    --batch-size)
      export BATCH_SIZE="${2:?missing value for --batch-size}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "error: unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${#COMMANDS[@]}" -eq 0 ]]; then
  if [[ -t 0 ]]; then
    prompt_pipeline_commands
  else
    COMMANDS=(all)
  fi
fi

if [[ -f "$ROOT/.runtime" ]]; then
  if ! source "$ROOT/.runtime" >/dev/null 2>/dev/null; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
      PYTHON="$ROOT/.venv/bin/python"
    fi
    set -a
    [[ -f "$ROOT/.env.default" ]] && source "$ROOT/.env.default"
    [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
    set +a
  fi
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi
if [[ -f "$ROOT/.env" && -z "${DEFAULT_MODEL:-}" ]]; then
  set -a
  [[ -f "$ROOT/.env.default" ]] && source "$ROOT/.env.default"
  source "$ROOT/.env"
  set +a
elif [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-python3}"
RAW_TEXT_OUTPUT_DIR="${RAW_TEXT_OUTPUT_DIR:-${RAWTEXT_DIR:-text}}"
PREPARED_OUTPUT_DIR="${PREPARED_OUTPUT_DIR:-${PREPARED_DIR:-prepared}}"
CORPUS_OUTPUT_DIR="${CORPUS_OUTPUT_DIR:-${CORPUS_DIR:-corpus}}"
CORPUS_TOKEN_OUTPUT_DIR="${CORPUS_TOKEN_OUTPUT_DIR:-${DEFAULT_CORPUS_TOKEN_DIR:-$CORPUS_OUTPUT_DIR/tokenized}}"
RAG_OUTPUT_DIR="${RAG_OUTPUT_DIR:-${RAG_DIR:-rag}}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-${DEFAULT_DATASET_NUM_PROC:-1}}"
CORPUS_TOKEN_BLOCK_SIZE="${CORPUS_TOKEN_BLOCK_SIZE:-${DEFAULT_SEQ_LEN:-2048}}"
TRAINING_PAIRS_TRAIN="${TRAINING_PAIRS_TRAIN:-$CORPUS_OUTPUT_DIR/training_pairs_train.jsonl}"
TRAINING_PAIRS_VAL="${TRAINING_PAIRS_VAL:-$CORPUS_OUTPUT_DIR/training_pairs_val.jsonl}"

if [[ ! "$RAW_TEXT_JOBS" == <-> ]]; then
  print -u2 "error: --jobs / RAW_TEXT_JOBS must be a positive integer"
  exit 2
fi
if (( RAW_TEXT_JOBS < 1 )); then
  print -u2 "error: --jobs / RAW_TEXT_JOBS must be a positive integer"
  exit 2
fi

GUTENBERG_TASKS=(
  "science||400"
  "mathematics||250"
  "statistics||120"
  "economics||180"
  "finance||100"
  "banking||100"
  "commerce||120"
  "law||150"
  "government||100"
  "engineering||180"
  "logic||100"
  "philosophy||150"
  "history|Modern|120"
  "biography||150"
)

WIKIPEDIA_TITLES=(
  "Artificial intelligence"
  "Machine learning"
  "Natural language processing"
  "Information retrieval"
  "Statistics"
  "Economics"
  "Computer science"
  "Software engineering"
  "Data science"
  "Probability theory"
  "Linear algebra"
  "Optimization"
  "Algorithms"
  "Database"
  "Distributed computing"
  "Cybersecurity"
  "Chemical compound"
  "Chemistry"
  "Organic chemistry"
  "Physical chemistry"
  "Biochemistry"
  "Polymer chemistry"
  "Medicinal chemistry"
  "Quantum mechanics"
  "Quantum field theory"
  "Particle physics"
  "Condensed matter physics"
  "Atomic physics"
  "Optics"
  "Automotive engineering"
  "Automobile"
  "Internal combustion engine"
  "Electric vehicle"
  "Hybrid vehicle"
  "Vehicle dynamics"
  "Automotive industry"
)

WIKIPEDIA_CRAWL_SEEDS=(
  "organic|Organic chemistry|Functional group|Organic reaction|Stereochemistry|Aromaticity|Polymer chemistry|Medicinal chemistry"
  "quantum|Quantum mechanics|Quantum physics|Quantum field theory|Particle physics|Condensed matter physics|Atomic physics|Quantum information"
  "automotive|Automotive engineering|Automobile|Internal combustion engine|Electric vehicle|Hybrid vehicle|Vehicle dynamics|Automotive industry"
)

ORGANIC_CRAWL_TERMS=(
  "organic"
  "chem"
  "reaction"
  "compound"
  "carbon"
  "hydrocarbon"
  "functional"
  "synthesis"
  "polymer"
  "aromatic"
  "stereo"
)

QUANTUM_CRAWL_TERMS=(
  "quantum"
  "particle"
  "field"
  "atom"
  "atomic"
  "photon"
  "electron"
  "wave"
  "physics"
  "mechanics"
)

AUTOMOTIVE_CRAWL_TERMS=(
  "automotive"
  "automobile"
  "vehicle"
  "engine"
  "motor"
  "car"
  "transmission"
  "brake"
  "tire"
  "electric"
  "hybrid"
)

HTML_URLS=(
  "https://en.wikipedia.org/wiki/Artificial_intelligence"
  "https://en.wikipedia.org/wiki/Natural_language_processing"
)

typeset -i ACTIVE_JOBS=0
typeset -i JOB_FAILED=0
typeset -a JOB_PIDS=()

run_cmd() {
  print
  print "==> $*"
  "$@"
}

run_cmd_async() {
  print
  print "==> $*"
  "$@" &
  JOB_PIDS+=("$!")
  ACTIVE_JOBS="${#JOB_PIDS[@]}"
  if (( ACTIVE_JOBS >= RAW_TEXT_JOBS )); then
    wait_for_all
  fi
}

wait_for_all() {
  local pid

  for pid in "${JOB_PIDS[@]}"; do
    if ! wait "$pid"; then
      JOB_FAILED=1
    fi
  done
  JOB_PIDS=()
  ACTIVE_JOBS=0

  if (( JOB_FAILED )); then
    print -u2 "error: one or more downloader jobs failed"
    exit 1
  fi
}

download_gutenberg() {
  local spec query rest topic max_books
  local -a cmd

  for spec in "${GUTENBERG_TASKS[@]}"; do
    query="${spec%%|*}"
    rest="${spec#*|}"
    topic="${rest%%|*}"
    max_books="${rest##*|}"

    cmd=(
      "$PYTHON"
      "$ROOT/src/data_prep/download_gutenberg.py"
      --query "$query"
      --max-books "$max_books"
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
    )

    if [[ -n "$topic" ]]; then
      cmd+=(--topic "$topic")
    fi

    run_cmd_async "${cmd[@]}"
  done

  wait_for_all
}

download_wikipedia() {
  local title
  local group label rest seed term
  local -a cmd seeds terms

  for title in "${WIKIPEDIA_TITLES[@]}"; do
    run_cmd_async \
      "$PYTHON" \
      "$ROOT/src/data_prep/download_web_text.py" \
      --wikipedia-title "$title" \
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
  done

  wait_for_all

  for group in "${WIKIPEDIA_CRAWL_SEEDS[@]}"; do
    label="${group%%|*}"
    rest="${group#*|}"
    seeds=("${(@ps:|:)rest}")

    case "$label" in
      organic)
        terms=("${ORGANIC_CRAWL_TERMS[@]}")
        ;;
      quantum)
        terms=("${QUANTUM_CRAWL_TERMS[@]}")
        ;;
      automotive)
        terms=("${AUTOMOTIVE_CRAWL_TERMS[@]}")
        ;;
      *)
        terms=()
        ;;
    esac

    cmd=(
      "$PYTHON"
      "$ROOT/src/data_prep/download_web_text.py"
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
      --crawl-depth "$WIKI_CRAWL_DEPTH"
      --crawl-max-pages "$WIKI_CRAWL_MAX_PAGES"
      --crawl-links-per-page "$WIKI_CRAWL_LINKS_PER_PAGE"
    )

    for seed in "${seeds[@]}"; do
      cmd+=(--wikipedia-crawl-title "$seed")
    done

    for term in "${terms[@]}"; do
      cmd+=(--crawl-include "$term")
    done

    run_cmd_async "${cmd[@]}"
  done

  wait_for_all
}

download_html() {
  local url

  for url in "${HTML_URLS[@]}"; do
    run_cmd_async \
      "$PYTHON" \
      "$ROOT/src/data_prep/download_web_text.py" \
      --url "$url" \
      --output-dir "$RAW_TEXT_OUTPUT_DIR"
  done

  wait_for_all
}

has_arg() {
  local name="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$name" || "$arg" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

resolve_pretrain_attention() {
  local attention="${CONTINUED_PRETRAIN_ATTENTION:-auto}"
  local model="${DEFAULT_MODEL:-${BASE_MODEL:-${GENERATOR_MODEL:-}}}"

  if [[ "${model:l}" == *gpt-oss* && "${attention:l}" != "eager" ]]; then
    print -u2 "note: $model requires eager attention in this Transformers build; using --attention eager."
    attention="eager"
  fi

  print -r -- "$attention"
}

resolve_pretrain_device_map() {
  local device_map="${CONTINUED_PRETRAIN_DEVICE_MAP:-${DEFAULT_DEVICE_MAP:-auto}}"
  local model="${DEFAULT_MODEL:-${BASE_MODEL:-${GENERATOR_MODEL:-}}}"

  if [[ "${model:l}" == *gpt-oss* ]]; then
    case "${device_map:l}" in
      single|auto)
        print -u2 "note: $model continued pretraining needs trainable-tail placement; using --device_map trainable."
        device_map="trainable"
        ;;
    esac
  fi

  print -r -- "$device_map"
}

require_python_helpers() {
  if ! "$PYTHON" - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
import utils.http_client
PY
  then
    print -u2 "error: selected Python cannot import the project helpers: $PYTHON"
    print -u2 "Run ./install.zsh or set PYTHON=/path/to/.venv/bin/python and retry."
    exit 1
  fi
}

run_raw_text() {
  require_python_helpers
  mkdir -p "$RAW_TEXT_OUTPUT_DIR"

  print "Raw text output: $RAW_TEXT_OUTPUT_DIR"
  print "Python: $PYTHON"
  print "Parallel downloader jobs: $RAW_TEXT_JOBS"
  print "Wikipedia crawl: depth=$WIKI_CRAWL_DEPTH max_pages=$WIKI_CRAWL_MAX_PAGES links_per_page=$WIKI_CRAWL_LINKS_PER_PAGE"

  if (( RUN_GUTENBERG )); then
    print
    print "Project Gutenberg"
    download_gutenberg
  fi

  if (( RUN_WIKIPEDIA )); then
    print
    print "Wikipedia plaintext"
    download_wikipedia
  fi

  if (( RUN_HTML )); then
    print
    print "HTML to plaintext"
    download_html
  fi
}

run_clean_text() {
  mkdir -p "$PREPARED_OUTPUT_DIR"
  run_cmd "$PYTHON" "$ROOT/src/data_prep/clean_text.py" \
    --input-dir "$RAW_TEXT_OUTPUT_DIR" \
    --output-dir "$PREPARED_OUTPUT_DIR"
}

run_pretrain_corpus() {
  mkdir -p "$CORPUS_OUTPUT_DIR"
  run_cmd "$PYTHON" "$ROOT/src/data_prep/generate_pretrain_corpus.py" \
    --text_dir "$PREPARED_OUTPUT_DIR" \
    --corpus_dir "$CORPUS_OUTPUT_DIR" \
    --num_proc "$DATASET_NUM_PROC"
}

run_create_corpus_token() {
  mkdir -p "${CORPUS_TOKEN_OUTPUT_DIR:h}"
  run_cmd "$PYTHON" "$ROOT/src/data_prep/create_corpus_token.py" \
    --text_dir "$PREPARED_OUTPUT_DIR" \
    --output_dir "$CORPUS_TOKEN_OUTPUT_DIR" \
    --model_name "${DEFAULT_MODEL:-${BASE_MODEL:-${GENERATOR_MODEL:-}}}" \
    --block_size "$CORPUS_TOKEN_BLOCK_SIZE" \
    --num_proc "$DATASET_NUM_PROC" \
    "${PASSTHROUGH_ARGS[@]}"
}

run_pairs() {
  mkdir -p "$CORPUS_OUTPUT_DIR"
  run_cmd "$PYTHON" "$ROOT/src/data_prep/make_training_pairs.py" \
    --text-dir "$PREPARED_OUTPUT_DIR" \
    --output-train "$TRAINING_PAIRS_TRAIN" \
    --output-val "$TRAINING_PAIRS_VAL"
}

run_corpus() {
  print "Corpus pipeline:"
  print "  raw text: $RAW_TEXT_OUTPUT_DIR"
  print "  prepared text: $PREPARED_OUTPUT_DIR"
  print "  corpus: $CORPUS_OUTPUT_DIR"
  print "  training pairs train: $TRAINING_PAIRS_TRAIN"
  print "  training pairs val: $TRAINING_PAIRS_VAL"

  if (( RUN_DOWNLOAD )); then
    run_raw_text
  fi
  if (( RUN_CLEAN )); then
    run_clean_text
  fi
  if (( RUN_PRETRAIN_CORPUS )); then
    run_pretrain_corpus
  fi
  if (( RUN_PAIRS )); then
    run_pairs
  fi
}

run_rag() {
  local input_dir output_dir embed rerank chunk_size overlap batch

  input_dir="$PREPARED_OUTPUT_DIR"
  output_dir="$RAG_OUTPUT_DIR"
  embed="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
  rerank="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
  chunk_size="${CHUNK_SIZE_CHARS:-1800}"
  overlap="${OVERLAP_CHARS:-250}"
  batch="${BATCH_SIZE:-32}"

  case "${embed:l}" in
    *gpt-oss*)
      print -u2 "error: EMBED_MODEL is set to '$embed', but gpt-oss is a generator model."
      print -u2 "Set EMBED_MODEL=BAAI/bge-base-en-v1.5 and rerun."
      exit 2
      ;;
  esac

  case "${rerank:l}" in
    *gpt-oss*)
      print -u2 "error: RERANKER_MODEL is set to '$rerank', but gpt-oss is a generator model."
      print -u2 "Set RERANKER_MODEL=BAAI/bge-reranker-v2-m3 and rerun."
      exit 2
      ;;
  esac

  run_cmd "$PYTHON" "$ROOT/src/rag/index_builder.py" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --embed-model "$embed" \
    --chunk-size-chars "$chunk_size" \
    --overlap-chars "$overlap" \
    --batch-size "$batch"
}

run_pretrain() {
  if [[ -z "${PYTORCH_ALLOC_CONF:-}" && -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  fi
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_ALLOC_CONF}"

  if [[ "$PYTHON" == */* && ! -x "$PYTHON" ]]; then
    print -u2 "error: Python executable not found: $PYTHON"
    print -u2 "Run ./install.zsh --backend cuda or ./install.zsh --backend rocm, then source .runtime."
    exit 1
  fi

  local eval_prompts
  eval_prompts="${EVAL_PROMPTS:-$ROOT/eval_prompts.txt}"
  if [[ ! -f "$eval_prompts" ]]; then
    print -u2 "error: eval prompts file not found: $eval_prompts"
    print -u2 "Set EVAL_PROMPTS=/path/to/prompts.txt or pass -- --eval_prompts /path/to/prompts.txt."
    exit 1
  fi

  local -a eval_prompt_args rocm_safe_args
  local pretrain_device_map=""
  eval_prompt_args=()
  rocm_safe_args=()

  if ! has_arg "--eval_prompts" "${PASSTHROUGH_ARGS[@]}"; then
    eval_prompt_args=(--eval_prompts "$eval_prompts")
  fi
  if ! has_arg "--corpus_dir" "${PASSTHROUGH_ARGS[@]}"; then
    eval_prompt_args+=(--corpus_dir "$CORPUS_OUTPUT_DIR")
  fi
  if ! has_arg "--attention" "${PASSTHROUGH_ARGS[@]}"; then
    rocm_safe_args+=(--attention "$(resolve_pretrain_attention)")
  fi
  if ! has_arg "--device_map" "${PASSTHROUGH_ARGS[@]}"; then
    pretrain_device_map="$(resolve_pretrain_device_map)"
    rocm_safe_args+=(--device_map "$pretrain_device_map")
  fi
  if ! has_arg "--max_memory" "${PASSTHROUGH_ARGS[@]}"; then
    if [[ -z "$pretrain_device_map" || "${pretrain_device_map:l}" == "auto" ]]; then
      rocm_safe_args+=(--max_memory "${CONTINUED_PRETRAIN_MAX_MEMORY:-4GiB}")
    fi
  fi
  if ! has_arg "--optim" "${PASSTHROUGH_ARGS[@]}"; then
    rocm_safe_args+=(--optim "${CONTINUED_PRETRAIN_OPTIM:-adamw_torch}")
  fi
  if ! has_arg "--mxfp4_dequantize" "${PASSTHROUGH_ARGS[@]}" && ! has_arg "--no-mxfp4_dequantize" "${PASSTHROUGH_ARGS[@]}"; then
    if [[ "${CONTINUED_PRETRAIN_MXFP4_DEQUANTIZE:-1}" != "0" ]]; then
      rocm_safe_args+=(--mxfp4_dequantize)
    fi
  fi

  print "Continued pretrain:"
  print "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
  print "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  print "  injected args: ${eval_prompt_args[*]} ${rocm_safe_args[*]}"

  run_cmd "$PYTHON" "$ROOT/src/training/continued_pretrain_partial.py" \
    "${eval_prompt_args[@]}" \
    "${rocm_safe_args[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
}

run_lora() {
  run_cmd "$PYTHON" "$ROOT/src/training/train_pipeline.py" "${PASSTHROUGH_ARGS[@]}"
}

run_install_gh() {
  run_cmd sudo apt update
  run_cmd sudo apt install -y curl gpg ca-certificates
  run_cmd zsh -c 'curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg >/dev/null'
  run_cmd sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
  run_cmd zsh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null'
  run_cmd sudo apt update
  run_cmd sudo apt install -y gh
  run_cmd zsh -c 'gh auth login --with-token < gh_token.txt'
}

run_amd_monitor() {
  run_cmd amd-smi version
  run_cmd amd-smi list
  run_cmd amd-smi metric
  run_cmd amd-smi monitor -putm
  run_cmd sudo apt install radeontop
  run_cmd sudo radeontop
}

expanded_commands=()
for command in "${COMMANDS[@]}"; do
  case "$command" in
    all)
      expanded_commands+=(corpus rag pretrain)
      ;;
    *)
      expanded_commands+=("$command")
      ;;
  esac
done

if [[ "${#PASSTHROUGH_ARGS[@]}" -gt 0 && "${#expanded_commands[@]}" -ne 1 ]]; then
  print -u2 "error: pass-through args after -- require exactly one command"
  exit 2
fi

for command in "${expanded_commands[@]}"; do
  case "$command" in
    corpus)
      run_corpus
      ;;
    raw-text)
      run_raw_text
      ;;
    clean-text)
      run_clean_text
      ;;
    pretrain-corpus)
      run_pretrain_corpus
      ;;
    create-corpus-token)
      run_create_corpus_token
      ;;
    pairs)
      run_pairs
      ;;
    rag)
      run_rag
      ;;
    pretrain)
      run_pretrain
      ;;
    lora)
      run_lora
      ;;
    install-gh)
      run_install_gh
      ;;
    amd-monitor)
      run_amd_monitor
      ;;
    *)
      print -u2 "error: unknown command: $command"
      exit 2
      ;;
  esac
done
