#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

try:
    import _bootstrap  # noqa: F401
except ImportError:
    from . import _bootstrap  # noqa: F401

from utils.runtime_env import env_int, env_str

DEFAULT_MODEL = env_str("INVOICE_MODEL", "impira/layoutlm-invoices")
DEFAULT_DEVICE = env_str("INVOICE_DEVICE", "auto")
DEFAULT_HOST = env_str("INVOICE_HOST", "127.0.0.1")
DEFAULT_PORT = env_int("INVOICE_PORT", 7862)
DEFAULT_TOP_K = env_int("INVOICE_TOP_K", 1)
DEFAULT_MAX_ANSWER_LEN = env_int("INVOICE_MAX_ANSWER_LEN", 64)
DEFAULT_OCR_LANG = env_str("INVOICE_OCR_LANG", "eng")
DEFAULT_TESSERACT_CONFIG = env_str("INVOICE_TESSERACT_CONFIG", "")
DEFAULT_OCR_TIMEOUT = env_int("INVOICE_OCR_TIMEOUT", 0)
USE_HF_TOKEN = env_str("INVOICE_USE_HF_TOKEN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def hf_token_kwargs() -> dict[str, str]:
    if not USE_HF_TOKEN:
        return {}
    token = (
        env_str("HF_TOKEN")
        or env_str("HF_HUB_TOKEN")
        or env_str("HUGGING_FACE_HUB_TOKEN")
    )
    return {"token": token} if token else {}


def resolve_pipeline_device(device: str) -> int:
    value = device.strip().lower()
    if value in {"", "auto"}:
        return 0 if torch.cuda.is_available() else -1
    if value in {"cpu", "-1"}:
        return -1
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    return int(value)


def ensure_ocr_available() -> None:
    if shutil.which("tesseract") is None:
        raise SystemExit(
            "Tesseract OCR binary not found. Install it with your OS package "
            "manager, for example: sudo apt install tesseract-ocr"
        )
    try:
        import pytesseract  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Python package pytesseract is missing. Install dependencies with "
            "./install.zsh or: .venv/bin/python -m pip install pytesseract"
        ) from exc


def open_invoice_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        raise SystemExit(
            "PDF input is not supported directly by this wrapper. Convert the "
            "invoice page to PNG/JPEG first, then pass it with --image."
        )
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError as exc:
        raise SystemExit(f"Invoice image not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise SystemExit(f"Could not read invoice image: {path}") from exc


def open_invoice_image_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc


def normalize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in answer.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            value = round(value, 6)
        normalized[key] = value
    return normalized


class InvoiceQuestionAnswering:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        top_k: int = DEFAULT_TOP_K,
        max_answer_len: int = DEFAULT_MAX_ANSWER_LEN,
        ocr_lang: str = DEFAULT_OCR_LANG,
        tesseract_config: str = DEFAULT_TESSERACT_CONFIG,
        ocr_timeout: int = DEFAULT_OCR_TIMEOUT,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.pipeline_device = resolve_pipeline_device(device)
        self.top_k = top_k
        self.max_answer_len = max_answer_len
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config
        self.ocr_timeout = ocr_timeout if ocr_timeout > 0 else None
        self._pipe = None

    def load(self):
        if self._pipe is None:
            ensure_ocr_available()
            print(f"Loading invoice QA model: {self.model_name}")
            print(f"Pipeline device: {self.pipeline_device}")
            self._pipe = pipeline(
                "document-question-answering",
                model=self.model_name,
                device=self.pipeline_device,
                **hf_token_kwargs(),
            )
        return self._pipe

    def answer(
        self,
        image: Image.Image,
        question: str,
        top_k: int | None = None,
        max_answer_len: int | None = None,
    ) -> list[dict[str, Any]]:
        pipe = self.load()
        result = pipe(
            image=image,
            question=question,
            top_k=top_k or self.top_k,
            max_answer_len=max_answer_len or self.max_answer_len,
            lang=self.ocr_lang,
            tesseract_config=self.tesseract_config,
            timeout=self.ocr_timeout,
        )
        answers = result if isinstance(result, list) else [result]
        return [normalize_answer(answer) for answer in answers]


def read_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.question:
        questions.extend(args.question)
    if args.questions_file:
        path = Path(args.questions_file).expanduser()
        questions.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return questions


def run_once(args: argparse.Namespace) -> int:
    questions = read_questions(args)
    if not questions:
        raise SystemExit("Provide at least one --question or --questions-file.")

    image = open_invoice_image(Path(args.image).expanduser())
    runtime = InvoiceQuestionAnswering(
        model_name=args.model,
        device=args.device,
        top_k=args.top_k,
        max_answer_len=args.max_answer_len,
        ocr_lang=args.ocr_lang,
        tesseract_config=args.tesseract_config,
        ocr_timeout=args.ocr_timeout,
    )

    payload = []
    for question in questions:
        payload.append(
            {
                "question": question,
                "answers": runtime.answer(image, question),
            }
        )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload:
            print(f"\nQuestion: {item['question']}")
            for index, answer in enumerate(item["answers"], 1):
                text = answer.get("answer", "")
                score = answer.get("score")
                suffix = f" (score={score})" if score is not None else ""
                print(f"  {index}. {text}{suffix}")
    return 0


def create_app(runtime: InvoiceQuestionAnswering):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    app = FastAPI(title="LayoutLM Invoice QA", version="1.0")

    @app.on_event("startup")
    def _load_model() -> None:
        runtime.load()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": runtime.model_name,
            "device": runtime.device,
            "pipeline_device": runtime.pipeline_device,
        }

    @app.post("/answer")
    async def answer(
        question: str = Form(...),
        file: UploadFile = File(...),
        top_k: int | None = Form(None),
        max_answer_len: int | None = Form(None),
    ) -> dict[str, Any]:
        data = await file.read()
        try:
            image = open_invoice_image_bytes(data)
            answers = runtime.answer(
                image,
                question,
                top_k=top_k,
                max_answer_len=max_answer_len,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "question": question,
            "filename": file.filename,
            "answers": answers,
        }

    return app


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    runtime = InvoiceQuestionAnswering(
        model_name=args.model,
        device=args.device,
        top_k=args.top_k,
        max_answer_len=args.max_answer_len,
        ocr_lang=args.ocr_lang,
        tesseract_config=args.tesseract_config,
        ocr_timeout=args.ocr_timeout,
    )
    with tempfile.TemporaryDirectory(prefix="layoutlm-invoices-"):
        app = create_app(runtime)
        print(f"Starting invoice QA server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run impira/layoutlm-invoices for document question answering."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--image", "-i", help="Invoice image path for one-shot QA.")
    parser.add_argument("--question", "-q", action="append")
    parser.add_argument("--questions-file")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-answer-len", type=int, default=DEFAULT_MAX_ANSWER_LEN)
    parser.add_argument("--ocr-lang", default=DEFAULT_OCR_LANG)
    parser.add_argument("--tesseract-config", default=DEFAULT_TESSERACT_CONFIG)
    parser.add_argument("--ocr-timeout", type=int, default=DEFAULT_OCR_TIMEOUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.serve is True or (args.serve is None and not args.image):
        return serve(args)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
