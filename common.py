"""Shared model registry and loading for the Jacobian lens scripts."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

import jlens

# Pre-fitted lenses from Neuronpedia (1000 wikitext prompts each).
LENS_REPO = "neuronpedia/jacobian-lens"


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    lens_file: str
    lens_revision: str
    title: str


MODELS = {
    # Ungated mirror of google/gemma-3-4b-it (same weights, no HF licence click-through).
    "gemma-3-4b": ModelSpec(
        "unsloth/gemma-3-4b-it",
        "gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt",
        "main",
        "Gemma 3 4B",
    ),
    "qwen3.5-4b": ModelSpec(
        "Qwen/Qwen3.5-4B",
        "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        "qwen-n1000",
        "Qwen3.5 4B",
    ),
    # ~56 GB of bf16 weights: needs an 80 GB GPU (see RUNPOD.md), not the MacBook.
    "qwen3.6-27b": ModelSpec(
        "Qwen/Qwen3.6-27B",
        "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
        "qwen-n1000",
        "Qwen3.6 27B",
    ),
}
DEFAULT_MODEL = "gemma-3-4b"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)


def slice_dir(model: str, prompt: str) -> Path:
    """Where ``lens.py --page`` writes the slice view for a free-text prompt."""
    return Path("slices") / model / re.sub(r"\W+", "-", prompt.lower())[:40].strip("-")


def load_tokenizer(model: str):
    return transformers.AutoTokenizer.from_pretrained(MODELS[model].hf_id)


def load_model(model: str):
    spec = MODELS[model]
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16
    ).to(DEVICE)
    tokenizer = load_tokenizer(model)
    return jlens.from_hf(hf_model, tokenizer), tokenizer, hf_model


def generate(hf_model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy continuation of ``prompt`` (raw text, no chat template)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(hf_model.device)
    out = hf_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load_lens(model: str, path: str | None = None) -> jlens.JacobianLens:
    if path:
        return jlens.JacobianLens.load(path)
    spec = MODELS[model]
    return jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=spec.lens_file, revision=spec.lens_revision
    )
