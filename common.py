"""Shared loading for the Gemma 3 4B Jacobian lens scripts (Apple Silicon / MPS)."""

import re
from pathlib import Path

import torch
import transformers

import jlens

# Ungated mirror of google/gemma-3-4b-it (same weights, no HF licence click-through).
MODEL = "unsloth/gemma-3-4b-it"
# Pre-fitted lens from Neuronpedia (1000 wikitext prompts, fit on google/gemma-3-4b-it).
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "gemma-3-4b-it/jlens/Salesforce-wikitext/gemma-3-4b-it_jacobian_lens.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def slice_dir(prompt: str) -> Path:
    """Where ``lens.py --page`` writes the slice view for a free-text prompt."""
    return Path("slices") / re.sub(r"\W+", "-", prompt.lower())[:40].strip("-")


def load_model():
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    return jlens.from_hf(hf_model, tokenizer), tokenizer, hf_model


def generate(hf_model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy continuation of ``prompt`` (raw text, no chat template)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(hf_model.device)
    out = hf_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load_lens(path: str | None = None) -> jlens.JacobianLens:
    if path:
        return jlens.JacobianLens.load(path)
    return jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

