"""Fit your own Jacobian lens (resumable; slow on a MacBook, see README).

    uv run python fit.py --n-prompts 100
    uv run python lens.py --lens lenses/gemma-3-4b_n100.pt
    uv run python fit.py --model qwen3.6-27b --dim-batch 8   # on a GPU pod, see RUNPOD.md
"""

import argparse
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

from common import add_model_arg, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    add_model_arg(parser)
    parser.add_argument("--n-prompts", type=int, default=100)
    # Memory scales with dim_batch, total compute does not: keep it small on a 17 GB Mac.
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    jlens.configure_logging()
    model, _, _ = load_model(args.model)
    prompts = load_wikitext_prompts(n_prompts=args.n_prompts)

    out = Path("lenses")
    out.mkdir(exist_ok=True)
    stem = f"{args.model}_n{args.n_prompts}"
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(out / f"{stem}.ckpt.pt"),
        resume=True,
    )
    lens.save(str(out / f"{stem}.pt"))
    print(f"saved {out / f'{stem}.pt'}")


if __name__ == "__main__":
    main()
