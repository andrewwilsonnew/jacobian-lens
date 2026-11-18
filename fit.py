"""Fit your own Jacobian lens on Gemma 3 4B locally (resumable; slow, see README).

    uv run python fit.py --n-prompts 100
    uv run python lens.py --lens lenses/gemma-3-4b-it_n100.pt
"""

import argparse
from pathlib import Path

import jlens
from jlens.examples import load_wikitext_prompts

from common import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    jlens.configure_logging()
    model, _, _ = load_model()
    prompts = load_wikitext_prompts(n_prompts=args.n_prompts)

    out = Path("lenses")
    out.mkdir(exist_ok=True)
    stem = f"gemma-3-4b-it_n{args.n_prompts}"
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
