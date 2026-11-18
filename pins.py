"""Compare how strongly words show up in the J-lens over a generated answer.

    uv run python lens.py --page --generate 80 --pin pink --pin elephant --pin giraffe \\
        "dont think about a pink elephant, what is (17 + 13)*2 + 14"
    uv run python lens.py --page --generate 80 --pin pink --pin elephant --pin giraffe \\
        "what is (17 + 13)*2 + 14"
    uv run python pins.py --pin pink --pin elephant --pin giraffe \\
        "dont think about a pink elephant, what is (17 + 13)*2 + 14" "what is (17 + 13)*2 + 14"

Reads the rank data embedded in the slice pages ``lens.py --page`` wrote (no model
load). For each answer position it takes the word's best rank over layers, then
reports the best, median and how many positions came within the top 100 / 1000.
Pin the words when building the page so they always have rank data.
"""

import argparse
import base64
import gzip
import json

import numpy as np
import transformers

from common import MODEL, slice_dir


def load_page(prompt: str) -> tuple[dict, dict]:
    path = slice_dir(prompt) / "index.html"
    html = path.read_text()
    bootstrap, _ = json.JSONDecoder().raw_decode(html[html.index('{"mode"') :])
    if bootstrap["mode"] != "embed":
        raise ValueError(f"{path} is not a self-contained (embed mode) page")
    return bootstrap["meta"], bootstrap["files"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="+", help="prompts whose pages to compare")
    parser.add_argument("--pin", action="append", required=True, help="word (repeatable)")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    for prompt in args.prompts:
        meta, files = load_page(prompt)
        strs, layers, seq_len = meta["ctx_strs"], meta["layers"], meta["T"]
        answer_start = len(tokenizer(prompt).input_ids)
        print(f"\n{prompt!r}: {seq_len - answer_start} answer positions")

        for word in args.pin:
            tid = tokenizer.encode(" " + word, add_special_tokens=False)[0]
            blob = files.get(f"ranks/{tid}.bin")
            if blob is None:
                print(f"  {word:>10}: no rank data; rebuild the page with --pin {word}")
                continue
            ranks = np.frombuffer(gzip.decompress(base64.b64decode(blob)), "<i4")
            ranks = ranks.reshape(seq_len, len(layers))
            best = ranks[answer_start:].min(axis=1)  # best rank over layers per position
            pos = int(best.argmin()) + answer_start
            layer = layers[int(ranks[pos].argmin())]
            context = "".join(strs[max(0, pos - 4) : pos + 1])
            print(
                f"  {word:>10}: best {int(best.min()):>6} (L{layer} at ...{context!r})"
                f" | median {int(np.median(best)):>6}"
                f" | <100: {int((best < 100).sum())}  <1000: {int((best < 1000).sum())}"
            )


if __name__ == "__main__":
    main()
