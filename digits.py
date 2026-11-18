"""Track when each result digit of a worked calculation appears across layers.

    uv run python digits.py
    uv run python digits.py "(23 + 17)*2 + 17 =" --generate 80 --stride 1

Generates the model's answer, then for every number written after an "=" reads
the J-lens at the position that produces each digit (Gemma tokenises numbers one
digit at a time) and prints the digit's full-vocab rank (0 = top) per layer.
"""

import argparse

import jlens

from common import generate, load_lens, load_model

DEFAULT_PROMPT = "(23 + 17)*2 + 17 ="


def result_digit_positions(strs: list[str]) -> list[int]:
    """Positions whose next token is a digit of a number following ``=``."""
    positions = []
    for i in range(1, len(strs) - 1):
        if strs[i - 1].strip() != "=" or strs[i] != " ":
            continue
        j = i
        while j + 1 < len(strs) and strs[j + 1].isdigit():
            positions.append(j)
            j += 1
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--lens", help="local lens .pt (default: Neuronpedia pre-fit)")
    parser.add_argument("--generate", type=int, default=80, help="answer tokens to generate")
    parser.add_argument("--stride", type=int, default=3, help="read every Nth layer")
    args = parser.parse_args()

    jlens.configure_logging()
    model, tokenizer, hf_model = load_model()
    lens = load_lens(args.lens)

    answer = generate(hf_model, tokenizer, args.prompt, args.generate)
    text = args.prompt + answer
    print(f"\nanswer: {answer!r}")

    ids = model.encode(text)[0].tolist()
    strs = [tokenizer.decode([t]) for t in ids]
    targets = result_digit_positions(strs)
    if not targets:
        print("no '= <number>' results in the answer; try a larger --generate")
        return

    layers = list(range(4, model.n_layers - 1, args.stride))
    j_logits, _, _ = lens.apply(model, text, layers=layers, positions=targets)

    for k, pos in enumerate(targets):
        next_id = ids[pos + 1]
        context = "".join(strs[max(0, pos - 4) : pos + 1]).strip()
        print(f"\n'{context}' -> next token {strs[pos + 1]!r}   (pos {pos})")
        for layer in layers:
            logits = j_logits[layer][k]
            rank = int((logits > logits[next_id]).sum())
            top = [tokenizer.decode([t]) for t in logits.topk(5).indices]
            print(f"  L{layer:>2} rank {rank:>6}   top5 {top}")


if __name__ == "__main__":
    main()
