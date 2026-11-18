"""Apply a Jacobian lens to Gemma 3 and compare it with the plain logit lens.

    uv run python lens.py "Fact: The currency used in the country shaped like a boot is"
    uv run python lens.py --page "The capital of the state containing Dallas is"
    uv run python lens.py --example ascii-face --page
    uv run python lens.py --page --pin pink --pin elephant "dont think about a pink elephant, what is 2 + 2"

--page writes an interactive layer x position slice view to slices/<slug>/index.html,
covering the prompt plus --generate N greedily generated answer tokens.
"""

import argparse
from pathlib import Path

import jlens
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.vis import build_page, compute_slice

from common import generate, load_lens, load_model, slice_dir

DEFAULT_PROMPT = "Fact: The currency used in the country shaped like a boot is"


def report_pins(slice_data, pins: dict[str, int], answer_start: int) -> None:
    """Print each pinned word's best (lowest) rank over prompt vs answer positions."""
    tracked = slice_data.tracked_token_ids
    print("\npinned word ranks (0 = top; best over positions, per layer group):")
    for word, tid in pins.items():
        ranks = slice_data.rank_tensor[:, :, tracked.index(tid)]  # [seq, n_layers]
        for name, span in (("prompt", ranks[:answer_start]), ("answer", ranks[answer_start:])):
            if not len(span):
                continue
            row = "  ".join(
                f"L{layer}:{int(span[:, li].min())}"
                for li, layer in enumerate(slice_data.layers)
                if li % 4 == 0
            )
            print(f"  {word:>10} {name}: {row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--lens", help="local lens .pt (default: Neuronpedia pre-fit)")
    parser.add_argument("--position", type=int, default=-1, help="token position to read")
    parser.add_argument("--example", choices=[e.slug for e in EXAMPLES])
    parser.add_argument("--page", action="store_true", help="write the slice view HTML")
    parser.add_argument("--generate", type=int, default=20, help="answer tokens to add to the page")
    parser.add_argument("--pin", action="append", default=[], help="word to track on the page (repeatable)")
    args = parser.parse_args()

    jlens.configure_logging()
    model, tokenizer, hf_model = load_model()
    lens = load_lens(args.lens)

    example = next((e for e in EXAMPLES if e.slug == args.example), None)
    if example:
        prompt = resolve_prompt(example, tokenizer)
    else:
        prompt = args.prompt or DEFAULT_PROMPT

    n = model.n_layers
    layers = [n // 4, n // 2, 3 * n // 4, n - 2]
    j_logits, model_logits, _ = lens.apply(
        model, prompt, layers=layers, positions=[args.position]
    )
    l_logits, _, _ = lens.apply(
        model, prompt, layers=layers, positions=[args.position], use_jacobian=False
    )

    def top5(logits):
        return [tokenizer.decode([t]) for t in logits.topk(5).indices]

    print(f"\n{model}\nprompt: {prompt!r}\n")
    for layer in layers:
        print(f"L{layer:>3} logit-lens: {top5(l_logits[layer][0])}")
        print(f"L{layer:>3} J-lens:     {top5(j_logits[layer][0])}")
    print(f"model output:    {top5(model_logits[0])}")

    if args.page:
        out_dir = Path("slices") / example.slug if example else slice_dir(prompt)
        out_dir.mkdir(parents=True, exist_ok=True)
        answer = generate(hf_model, tokenizer, prompt, args.generate) if args.generate else ""
        if answer:
            print(f"\nanswer: {answer!r}")
        text = prompt + answer
        # " word" with a leading space is the usual mid-sentence token; take its first piece.
        pins = {w: tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in args.pin}
        slice_data = compute_slice(
            model, lens, text, mask_display=True, pinned_token_ids=set(pins.values())
        )
        if pins:
            report_pins(slice_data, pins, len(model.encode(prompt)[0]))
        page, _, _ = build_page(
            slice_data,
            text,
            title=example.section if example else "Gemma 3 4B J-lens",
            description=example.description if example else text,
        )
        (out_dir / "index.html").write_text(page)
        print(f"\nslice view: {(out_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
