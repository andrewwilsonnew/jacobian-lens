# Jacobian lens on Gemma 3 4B (MacBook / MPS)

Anthropic's [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
(`jlens`, reference code at github.com/anthropics/jacobian-lens) applied to Gemma 3 4B.

The lens needs gradients through the residual stream, so it runs on the HuggingFace
weights in PyTorch on MPS (`unsloth/gemma-3-4b-it`, an ungated mirror of
`google/gemma-3-4b-it`). Ollama is not used: llama.cpp is inference-only and exposes no
gradients or activations. Stop it while running the lens (`pkill ollama`) since a loaded
`gemma3:4b` holds ~4 GB next to the 8.6 GB PyTorch copy.

The pre-fitted lens comes from `neuronpedia/jacobian-lens` (1000 wikitext prompts).

## Result: the boot-shaped country

Prompt: *"Fact: The currency used in the country shaped like a boot is"*. Gemma 3 4B
answers " the Euro (€)". The answer requires a hidden step (boot → Italy → euro), and the
J-lens shows that step in the middle layers, at the final " is" token, before the model
writes anything:

```
L17 logit-lens: [' iconic', ' called', ' ', ' legendary', ' now']
L17 J-lens:     [' Portugal', ' Portuguese', ' France', ' Belgium', ' Italy']
L25 logit-lens: [' called', ' Italian', ' the', ' euros', ' currency']
L25 J-lens:     [' Italian', ' euros', ' Euros', ' euro', ' Italy']
L32 J-lens:     [' the', ' Euro', ' called', ' Euros', ' Italian']
model output:   [' the', ' Euro', ' called', ' Euros', ' Italian']
```

The plain logit lens is noise at L17. The J-lens shows candidate European countries by
L17 and Italy/euro by L25.

**[Open the interactive slice view](https://andrewwilsonnew.github.io/gemma-jacobian-lens/results/boot-currency/?ctx=13&layer=0&pinned=506-10734-23613-16152-100608-14135-12896-35594-74906&ht=74906&ws=1)**
(prompt plus 20 answer tokens, " is" selected, with " the", " eventually", " Portugal",
" Euro", " Euros", " euro", " Italian", " EUR" and " EURO" pinned). The file is
[`results/boot-currency/index.html`](results/boot-currency/index.html).

![Slice view of the boot-shaped country prompt](results/boot-currency/screenshot.png)

Reading the page: the left grid is position × layer (top-1 lens token per cell). **By
Layer** fixes the selected position and lists every layer's top-10. **By Pos** fixes the
selected layer and lists every position's top-10. Each has a rank chart for the pinned
tokens. Click a token to pin it; use the arrow keys to move through positions and layers.

## Other experiments (scripts in this repo)

- **Multi-step arithmetic** (`digits.py`): on "(23 + 17)*2 + 17 =" the model shows its
  working (40, 80, 97) even when told "don't show working". No number appears at the
  final "=". Each intermediate result forms in a narrow band at L21–L25, first as a
  number word (" eighty", " ninety") and then as the digit. " Forty" shows at L21 one
  layer before " eighty" at L22 (the doubling step).
- **"Don't think about a pink elephant"** (`lens.py --pin`, `pins.py`): during the
  arithmetic answer, "pink" and "elephant" rank no higher than they do on the same sum
  without the elephant sentence, apart from a faint lift for animal words (control:
  " giraffe"). An apparent "pink" trace turned out to be present in the control prompt
  too.

## Usage

```bash
uv sync
uv run python lens.py          # boot-shaped-country prompt
uv run python lens.py "The capital of the state containing Dallas is" --page
uv run python lens.py --example ascii-face --page   # paper examples: see jlens.examples.EXAMPLES
open slices/*/index.html       # interactive layer x position view
uv run python digits.py "(23 + 17)*2 + 17 ="        # per-layer rank of each result digit
uv run python lens.py --page --generate 80 --pin pink --pin giraffe "<prompt>"  # track words
uv run python pins.py --pin pink --pin giraffe "<prompt A>" "<prompt B>"        # compare pages

uv run python fit.py --n-prompts 100                # fit your own lens (very slow, see below)
uv run python lens.py --lens lenses/gemma-3-4b-it_n100.pt
```

## Usage notes

**Why not Gemma 3 1B.** We tried the 1B model and dropped it. On the boot-shaped-country
prompt, the J-lens shows 1B settling on Spain/Portugal and it never reaches Italy, whereas 4B
reads out Portugal → Italy → euro across layers L17–L25 and answers "the Euro". That pattern
held with both the Neuronpedia lens and a locally fitted one, so it reflects the model and
not the lens. 1B is too weak for the multi-hop examples the lens is interesting on.

**Fitting is slow on a 17 GB MacBook.** Measured on 1B (M5, bf16, MPS): about 90 s per prompt
at `--dim-batch 8`, which works out to ~2.5 h for 100 prompts. 4B has more layers and a
wider residual stream, so expect several times longer per prompt (not measured).
Use the pre-fitted lens unless you need one fitted on a different corpus. `fit.py`
checkpoints after every prompt and resumes, so it can run overnight in chunks.

**Keep `--dim-batch` small.** `jlens` replicates the prompt `dim_batch` times and retains
the whole backward graph, so memory scales linearly with it, but total compute does not.
On 1B, `dim_batch=32` reached a 13.6 GB footprint and spent its time in swap (no prompt
finished in 18 min). `dim_batch=8` stayed at 5.7 GB. `fit.py` defaults to 4 for 4B, which
also has to hold ~8.6 GB of weights. Watch `sysctl vm.swapusage` while it runs.

A 3-prompt local fit on 1B already matched the n=1000 lens's top tokens at the
mid/late layers. Lens quality plateaus quickly, as the paper notes (§9.3).

## Licence

Apache-2.0 (see [LICENSE](LICENSE)), the same as Anthropic's `jlens`, which this builds
on. The slice page in `results/` is rendered by `jlens` (Apache-2.0) and inlines d3 (ISC).
No model weights or lens files are included; Gemma is subject to Google's Gemma terms.
