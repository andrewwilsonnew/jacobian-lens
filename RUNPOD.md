# Running on a RunPod GPU

The MacBook handles the 4B models. Use a RunPod GPU pod for:

- **Qwen3.6-27B.** It has ~56 GB of bf16 weights, so it needs an 80 GB+ GPU.
- **Fitting your own lens.** On a 17 GB Mac it takes hours (see the README); a datacentre
  GPU is much faster.

A pod bills by the hour from creation until you terminate it, so stop or delete it
when you're done.

## Pick a GPU

Secure-cloud prices and stock from RunPod's catalog on 2026-11-18. Prices change; check
the console or `list-gpu-types` before creating.

| GPU | VRAM | $/hr | Notes |
|---|---|---|---|
| RTX PRO 6000 (Blackwell Server) | 96 GB | 2.69 | **Recommended.** Best stock (MEDIUM, 11 data centres) and headroom for fitting |
| A100 SXM | 80 GB | 2.19 | Cheaper; LOW stock |
| A100 PCIe | 80 GB | 2.09 | Cheapest 80 GB; LOW stock, one data centre |
| H100 SXM | 80 GB | 4.39 | Fastest, but costs more than the job needs |

Our test session (pod boot, ~60 GB download including three stalled-download restarts,
three lens runs of ~1 minute each) took 36 minutes, about $1.61 on an RTX PRO 6000. A
session without the stalls should be about 20–25 minutes.

## 1. Create the pod

In the RunPod console (or via the RunPod MCP `create-pod`):

| Setting | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Server Edition × 1, Secure Cloud |
| Template / image | Runpod Pytorch 2.8.0 (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) |
| Minimum CUDA | **13.2**: the locked PyTorch wheel is built against CUDA 13.2 |
| Container disk | 30 GB |
| Volume disk | 120 GB at `/workspace` (weights + lens + venv) |
| Ports | `22/tcp` only, SSH on, using your registered public key |

The project installs its own PyTorch with `uv`, so the image's torch version doesn't
matter. The CUDA floor does.

Once the pod is running, copy its **direct TCP SSH** address from the pod's Connect tab.
It looks like `ssh root@<ip> -p <port>`. (The proxy SSH address can't do rsync.)

## 2. Copy the code and set up

From the Mac, in this repo:

```bash
runpod/sync.sh push <ip> <port>
ssh root@<ip> -p <port>
```

On the pod:

```bash
cd /workspace/jacobian-lens
bash runpod/bootstrap.sh qwen3.6-27b    # uv + venv + prefetch weights and lens (~10 min)
source ~/.bashrc
```

`bootstrap.sh` keeps the HuggingFace and uv caches on `/workspace`, so a stopped and
restarted pod doesn't download again. It also tries to install `flash-linear-attention`
(fast kernels for Qwen's linear-attention layers). If that fails, the reference kernels
still work, just slower.

If a download stops growing for a minute or more, check the newest files with
`ls -lt /workspace/hf-cache/hub/*/blobs | head` (the cache links finished files into
`blobs/`). Then kill the download **by PID** and rerun `bootstrap.sh`. Don't use
`pkill -f <script name>` over SSH: the pattern also matches the SSH session's own command
and kills it. Bootstrap disables HF's Xet backend (`HF_HUB_DISABLE_XET=1`), because Xet
stalled repeatedly in our test. If the lens file still stalls, fetch it directly and pass
`--lens`:

```bash
curl -fL -C - --retry 5 --speed-limit 100000 --speed-time 20 -o /workspace/lens-27b.pt \
  https://huggingface.co/neuronpedia/jacobian-lens/resolve/qwen-n1000/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt
uv run python lens.py --model qwen3.6-27b --lens /workspace/lens-27b.pt --page
```

## 3. Run

The same scripts work with `--model`:

```bash
uv run python lens.py --model qwen3.6-27b --page
uv run python lens.py --model qwen3.6-27b --page --generate 60 "don't show working : (23+17)*2 + 17="
uv run python digits.py --model qwen3.6-27b --stride 1
uv run python fit.py --model qwen3.6-27b --n-prompts 100 --dim-batch 8   # own lens
```

Qwen3.6-27B is a reasoning model. On instruction-style prompts like "don't show
working…" it starts a `<think>` block, so give `--generate` a few hundred tokens if you
want the final answer on the page. Plain-completion prompts (the boot question) answer
directly.

`--dim-batch` sets memory, not total compute. Raise it only while `nvidia-smi` shows
headroom.

## 4. Bring results back, then stop paying

From the Mac:

```bash
runpod/sync.sh pull <ip> <port>     # slices/ and lenses/ -> this repo
open slices/qwen3.6-27b/*/index.html
```

Then in the console, or via the RunPod MCP:

- **Terminate** the pod when you're finished. That deletes it and its volume disk, and
  billing stops.
- **Stop** the pod to keep `/workspace` (cached weights and venv) for next time. The GPU
  bill stops, but the volume disk keeps billing at the stopped-disk rate until you
  terminate. A restart may land without a free GPU on that host.
