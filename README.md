# ten

Open-weight video search for large libraries.

- **Visual embeddings** — [V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) (Meta; self-supervised, strong temporal understanding)
- **Captions / summaries** — [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (256K context, hours-long video)
- **Caption embeddings** — [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Vector index** — Qdrant, two collections, RRF-fused at query time
- **API + UI** — FastAPI + Next.js 15
- **CLI** — `ten`

> **Status: personal experiment.** Built quickly to validate that open-weight models can replace TwelveLabs-style hosted video search end-to-end. APIs may change without notice. No support promises.

## Quickstart

Roughly 10 minutes of human time + 30 minutes of model-download + ingest time, on a single CUDA GPU with ≥24 GB VRAM. Hand-built smoke set is 4 public-domain videos (~1.5 GB).

```bash
make ffmpeg                              # one-time, requires sudo
make bootstrap                           # uv sync + pnpm install + install-pc
make qdrant-up                           # Qdrant via Docker

make fetch-smoke                         # 4 PD videos -> ./videos/smoke (~1.5 GB)
make ui-build                            # Next.js static export -> ui/out
make index FOLDER=./videos/smoke         # first run downloads ~20 GB of weights
                                         # then ingests 337 clips in ~32 min

make search Q="a dragon breathing fire"  # CLI search
make serve                               # http://127.0.0.1:8765 (api + UI)
```

### Example results

Smoke run over Big Buck Bunny, Sintel, Sintel trailer, and Charlie Chaplin's *The Vagabond* (1916):

| query | top match |
|---|---|
| "a white rabbit in a forest" | `BigBuckBunny.mov` 7:48–7:58 — *"A white rabbit with long ears and a pink nose peeks out from under a large tree…"* |
| "a dragon breathing fire" | `Sintel.mkv` 5:24–5:34 — *"A young girl looks up in shock as a large, dark dragon with outstretched wings soars overhead…"* |
| "a man playing a violin" | `ChaplinVagabond.mp4` 1:39–1:49 — *"A man in a dark suit and bowler hat stands in a doorway, holding a violin and bow…"* |
| "sword fight in a desert canyon" | `Sintel.mkv` 6:27–6:37 — *"A warrior in armor battles a skeletal horse-like creature in a rocky canyon, swinging a weapon…"* |

All 7 queries from the smoke test returned a top hit from the correct source video.

## How it works

1. **Index.** Each video is split into overlapping ~10 s clips. For every clip:
   - **V-JEPA 2** produces a normalized visual vector (mean-pooled tokens).
   - **Qwen3-VL** writes a 1–2 sentence caption.
   - **Qwen3-Embedding** embeds the caption.
   - Both vectors land in Qdrant under the same clip id, and a JPEG thumbnail is cached.
2. **Search.**
   - **Text** → caption-vector search.
   - **Image / video** → V-JEPA visual search.
   - Combine both → reciprocal-rank fusion.
3. **Summarize on demand.** The "summarize" button (and `ten summary <clip_id>`) re-decodes the clip and asks Qwen3-VL for a paragraph-length summary. Cached in memory.

The two-collection layout means a text-aligned video encoder is unnecessary — the LLM does the text alignment by writing captions, and you still get fast vector retrieval at query time.

## Requirements

- NVIDIA GPU, ≥24 GB VRAM recommended (Qwen3-VL-8B in bf16).
- Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, `ffmpeg`.
- Node 20+ with corepack enabled (`corepack enable`) for the UI — `pnpm` is pinned via `packageManager` in `ui/package.json`.

See [Hardware notes](#hardware-notes) for Blackwell / DGX Spark specifics.

## Setup your own library

`make fetch-smoke` is for the quickstart only. Point at your own videos with:

```bash
make index FOLDER=/path/to/your/videos
```

Ingest is recursive and **resumable**: clip ids are deterministic hashes of `(video_path, t_start, t_end)`, so re-running only embeds new clips. Failures on individual videos don't abort the run.

## Local dev

For a single TUI with both api + UI (one Ctrl-C tears them down, color-coded interleaved logs):

```bash
make qdrant-up    # once per boot
make dev-ui       # api + Next.js dev server  ->  http://127.0.0.1:3000
```

`make dev` (api only) is also there if you don't need UI hot-reload. Direct CLI usage (`uv run ten ...`, `make serve`, `make ui-dev`) still works — see `make help` and `uv run ten --help` for the full surface.

## Configuration

All settings are env vars (see `src/ten/config.py`):

| var | default | meaning |
|---|---|---|
| `TEN_DATA_DIR` | `.ten` | Where thumbnails live |
| `TEN_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `TEN_VJEPA_MODEL` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | Visual encoder |
| `TEN_VLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Captioner / summarizer |
| `TEN_TEXT_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Caption embedder |
| `TEN_CLIP_SECONDS` | `10` | Clip window length |
| `TEN_CLIP_OVERLAP` | `1` | Overlap between adjacent clips |
| `TEN_FRAMES_PER_CLIP` | `16` | Frames sampled per clip |
| `TEN_FRAME_RESIZE` | `256` | Short-side resize before encoding |
| `TEN_DEVICE` | `cuda` | Torch device |
| `TEN_DTYPE` | `bfloat16` | Model dtype |
| `TEN_BATCH_SIZE` | `4` | Clips per ingest batch |

## Hardware notes

- **DGX Spark (Grace Blackwell GB10, aarch64, sm_121):** the project is pinned to the **PyTorch cu129 index** in `pyproject.toml`. cu128 omits `compute_120` PTX, which causes NVRTC to fail with `invalid value for --gpu-architecture` on first kernel JIT.
- **decord is not installed** — no aarch64 wheels. Decoding goes through PyAV, which ships its own bundled libav so it doesn't conflict with a system `ffmpeg`.
- **Qdrant is pinned to 1.12.4** because 1.18+ refuses to load segments written by 1.12. To upgrade Qdrant, do a clean re-index.

## Scaling

For thousands of hours of video, captioning is the throughput bottleneck. Two backends:

| backend | when to use | how |
|---|---|---|
| `transformers` *(default)* | Small/medium libraries, on-demand summaries, zero extra infra. | Loads Qwen3-VL in-process. No env needed. |
| `vllm` | Heavy ingest. ~3–5× faster decode via continuous batching. | Run a vLLM server, then `export TEN_VLM_BACKEND=vllm`. |

### Run vLLM via docker-compose

```bash
# Optional HF token if you've gated any models
export HUGGING_FACE_HUB_TOKEN=hf_...

make vllm-up                                  # docker compose --profile vllm
export TEN_VLM_BACKEND=vllm
make index FOLDER=/path/to/videos
```

vLLM env vars:

| var | default | meaning |
|---|---|---|
| `TEN_VLM_BACKEND` | `transformers` | `transformers` or `vllm` |
| `TEN_VLLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `TEN_VLLM_MODEL` | same as `TEN_VLM_MODEL` | Model name as served by vLLM |
| `TEN_VLLM_API_KEY` | `EMPTY` | Bearer token (vLLM doesn't enforce by default) |
| `TEN_VLLM_CONCURRENCY` | `8` | Inflight clip captions; vLLM's continuous batching does the rest |
| `TEN_VLLM_TIMEOUT` | `300` | Per-request client timeout (seconds). Bump higher if you see `ReadTimeout` under heavy concurrency. |

> The vLLM image is pinned to `nvcr.io/nvidia/vllm:25.12.post1-py3` in `docker-compose.yml` — NVIDIA's NGC build, blessed for Blackwell (GB10 / sm_121). The compose `command` starts with `vllm serve …` because the NGC entrypoint is just an env-setup wrapper.

### Other levers

- **Smaller captioner:** `TEN_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct` for a modest speed-up with quality cost.
- **Sharded ingest:** Python decode is single-threaded per process. For >10k videos, run `make index FOLDER=<subset>` per subset under GNU parallel — deterministic clip ids make it safe.
- **Quantization:** at >1M points, enable Qdrant int8 scalar quantization on the visual collection. ~4× smaller, negligible recall drop.
- **Cold start:** first ingest downloads ~20 GB of weights. Subsequent runs warm up in ~30 s.

## License

MIT — see [LICENSE](LICENSE).
