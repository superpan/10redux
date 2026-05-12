# ten

Open-weight video search for large libraries.

- **Visual embeddings:** [V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) (Meta, self-supervised, strong temporal understanding)
- **Captions / summaries:** [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (256K context, hours-long video)
- **Caption embeddings:** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Vector index:** Qdrant (two collections, RRF-fused at query time)
- **API + UI:** FastAPI + React (Vite)
- **CLI:** `ten`

## How it works

1. **Index.** Each video is split into ~10 s overlapping clips. For every clip:
   - **V-JEPA 2** produces a normalized visual vector (mean-pooled tokens).
   - **Qwen3-VL** writes a 1–2 sentence caption.
   - **Qwen3-Embedding** embeds the caption.
   - Both vectors land in Qdrant under the same clip id, and a JPEG thumbnail is cached.
2. **Search.**
   - **Text** → caption-vector search.
   - **Image / video** → V-JEPA visual search.
   - Combine both → reciprocal-rank fusion.
3. **Summarize on demand.** The "summarize" button (and `ten summary <clip_id>`) re-decodes the clip and asks Qwen3-VL for a paragraph-length summary. Cached in memory.

The two-collection layout means you don't need a text-aligned video encoder — the LLM does the text alignment by writing captions, and you still get fast vector retrieval at query time.

## Requirements

- NVIDIA GPU with ≥24 GB recommended (Qwen3-VL-8B in bf16). Tested on DGX Spark (Grace Blackwell GB10).
- `ffmpeg` on PATH (`sudo apt install ffmpeg`)
- Docker (for Qdrant) — or point `TEN_QDRANT_URL` at any Qdrant instance.
- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Node 20+ for the UI

## Setup & use

Everything goes through the **Makefile**. Run `make` (or `make help`) to see the menu.

```bash
make ffmpeg                              # one-time: install ffmpeg
make bootstrap                           # uv sync + npm install
make qdrant-up                           # start Qdrant in Docker
make ui-build                            # build the React frontend

make index FOLDER=/path/to/videos        # ingest (resumable)
make smoke FOLDER=/path/to/videos        # only first 3 videos (quick check)
make search Q="a person catching a ball at sunset"
make status                              # config + Qdrant stats
make serve                               # http://127.0.0.1:8765
```

For high-throughput ingest (~3–5× faster), bring up vLLM and use the vLLM-backed indexer:

```bash
make vllm-up
make index-vllm FOLDER=/path/to/videos
```

For frontend hot-reload, run the API and the Vite dev server in two terminals:

```bash
make serve-reload    # terminal 1
make ui-dev          # terminal 2  -> http://127.0.0.1:5173
```

Direct CLI usage (`uv run ten ...`) still works — see `make help` and `uv run ten --help` for the full surface.

## Captioner backends

Captioning is the throughput bottleneck of ingest. Two backends:

| backend | when to use | how |
|---|---|---|
| `transformers` *(default)* | Small/medium libraries, on-demand summaries, zero extra infra. | Loads Qwen3-VL in-process. No env needed. |
| `vllm` | Indexing thousands of hours of video. ~3–5× faster decode. | Run a vLLM server, then `export TEN_VLM_BACKEND=vllm`. |

### Run vLLM via docker-compose

```bash
# Optional HF token if you've gated any models
export HUGGING_FACE_HUB_TOKEN=hf_...

docker compose --profile vllm up -d           # starts both qdrant and vllm
export TEN_VLM_BACKEND=vllm                   # tell ten to use it
uv run ten index /path/to/videos
```

vLLM relevant env vars:

| var | default | meaning |
|---|---|---|
| `TEN_VLM_BACKEND` | `transformers` | `transformers` or `vllm` |
| `TEN_VLLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `TEN_VLLM_MODEL` | same as `TEN_VLM_MODEL` | Model name as served by vLLM |
| `TEN_VLLM_API_KEY` | `EMPTY` | Bearer token (vLLM doesn't enforce by default) |
| `TEN_VLLM_CONCURRENCY` | `8` | Inflight clip captions; vLLM's continuous batching does the rest |

> **DGX Spark note.** `vllm/vllm-openai:latest` ships multi-arch images, but if Blackwell (sm_100) support is missing in your tag, swap the image for NVIDIA's NGC build (`nvcr.io/nvidia/vllm:25.04-py3` or newer) — same CLI args.

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

## Scaling notes

- **Throughput** is dominated by Qwen3-VL captioning. Switch to the **vLLM backend** (see above) for ~3–5× faster ingest, or drop to `Qwen/Qwen3-VL-2B-Instruct` for a smaller speed-up at modest quality cost.
- **Qdrant** scales to hundreds of millions of points on a single node with proper sharding; for huge libraries enable on-disk vectors and quantization.
- **Resumable**: clip ids are deterministic hashes of `(video_path, t_start, t_end)`. Re-running `ten index` only embeds new clips.
- **Cold start**: first call loads ~3 models (V-JEPA + Qwen3-VL + Qwen3-Embedding). Plan for ~30 s of GPU warmup.
