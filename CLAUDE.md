# Repo guide for Claude

`ten` — open-weight video search over large libraries. See `README.md` for user-facing docs; this file is the operating manual for agents working on the codebase.

## Environment

- **Hardware**: DGX Spark — Grace Blackwell GB10, **aarch64**, CUDA 13.
- **Python**: 3.12, managed by **uv**. Always invoke as `uv run …`. Do not call `python` / `pip` directly and do not edit `.venv` by hand.
- **No `decord`** — no aarch64 wheels. Use **PyAV** (`av`) for all video decoding/probing.
- **PyTorch**: pulled from the `cu128` index (configured in `pyproject.toml` under `[tool.uv.sources]`). Don't switch indexes without checking aarch64 wheel availability.
- `ffmpeg` must be on PATH for thumbnails / fallback decode.

## Stack at a glance

| concern | choice | file |
|---|---|---|
| Visual embeddings | V-JEPA 2 (`facebook/vjepa2-vitl-fpc16-256-ssv2`) | `src/ten/embed_video.py` |
| Captions / summaries | Qwen3-VL-8B-Instruct | `src/ten/caption.py` |
| Caption embeddings | Qwen3-Embedding-0.6B (sentence-transformers) | `src/ten/embed_text.py` |
| Vector store | Qdrant (two collections, RRF-fused) | `src/ten/store.py`, `src/ten/search.py` |
| Ingest orchestrator | PyAV chunking → batched embed → upsert | `src/ten/ingest.py` |
| HTTP API + Range stream | FastAPI | `src/ten/api.py` |
| CLI | typer, entry point `ten` | `src/ten/cli.py` |
| Frontend | React (Vite) | `ui/` |

## Captioner backends

`src/ten/caption.py` exposes `CaptionerProtocol` with two implementations selected by `TEN_VLM_BACKEND`:

- `transformers` (default): in-process Qwen3-VL via HF `AutoModelForImageTextToText`.
- `vllm`: HTTP client to a vLLM OpenAI-compatible server — frames sent as multiple `image_url` blocks; concurrency knob feeds vLLM's continuous batching. Run via `docker compose --profile vllm up -d`.

When adding a new backend, implement `CaptionerProtocol` and register it in `make_captioner()`. Don't import the backend class anywhere else — call the factory.

## Conventions

- **Clip ids** are deterministic `blake2b(video_path | t_start | t_end)` hashes — see `Clip.clip_id` in `src/ten/video.py`. They map to Qdrant point ids via `clip_id_to_uuid`. This is what makes ingest resumable; don't change the hash inputs.
- **Two Qdrant collections** (`ten_visual`, `ten_text`) share an identical payload so a hit in either side renders the same UI card. Keep payload writes symmetric in `Store.upsert`.
- **Lazy model loading**: every model wrapper loads on first use, behind a lock. Don't eagerly load in `__init__` — it makes the CLI/API slow to start and breaks `ten status`.
- **Config** is env-driven, single source of truth in `src/ten/config.py`. Add new knobs there, not as scattered constants.
- **No comments on what code does** — keep the codebase explanation in this file and the docstrings already at module tops.

## Common commands

The **Makefile** is the canonical task runner — it's the single place workflows are codified. Run `make help` for the full menu.

```bash
make bootstrap                                # uv sync + npm install
make qdrant-up                                # start Qdrant
make vllm-up                                  # optional: start vLLM (profile=vllm)
make index FOLDER=/path/to/videos             # ingest (resumable)
make index-vllm FOLDER=/path/to/videos        # same, but via vLLM backend
make search Q="a person catching a ball"      # CLI search
make serve                                    # FastAPI on :8765 (auto-mounts ui/dist)
make serve-reload                             # serve with --reload
make ui-build                                 # build React UI into ui/dist
make ui-dev                                   # Vite dev server on :5173, proxies /search etc
make status                                   # print resolved config + qdrant stats
make lint   |   make format                   # ruff
```

When adding a new blessed workflow (new service, new ingest mode, new maintenance task), add a Makefile target with a `## description` doc-comment so it shows up in `make help`. Keep the section comments (`##@ Section`) in sync.

The raw `uv run ten ...` and `docker compose ...` commands still work — Makefile targets are thin wrappers, not gatekeepers.

## Gotchas

- **First ingest** downloads ~20 GB of weights (V-JEPA + Qwen3-VL + Qwen3-Embedding) into `~/.cache/huggingface`. Surface this clearly when asked about runtime.
- **Qdrant URL** defaults to `http://localhost:6333`. If `ten status` shows "Connection refused", the user hasn't started Docker yet.
- **`AutoVideoProcessor` import** in `embed_video.py` is from `transformers` ≥ 4.49. If transformers is downgraded for any reason, V-JEPA 2 won't load.
- **vLLM image** on DGX Spark: `vllm/vllm-openai:latest` may lag sm_100 (Blackwell) support. Fallback is NVIDIA's NGC build (`nvcr.io/nvidia/vllm:25.04-py3` or newer).
- The Vite proxy whitelist (`ui/vite.config.js`) must mirror the FastAPI route prefixes. When adding a new endpoint family, add it to the proxy too.
