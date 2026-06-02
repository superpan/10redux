# Repo guide for Claude

`ten` — open-weight video search over large libraries. See `README.md` for user-facing docs; this file is the operating manual for agents working on the codebase.

## Environment

- **Hardware**: DGX Spark — Grace Blackwell GB10, **aarch64**, CUDA 13.
- **Python**: 3.12, managed by **uv**. Always invoke as `uv run …`. Do not call `python` / `pip` directly and do not edit `.venv` by hand.
- **No `decord`** — no aarch64 wheels. Use **PyAV** (`av`) for all video decoding/probing.
- **PyTorch**: pulled from the `cu129` index (configured in `pyproject.toml` under `[tool.uv.sources]`). **Do not downgrade to cu128** — it lacks `compute_120` PTX, so on GB10 (device cap 12.1 / sm_121) PyTorch falls back to NVRTC JIT which then fails with `nvrtc: error: invalid value for --gpu-architecture (-arch)`. cu129 ships `compute_120` PTX and JITs fine to sm_121.
- `ffmpeg` must be on PATH for thumbnails / fallback decode.

## Stack at a glance

| concern | choice | file |
|---|---|---|
| Visual embeddings | V-JEPA 2 (`facebook/vjepa2-vitl-fpc16-256-ssv2`) | `src/ten/embed_video.py` |
| Captions / summaries | Qwen3-VL-8B-Instruct | `src/ten/caption.py` |
| ASR (opt-in) | Whisper-large-v3 via HF transformers (default) or faster-whisper | `src/ten/asr.py`, `src/ten/audio.py` |
| Caption embeddings | Qwen3-Embedding-0.6B (sentence-transformers) | `src/ten/embed_text.py` |
| Audio embeddings (opt-in) | LAION CLAP (`laion/clap-htsat-fused`) — bounded post-fusion reranker | `src/ten/embed_audio.py` |
| Vector store | Qdrant (2–3 collections; text+visual RRF-fused, audio reranks the head) | `src/ten/store.py`, `src/ten/search.py` |
| Ingest orchestrator | PyAV chunking → batched embed → upsert | `src/ten/ingest.py` |
| HTTP API + Range stream | FastAPI | `src/ten/api.py` |
| CLI | typer, entry point `ten` | `src/ten/cli.py` |
| Frontend | Next.js 15 App Router (static export) | `ui/` |

## Captioner backends

`src/ten/caption.py` exposes `CaptionerProtocol` with two implementations selected by `TEN_VLM_BACKEND`:

- `transformers` (default): in-process Qwen3-VL via HF `AutoModelForImageTextToText`.
- `vllm`: HTTP client to a vLLM OpenAI-compatible server — frames sent as multiple `image_url` blocks; concurrency knob feeds vLLM's continuous batching. Run via `docker compose --profile vllm up -d`.

When adding a new backend, implement `CaptionerProtocol` and register it in `make_captioner()`. Don't import the backend class anywhere else — call the factory.

## Conventions

- **Clip ids** are deterministic `blake2b(video_path | t_start | t_end)` hashes — see `Clip.clip_id` in `src/ten/video.py`. They map to Qdrant point ids via `clip_id_to_uuid`. This is what makes ingest resumable; don't change the hash inputs.
- **Two Qdrant collections** (`ten_visual`, `ten_text`) share an identical payload so a hit in either side renders the same UI card. Keep payload writes symmetric in `Store.upsert`. A third `ten_audio` collection appears only when `TEN_CLAP_BACKEND=clap` is set during ingest.
- **CLAP audio integration** is a *post-fusion reranker*, not a third RRF source. Equal-weight 3-way RRF regressed retrieval on the QVHighlights pilot (R@1 0.68→0.21) because CLAP's text encoder is trained for audio alignment, not general semantics. As a bounded reranker (top-K, score-threshold-gated, additive boost only — never demotes) it is at-worst a no-op on visual queries and provides real signal when the query genuinely concerns sound content. Knobs: `TEN_AUDIO_RERANK_{TOP_K,THRESHOLD,WEIGHT}`.
- **VAD pre-gates ASR.** Whisper hallucinates on music / wind / silence (`¶¶¶`, `Hjælp! Hjælp!`, etc.). Silero VAD runs in `src/ten/vad.py` before Whisper and short-circuits clips with < `TEN_VAD_MIN_SPEECH_FRACTION` (default 0.10) speech to an empty transcript. No effect when ASR is off. Cuts garbage transcript rate from ~35% to ~11% on outdoor/music-heavy content — see EVAL.md.
- **Library tag.** Each clip carries a `library` payload field derived from the parent directory of the source video (`videos/personal/foo.mp4` → `personal`). All three `Store.search_*` methods accept an optional `library=` kwarg that becomes a Qdrant `FieldCondition`. Plumbed through `Searcher.search`, REST `/search?library=...`, `ten search --library`, and the UI dropdown. Use it; don't grow ad-hoc `video_path` substring filters in callers.
- **Display rotation.** PyAV's `frame.to_ndarray()` returns raw codec pixels and ignores the container's display-matrix tag. `src/ten/video._video_rotation` reads the rotation via ffprobe (cached) and both `sample_clip_frames` and `write_thumbnail` apply `np.rot90` before any model sees the frame. Without this, every iPhone-portrait clip is ingested sideways — wrong captions, wrong embeddings, wrong thumbs. Don't bypass the helper when adding new decode paths.
- **MCP server** is mounted onto the FastAPI app at `/mcp/` (streamable-HTTP transport) and exposes `search` / `inspect_clip` / `list_clips` / `list_libraries` / `summarize_clip` / `index_status` as tools. Mounting requires chaining FastMCP's `session_manager.run()` into FastAPI's lifespan — see `_lifespan` in `src/ten/api.py`. If you add a new MCP tool, define it in `src/ten/mcp_server.py` with the `@mcp.tool()` decorator; the schema is generated from the signature + docstring. Don't import models eagerly in `mcp_server.py` — `_ensure()` lazy-initializes `Store`/`Searcher` so server start stays fast.
- **Lazy model loading**: every model wrapper loads on first use, behind a lock. Don't eagerly load in `__init__` — it makes the CLI/API slow to start and breaks `ten status`.
- **Config** is env-driven, single source of truth in `src/ten/config.py`. Add new knobs there, not as scattered constants.
- **No comments on what code does** — keep the codebase explanation in this file and the docstrings already at module tops.

## Common commands

The **Makefile** is the canonical task runner — it's the single place workflows are codified. Run `make help` for the full menu.

```bash
make bootstrap                                # uv sync + pnpm install + install-pc
make qdrant-up                                # start Qdrant (one-shot per machine boot)
make vllm-up                                  # optional: start vLLM (profile=vllm)
make index FOLDER=/path/to/videos             # ingest (resumable)
make index-vllm FOLDER=/path/to/videos        # same, but via vLLM backend
make search Q="a person catching a ball"      # CLI search
make serve                                    # FastAPI on :8765 (auto-mounts ui/out)
make serve-reload                             # serve with --reload (use directly, NOT in process-compose)
make ui-build                                 # build Next.js UI into ui/out
make ui-dev                                   # Next.js dev server on :3000, proxies /search etc
make dev                                      # process-compose TUI: api only
make dev-ui                                   # process-compose TUI: api + ui (UI on :3000)
make status                                   # print resolved config + qdrant stats
make lint   |   make format                   # ruff
```

## Local dev orchestration

`process-compose.yaml` runs **host processes only** (api, ui). Containers
(Qdrant, vLLM) live in `docker-compose.yml` and are managed independently.

Typical loop:

```bash
make qdrant-up           # once per boot — Qdrant stays up across dev sessions
make dev-ui              # TUI with api + Next dev server. Ctrl-C tears them down.
```

Conventions: every process invokes a `make` target (one canonical way to
run a service), `is_dotenv_disabled: true` so process-compose doesn't
auto-load `.env` into children, and `signal: 2` (SIGINT) for shutdown.
**Don't switch to SIGTERM** — uvicorn workers leak with SIGTERM under
`--reload`. The process-compose `api` deliberately omits `--reload` for the
same reason; for backend hot-reload, run `make serve-reload` in another
terminal instead.

When adding a new blessed workflow (new service, new ingest mode, new maintenance task), add a Makefile target with a `## description` doc-comment so it shows up in `make help`. Keep the section comments (`##@ Section`) in sync.

The raw `uv run ten ...` and `docker compose ...` commands still work — Makefile targets are thin wrappers, not gatekeepers.

## Gotchas

- **NVRTC arch error** (`invalid value for --gpu-architecture (-arch)`): PyTorch trying to JIT a kernel for sm_121 from a wheel that doesn't ship `compute_120` PTX. Fix is the cu129 index (already configured) — don't paper over with `TORCH_CUDA_ARCH_LIST`.
- **Qdrant is pinned to 1.12.4** (see `docker-compose.yml`). Newer images (≥1.18) refuse to load 1.12-written segments (`unknown variant 'on_disk'`). The qdrant-client warning about version skew is benign. To upgrade, do a clean re-index or follow Qdrant's migration guide; don't just bump the tag.
- **qdrant-client 1.18 dropped `.search()`** in favor of `.query_points(...).points`. The store layer already uses the new API; don't reintroduce `.search()`.
- **First ingest** downloads ~20 GB of weights (V-JEPA + Qwen3-VL + Qwen3-Embedding) into `~/.cache/huggingface`. Surface this clearly when asked about runtime.
- **Qdrant URL** defaults to `http://localhost:6333`. If `ten status` shows "Connection refused", the user hasn't started Docker yet.
- **`AutoVideoProcessor` import** in `embed_video.py` is from `transformers` ≥ 4.49. If transformers is downgraded for any reason, V-JEPA 2 won't load.
- **vLLM image** on DGX Spark: `vllm/vllm-openai:latest` may lag sm_100 (Blackwell) support. Fallback is NVIDIA's NGC build (`nvcr.io/nvidia/vllm:25.04-py3` or newer).
- **faster-whisper on aarch64** lacks GPU support — its CTranslate2 wheel ships CPU-only on ARM. Stay on `TEN_ASR_BACKEND=whisper` (HF transformers, runs on the existing torch CUDA). `fasterwhisper` is the right choice on x86_64 GPU boxes.
- The Next.js dev rewrites (`ui/next.config.mjs`) must mirror the FastAPI route prefixes. When adding a new endpoint family, add it to the `rewrites()` array too.
- UI is **static-export Next.js** (`output: 'export'`). All pages must be client components if they use state/effects (`"use client"` directive). Don't add server actions, route handlers in `app/api/`, or middleware — they'd require a Node runtime in prod and we want a single-process FastAPI deployment. `next build` writes to `ui/out/`, which `api.py` mounts.
