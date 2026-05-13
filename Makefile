# ten — common workflows. Run `make` (or `make help`) for the menu.

.DEFAULT_GOAL := help
SHELL := bash
.ONESHELL:

# Override with: `make index FOLDER=/data/videos` or `make search Q="query"`
FOLDER ?= ./videos
Q      ?= a person catching a ball at sunset
LIMIT  ?= 20
PORT   ?= 8765

##@ Setup

install: ## Install Python deps via uv (PyTorch from cu129 index, aarch64-friendly)
	uv sync

ui-install: ## Install JS deps for the Next.js frontend (pnpm via corepack)
	cd ui && pnpm install

ffmpeg: ## Install ffmpeg via apt (asks for sudo password)
	sudo apt install -y ffmpeg

install-pc: ## Install latest process-compose to ~/.local/bin (official installer)
	mkdir -p $$HOME/.local/bin
	sh -c "$$(curl --location https://raw.githubusercontent.com/F1bonacc1/process-compose/main/scripts/get-pc.sh)" -- -d -b $$HOME/.local/bin
	@echo "ensure ~/.local/bin is on PATH"

bootstrap: install ui-install install-pc ## One-shot: Python + UI + process-compose

##@ Services

qdrant-up: ## Start Qdrant (foreground -d)
	docker compose up -d qdrant

qdrant-down: ## Stop Qdrant
	docker compose stop qdrant

vllm-up: ## Start vLLM Qwen3-VL server (profile=vllm)
	docker compose --profile vllm up -d vllm

vllm-down: ## Stop vLLM
	docker compose --profile vllm stop vllm

vllm-logs: ## Tail vLLM logs
	docker compose --profile vllm logs -f vllm

services-up: qdrant-up ## Bring up all required services (qdrant only by default)

services-down: ## Stop everything Docker-managed
	docker compose --profile vllm down

##@ Datasets

fetch-smoke: ## Download 4 public-domain videos to ./videos/smoke for smoke testing
	uv run python tools/fetch_smoke.py

fetch-msrvtt: ## Download MSR-VTT (videos + 1K-A test split) for retrieval eval (~2.2 GB)
	uv run python tools/fetch_msrvtt.py

eval: ## Run MSR-VTT 1K-A text->video retrieval eval; writes data/eval/msrvtt_<ts>.json
	uv run python tools/eval_msrvtt.py

ingest-msrvtt: ## Ingest MSR-VTT with one-clip-per-video chunking (use after fetch-msrvtt)
	TEN_CLIP_SECONDS=60 TEN_CLIP_OVERLAP=0 uv run ten index ./videos/msrvtt

##@ Index / search

index: ## Ingest a folder of videos (FOLDER=...). Resumable.
	uv run ten index $(FOLDER)

index-vllm: ## Ingest using the vLLM captioner backend (~3-5x faster)
	TEN_VLM_BACKEND=vllm uv run ten index $(FOLDER)

index-asr: ## Ingest with Whisper ASR transcripts (FOLDER=..., uses vLLM if available)
	TEN_VLM_BACKEND=vllm TEN_ASR_BACKEND=whisper uv run ten index $(FOLDER) --asr

reindex: ## Re-embed every clip in FOLDER, ignoring existing ids
	uv run ten index $(FOLDER) --force

smoke: ## Index just the first 3 videos in FOLDER (sanity check)
	uv run ten index $(FOLDER) --max-videos 3

search: ## CLI search (Q="..."). Set LIMIT=n for more results.
	uv run ten search "$(Q)" --limit $(LIMIT)

status: ## Print resolved config and Qdrant stats
	uv run ten status

##@ Run

serve: ## Run FastAPI on :$(PORT). Auto-mounts ui/out (Next export) if built.
	uv run ten serve --port $(PORT)

serve-reload: ## Same as `serve` but reloads on Python changes
	uv run ten serve --port $(PORT) --reload

ui-build: ## Build the Next.js UI into ui/out (served by `make serve`)
	cd ui && pnpm build

ui-dev: ## Run the Next.js dev server on :3000, proxying /search etc to :$(PORT)
	cd ui && TEN_API=http://127.0.0.1:$(PORT) pnpm dev

dev: ## TUI dev session: api in process-compose. Run `make qdrant-up` first.
	process-compose up api

dev-ui: ## TUI dev session: api + Next.js dev server. UI on http://127.0.0.1:3000
	process-compose up api ui

##@ Maintenance

lint: ## Ruff lint
	uv run ruff check .

format: ## Ruff format
	uv run ruff format .

clean-thumbs: ## Delete cached thumbnails (next /thumb call regenerates them)
	rm -rf .ten/thumbs

clean-qdrant: ## DESTRUCTIVE: wipe Qdrant storage on disk
	@read -p "Delete ./qdrant_storage? Type yes: " ans && [ "$$ans" = "yes" ]
	docker compose stop qdrant || true
	rm -rf ./qdrant_storage

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n\nVariables (override on the CLI):\n  FOLDER=./videos   Q=\"a query\"   LIMIT=20   PORT=8765\n\n"} \
		/^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

.PHONY: install ui-install ffmpeg install-pc bootstrap \
        qdrant-up qdrant-down vllm-up vllm-down vllm-logs services-up services-down \
        fetch-smoke fetch-msrvtt eval ingest-msrvtt \
        index index-vllm index-asr reindex smoke search status \
        serve serve-reload ui-build ui-dev dev dev-ui \
        lint format clean-thumbs clean-qdrant help
