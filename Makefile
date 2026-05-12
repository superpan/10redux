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

install: ## Install Python deps via uv (PyTorch from cu128 index, aarch64-friendly)
	uv sync

ui-install: ## Install JS deps for the React frontend
	cd ui && npm install

ffmpeg: ## Install ffmpeg via apt (asks for sudo password)
	sudo apt install -y ffmpeg

bootstrap: install ui-install ## One-shot: Python deps + UI deps

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

fetch-smoke: ## Download 5 public-domain videos to ./videos/smoke for smoke testing
	uv run python tools/fetch_smoke.py

##@ Index / search

index: ## Ingest a folder of videos (FOLDER=...). Resumable.
	uv run ten index $(FOLDER)

index-vllm: ## Ingest using the vLLM captioner backend (~3-5x faster)
	TEN_VLM_BACKEND=vllm uv run ten index $(FOLDER)

reindex: ## Re-embed every clip in FOLDER, ignoring existing ids
	uv run ten index $(FOLDER) --force

smoke: ## Index just the first 3 videos in FOLDER (sanity check)
	uv run ten index $(FOLDER) --max-videos 3

search: ## CLI search (Q="..."). Set LIMIT=n for more results.
	uv run ten search "$(Q)" --limit $(LIMIT)

status: ## Print resolved config and Qdrant stats
	uv run ten status

##@ Run

serve: ## Run FastAPI on :$(PORT). Auto-mounts ui/dist if built.
	uv run ten serve --port $(PORT)

serve-reload: ## Same as `serve` but reloads on Python changes
	uv run ten serve --port $(PORT) --reload

ui-build: ## Build the React UI into ui/dist (served by `make serve`)
	cd ui && npm run build

ui-dev: ## Run the Vite dev server on :5173, proxying /search etc to :$(PORT)
	cd ui && npm run dev

dev: ## Tip: open two terminals -> `make serve-reload` and `make ui-dev`
	@echo 'Open two terminals:'
	@echo '  T1:  make serve-reload'
	@echo '  T2:  make ui-dev'
	@echo 'Then browse http://127.0.0.1:5173'

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

.PHONY: install ui-install ffmpeg bootstrap \
        qdrant-up qdrant-down vllm-up vllm-down vllm-logs services-up services-down \
        fetch-smoke \
        index index-vllm reindex smoke search status \
        serve serve-reload ui-build ui-dev dev \
        lint format clean-thumbs clean-qdrant help
