# Branch: `audio-lm`

Replaces ten's chained Whisper + CLAP + Silero VAD audio stack with a single audio language model (default: **MOSS-Audio 4B Instruct**, with a stub for **Voxtral 3B**).

## Why

The QVHighlights eval against TwelveLabs Marengo 3.0 (see `EVAL.md` §"External baseline") showed that **3 of the 9 R@1 gap queries** are *abstract speech-act semantics* — *"monologue"*, *"sing and play music"*, *"talking and eating"*. The current stack can't surface those:

- Whisper transcribes **literal words spoken** — *"hi guys today…"* never matches the query word *"monologue"*.
- CLAP **could** match those semantically — but ten runs it as a bounded reranker (top-K=30, threshold-gated), so it never reaches GT videos that aren't already in the text+visual top-30.
- Silero gates Whisper; it doesn't add semantic signal.

An audio LM produces an **abstract audio caption** (*"a woman gives a sustained monologue in an intimate setting"*) that the text embedder picks up the same way it picks up Qwen3-VL's visual captions. Architectural symmetry: visual caption ⊕ audio caption → `ten_text`.

## What's on this branch

- **`src/ten/audio_lm.py`** — new module. `AudioLMProtocol` (transcribe + caption + describe) and `MOSSAudioLM` (wraps `OpenMOSS-Team/MOSS-Audio-4B-Instruct` via `transformers.AutoModelForCausalLM` + `AutoProcessor`). Factory `make_audio_lm()` gated on `TEN_AUDIO_LM_BACKEND`.
- **`src/ten/config.py`** — new knobs: `audio_lm_backend`, `audio_lm_model`, `audio_lm_max_new_tokens`, `audio_lm_transcribe_prompt`, `audio_lm_caption_prompt`.
- **`src/ten/ingest.py`** — dispatch. When `audio_lm` is set, the legacy ASR + CLAP + VAD branches are **skipped entirely** for that ingest run. Both transcript and audio_caption flow into `ten_text`'s embed input. `ClipPayload.audio_caption` is the new persisted field.
- **`src/ten/store.py`** — `ClipPayload.audio_caption: str = ""` (default-safe; old payloads round-trip).
- **`src/ten/cli.py`** — `ten index --audio-lm` flag.
- **`Makefile`** — `make index-audio-lm FOLDER=…` target.

## What's NOT on this branch

Deliberately kept around so we can A/B before merging:

- `src/ten/asr.py`, `src/ten/embed_audio.py`, `src/ten/vad.py` — untouched. They keep working when `TEN_AUDIO_LM_BACKEND` is unset.
- The `ten_audio` Qdrant collection and CLAP-rerank logic in `Searcher` — unchanged.
- The Whisper / CLAP / VAD env vars in config — unchanged.

After merge with a clean eval win, the cleanup commit should:

- Delete the three modules above.
- Remove `make_audio_embedder()` call sites and the audio-rerank block from `Searcher`.
- Drop `ten_audio` from `Store.ensure_collections` (with a migration path for existing deployments).
- Remove `TEN_ASR_*`, `TEN_VAD_*`, `TEN_CLAP_*`, `TEN_AUDIO_RERANK_*` from `config.py`.
- Update `EVAL.md` "ASR" + "CLAP" sections to a single "Audio LM" section with the new numbers.

## What's untested

- **The model has not been downloaded.** No actual `MOSS-Audio-4B-Instruct` inference has been run on this branch. The `_generate` method uses the common multimodal-processor pattern (`apply_chat_template` with `audio` + `text` content blocks) which is consistent with similar models on the Hub, but the exact tensor names + chat template may need adjustment to match the upstream `infer.py`.
- **The audio_caption prompt is a first guess.** It should be tuned against real clips before any eval.

## How to actually run it

No predownload needed — the default `TEN_AUDIO_LM_MODEL=OpenMOSS-Team/MOSS-Audio-4B-Instruct` is a HF Hub ID, and `AutoModelForCausalLM.from_pretrained` lazy-downloads to `~/.cache/huggingface/hub/` on first use. Same pattern as every other model in ten (V-JEPA, Whisper, CLAP, Qwen3-Embedding).

```bash
# Smoke test on the snowsports demo (82 clips). First run downloads ~8 GB
# of MOSS-Audio weights into ~/.cache/huggingface; subsequent runs warm-load.
make index-audio-lm FOLDER=./videos/snowsports

# Then a search where the audio caption should help:
ten search "a sustained monologue" --library snowsports
ten search "wind whistling over mountains" --library snowsports
```

If you want to pin a specific local checkpoint (e.g., a fine-tuned MOSS variant), point `TEN_AUDIO_LM_MODEL` at the path:

```bash
export TEN_AUDIO_LM_MODEL=/path/to/my-finetune
```

## Eval gate before merge

This branch should not merge until at least one of:

1. **Re-run the TwelveLabs eval comparison** on the QVHighlights pilot (re-ingest with `--audio-lm`, re-query, compare top-vid R@K against the current baseline in `data/eval/twelvelabs_qvh_pilot.json`). Goal: close ≥ 2 of the 3 abstract-speech-act queries from the current gap.
2. **Personal-library qualitative check** — verify that *"a woman giving a monologue"* / *"singing"* / *"music with guitar"* queries surface meaningfully better top-1 results than current.

Either result gates whether to merge or kill the branch. If audio LM doesn't measurably help on the gap queries, the chained encoders are working better than we think and the right move is the *Voxtral 3B video-level audio-summary* path instead (recommendation #2 from the original analysis).
