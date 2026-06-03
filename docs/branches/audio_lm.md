# Branch: `audio-lm`  · status: spike — NOT FOR MERGE

Sketched the architectural shift to a single audio LM replacing Whisper + CLAP + Silero VAD. The wiring is functional but the model choices we could actually load on the Hub today don't deliver the captioning quality the architecture requires.

This branch stays available for context; **don't merge as-is**. The next attempt should either vendor MOSS-Audio's `modeling_moss_audio.py` from their GitHub or wait for the Hub snapshot to ship it.

---

## What this branch tried

Replaces ten's chained Whisper + CLAP + Silero VAD audio stack with a single audio language model.

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

## Smoke-test outcome (2026-06-03)

Smoke-tested both candidate backends against five known snowsports clips (helmet/speech, wind+climber-call, scenic, quiet ridge, skis-on-snow). Neither was usable:

### MOSS-Audio 4B Instruct — blocked on Hub packaging

`OpenMOSS-Team/MOSS-Audio-4B-Instruct` ships `configuration_moss_audio.py` and `processing_moss_audio.py` in the Hub snapshot, but **not `modeling_moss_audio.py`** (only exists in their GitHub repo). The model's `auto_map` only registers AutoConfig + AutoProcessor — no AutoModel. Result: `AutoModel*.from_pretrained` can't instantiate.

Workarounds, neither taken: (a) clone OpenMOSS/MOSS-Audio and put `src/` on PYTHONPATH; (b) vendor `modeling_moss_audio.py` + `audio_io.py` into `src/ten/_vendor/moss_audio/`. Both pull in their full dependency tree. The right fix is upstream — wait for them to either land `modeling_moss_audio.py` in the snapshot or add an AutoModel entry to `auto_map`. MOSS-Audio is architecturally the right call (Qwen3 backbone aligns with the rest of ten; audio captioning is its primary capability per the model card) and remains the recommended backend once their packaging is clean.

### Voxtral 3B — wrong model for the use case

`mistralai/Voxtral-Mini-3B-2507` loads cleanly via the official `VoxtralForConditionalGeneration` class. Transcription quality on snowsports clips is comparable to Whisper-large-v3 — fine but no improvement. The caption path is the problem: Voxtral 3B is positioned as transcription + voice-assistant (function calling from voice, Q&A over user-supplied audio), not as an "describe this acoustic scene" model.

Caption smoke against five known clips:

| prompt | t=18 (real speech) | t=99 (wind+climber) | t=153 (scenic) | t=558 (quiet ridge) | t=603 (skis-on-snow) |
|---|---|---|---|---|---|
| Loose prompt | "monologue" | "monologue with pauses" | **"monologue with traffic sounds"** ❌ | "monologue" | "monologue" |
| Tight prompt | "dominant sound is speech" | "dominant sound is speech" | "dominant sound is environmental" | "dominant sound is environmental/silence" | "dominant sound is environmental/silence" |

Loose prompts default to "monologue" on non-speech content. Tight prompts produce stripped category labels rather than acoustic descriptions ("wind across snow", "skis cutting ice") — the model doesn't have the fine-grained acoustic-description capability the architecture relies on. Indexing every clip with `"monologue"` or `"environmental"` would actively pollute `ten_text` clustering — strictly worse than the current Whisper+CLAP+VAD state where non-speech clips correctly return empty transcripts.

### Verdict

The architectural pattern (single audio LM producing transcript + caption into the text channel) is sound. The model layer isn't there yet on usable open-weight options.

**Next steps**, in priority order:

1. **Vendor MOSS-Audio's modeling file** when ready to invest ~60 min: copy `src/modeling_moss_audio.py` + `src/audio_io.py` from the OpenMOSS GitHub repo into `src/ten/_vendor/moss_audio/`, register them via `auto_map`, retry the smoke. Their model card frames audio captioning as a primary capability, so the caption quality should be qualitatively different from Voxtral's.
2. **Skip the per-clip audio LM** and try the alternative recommendation: **Voxtral 24B (or MOSS-Audio 8B) as a video-level audio-summary pass**. Voxtral's strength is long-form Q&A (32k context = 30 min audio), not per-10s-clip acoustic description. A separate per-video pass that emits one summary sentence per video might land better than per-clip captions for either model. This closes effect-1 of the Marengo gap (clip-level captioning misses video-level gestalt) rather than effect-2.
3. **Accept that the open-weight audio LM ecosystem isn't ready yet** and double down on a different EVAL.md recommendation — query-routing CLAP from reranker to peer source when audio cues are detected. Smaller architectural change, partial credit, available today.
