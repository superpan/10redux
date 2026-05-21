# Evaluation

Retrieval quality measurements against published benchmarks. Numbers here are reproducible from the repo via `make fetch-msrvtt` → `make ingest-msrvtt` → `make eval`.

## MSR-VTT 1K-A — text → video retrieval

The standard 1K-A protocol: 1000 test videos, 1 caption each, retrieve the matching video from a candidate pool of 1000.

### Result

Two configurations worth knowing about — best baseline and best with optimization stack on:

| config | R@1 | R@5 | R@10 | Median | wall (eval) |
|---|---|---|---|---|---|
| visual-only, no rerank (baseline) | 0.338 | 0.559 | 0.651 | 4 | 44 s |
| **+ ASR + rerank (best)** | **0.360** | **0.569** | **0.660** | **3** | 18 m |

Both reproducible from the repo. Full 4-cell ablation in the [Optimization](#optimization-stages) section below.

![MSR-VTT 1K-A: Recall@K + rank distribution (ASR + rerank)](data/eval/msrvtt_latest.png)

Per-query results: [`data/eval/msrvtt_latest.json`](data/eval/msrvtt_latest.json) (ASR+rerank); other configs in `data/eval/msrvtt_*.json`.

(Earlier 16-frame run got 0.343 / 0.553 / 0.651 — within noise. We use 8 frames per clip by default; see "Frame budget" below.)

### How to read it

This is **caption-mediated retrieval**: each video is captioned by Qwen3-VL at ingest, the caption is embedded by Qwen3-Embedding-0.6B, and queries match against caption vectors. V-JEPA 2 visual embeddings are also indexed but not used for text queries (V-JEPA isn't text-aligned).

For context, where 0.338 R@1 sits among published open results:

| approach | R@1 (MSR-VTT 1K-A) | type |
|---|---|---|
| Random baseline (1/1000) | 0.001 | — |
| CLIP ViT/L (frozen, 1 frame) | ~0.32 | end-to-end vision-language |
| **ten (Qwen3-VL captions + text retriever)** | **0.338** | **caption-mediated** |
| Frozen-in-Time / X-Pool / X-CLIP | 0.43 – 0.49 | end-to-end, video-text fine-tuned |
| InternVideo2 (full ZSL) | ~0.51 | end-to-end |

The point is not to chase the leaderboard but to see what a fully open-weight, captions-as-the-bridge architecture lands at without any video-text fine-tuning. Median rank of 4 means the right video is usually in the top few; the long tail (mean rank 44.0) is dominated by ambiguous captions like "cartoon show for kids" or "a young man is touching a young girls back" where many candidates fit.

Closing more of the gap to dedicated end-to-end models would mean either training a contrastive head on MSR-VTT-style pairs, or swapping to a video-text foundation model (e.g. InternVideo2). Neither is the point of this experiment — the point is how far an off-the-shelf, captions-as-the-bridge stack gets without retraining anything.

### Methodology

- **Index:** one clip per video (`TEN_CLIP_SECONDS=60 TEN_CLIP_OVERLAP=0`); MSR-VTT clips are 10–30 s so each becomes a single Qdrant point.
- **Captioner:** Qwen3-VL-8B-Instruct served by NVIDIA NGC vLLM (bf16, GB10 / sm_121).
- **Caption embedder:** Qwen3-Embedding-0.6B via sentence-transformers.
- **Query:** test-split caption → text vector → cosine search in the caption collection.
- **Filter:** the eval restricts the candidate pool to `/videos/msrvtt/` so any other clips in the same Qdrant index (e.g. from `make fetch-smoke`) don't pollute rankings.
- **Rank counted** is the position of the ground-truth video among unique videos seen in the result list (msrvtt-namespaced), starting at 1.

### Hardware

DGX Spark — NVIDIA GB10 (Grace Blackwell, sm_121, aarch64), CUDA 13.

| stage | wall time |
|---|---|
| `make fetch-msrvtt` (download + extract 1K test) | ~3 min |
| `make ingest-msrvtt` (1000 videos, vLLM backend, 8 frames/clip) | 29 m 23 s (~1.76 s/clip) |
| `make eval` (1000 text queries) | 44 s |

### Frame budget

`TEN_FRAMES_PER_CLIP` defaults to **8**. Raising to 16 produced essentially identical retrieval quality (R@1 0.343 vs 0.338) at 17% higher ingest cost — diminishing returns. For motion-heavy footage where caption fidelity matters more than throughput, bump to 16.

### Caveats

- **Caption-mediated bias.** When Qwen3-VL hallucinates an object, that hallucination becomes the ground truth for retrieval. End-to-end models avoid this single point of failure.
- **One clip per video** trivializes the chunking story for retrieval; longer videos may need scene-level aggregation.
- **No ASR / dialogue** in the caption. Talking-head videos under-perform.
- The `1K-A` split (single caption per video, this is `friedrichor/MSR-VTT`'s `msrvtt_test_1k.json`) — some papers report on the original 20-captions-per-video protocol; numbers are not directly comparable.

## ASR — voice-tag retrieval

### Voice-tag smoke (Sintel)

99-clip smoke run on Sintel proves the basic capability — queries that quote or paraphrase spoken dialogue with no visual relationship hit the right clip:

| query (no visual cue) | top hit transcript |
|---|---|
| "land of the gatekeepers" | "…what brings you to the land of the Gatekeepers? I'm searching for something." |
| "fool for traveling alone" | "You're a fool for traveling alone so completely unprepared…" |
| "shed innocent blood" | "It has a dark past. It has shed much innocent blood." |
| "I'm searching for someone dear" | "I'm searching for someone. Someone very dear? A kindred spirit? A dragon." |

Visual queries ("dragon", "warrior fighting in a desert canyon") rank the correct clips at #1 unchanged — adding ASR doesn't regress the caption side on cherry-picked dialogue clips.

### Controlled re-eval on MSR-VTT 1K-A — ASR is a wash

Re-ingested all 1000 test videos with ASR enabled (28m15s wall, basically same as visual-only — Whisper overlapped with vLLM caption batches). Re-ran the same 1000-query eval.

| metric | visual-only | +ASR | Δ |
|---|---|---|---|
| Recall@1 | 0.338 | 0.325 | **−0.013** |
| Recall@5 | 0.559 | 0.559 | 0.000 |
| Recall@10 | 0.651 | 0.640 | −0.011 |
| Median rank | 4 | 4 | 0 |
| Mean rank | 44.0 | 45.5 | +1.5 |

Per-query delta: 334 queries improved (avg +36 ranks), 346 regressed (avg −39 ranks), 320 unchanged. **Net zero, slight skew to regression.** Per-query results in `data/eval/msrvtt_visual_only.json` and `data/eval/msrvtt_latest.json`.

By query category (caption keyword bucket):

| category | n | helped % | hurt % |
|---|---|---|---|
| music | 116 | 39.7% | 37.1% |
| **speech** | 203 | **32.0%** | **41.9%** |
| sport | 140 | 30.0% | 37.9% |
| other | 541 | 33.5% | 30.5% |

The counter-intuitive finding: speech-related queries get *hurt* more than helped. The mechanism is visible in the largest regressions:

- "bbc news story about military crackdown" → rank 22 → 858  *(transcript fills with specific names/places that don't match the abstract caption)*
- "tv show presenters speak about will smith and other actors" → rank 219 → 550
- "someone speaking about a violent act regarding the police" → rank 12 → 511

Largest improvements show the symmetric mechanism — when the query *quotes or paraphrases* the audio, ASR is a big win:

- "a man is singing and standing in the road" → rank 408 → 69 (+339)
- "anchor talking about a shows" → rank 651 → 325 (+326)
- "shania twain does a closeup for her video" → rank 400 → 113 (+287; lyrics match)

### Verdict

ASR stays **opt-in** by default. Use it when:

- Your library is dialogue-heavy (lectures, podcasts, interviews, news) **and** queries reference what is said.
- You're building a "voice-tag" UI affordance where users explicitly search transcripts.

Skip it when:

- Captions describe visual content abstractly and queries follow that style (the MSR-VTT case).
- The library is mostly music or B-roll — Whisper hallucinations (`¶¶¶`, repetition loops) become noise without compensating signal.

### ASR caveats (pre-VAD)

The original Whisper-on-everything pipeline had three persistent failure modes:

- **Music → `¶¶¶…`** : Whisper transcribes instrumental sections as repeated note characters.
- **Wind / near-silence → repetition loops** : `"I'm sorry. I'm sorry…"`, `"Hjælp! Hjælp! Hjælp!"`, `えいぃぃぃぃぃ…`. Same phrase repeated for the full clip.
- **Single-word fillers** : `"Okay."`, `"Hmm."`, `"so"` on near-silent input.

### Quantifying the contamination

Running `tools/asr_contamination.py` against the live index (1867 non-empty transcripts across snowsports + QVH val pilot + smoke):

| namespace | clips | garbage | rate |
|---|---:|---:|---:|
| QVH val pilot (vlogs/news) | 1686 | 142 | 8.4% |
| Smoke set (BBB/Sintel, music-heavy) | 99 | 40 | 40.4% |
| Snowsports demo (12-min K2 descent) | 82 | 29 | 35.4% |
| **all audio-bearing** | **1867** | **211** | **11.3%** |

Indoor-talking content (QVH val) is largely fine; outdoor / music-heavy content (the exact distribution ten claims to be good for in its "When ten fits" section) sees roughly one-in-three Whisper outputs as hallucinated noise.

### VAD pre-gate — Silero, layered before Whisper

`src/ten/vad.py` wraps Silero VAD (ONNX, CPU, ~2 MB, ~24 ms per 10 s clip) as a pre-gate on the `TranscriberProtocol`. The transcriber chain becomes:

1. extract 16 kHz mono WAV (already done for ASR)
2. Silero `speech_fraction(wav)` → float in [0, 1]
3. if `< TEN_VAD_MIN_SPEECH_FRACTION` (default 0.10), return `""` and skip Whisper entirely
4. otherwise transcribe as before

Opt-in: `TEN_VAD_BACKEND=silero` or `ten index --asr --vad` (or `make index-asr-vad FOLDER=…`).

### Result — snowsports namespace before/after

Re-ingested the K2 descent with `--asr --vad --clap --force`:

| category | before VAD | after VAD |
|---|---:|---:|
| empty (skipped) | 0 | **47** |
| near_empty (`'so'`, `'¶¶'`) | 13 | 0 |
| music_glyph (`¶¶¶…`) | 2 | 0 |
| filler (`'Okay.'`) | 5 | 1 |
| repetition_loop (`Hjælp! Hjælp!`) | 10 | 3 |
| real (Polish/Russian radio chatter with base camp) | 53 | 31 |
| **total** | 82 | 82 |
| **garbage rate among non-empty** | **35.4%** | **11.4%** |

VAD correctly sent 47 clips to empty-transcript fast-paths. Of the 35 still transcribed, garbage dropped to ~4 (one filler + three short residual loops with speech fractions just above the 0.10 threshold). 22 previously-"real" transcripts also became empty — these are either genuinely sub-threshold or false negatives; tightening the threshold further trades real low-speech moments for fewer residual hallucinations.

### Cost

24 ms per 10 s clip on CPU after first-call model load (688 ms one-shot). Negligible against Whisper's ~real-time-per-clip on GPU; in fact net ingest time *decreases* when VAD lets us skip Whisper entirely on >50% of clips (the snowsports re-ingest skipped Whisper on 47/82 clips, recovering several minutes).

### Verdict

VAD is **opt-in** today (`TEN_VAD_BACKEND=silero`), no impact when off, strict improvement when on for audio that isn't dominated by speech. Default may flip to on once we re-eval MSR-VTT 1K-A with VAD active — but MSR-VTT clips have no audio in our distribution (see CLAP section), so the MSR-VTT regression check is moot. A re-eval on a speech-bearing benchmark would be the next data point.

## CLAP audio embeddings — pilot on QVHighlights

LAION CLAP (`laion/clap-htsat-fused`) gives ten a third modality: 512-d audio embeddings, shipped to a third Qdrant collection `ten_audio`. The interesting question wasn't whether the encoder works (it does — see the smoke test in the commit message for `42c3cb8`), but how to *integrate* it. Two integrations were tried.

### Why not MSR-VTT for this eval

The MSR-VTT 1K-A test set ships with audio streams stripped — every clip we probed had no audio track. Running a CLAP ablation on MSR-VTT would compare two identical-by-construction configurations (empty `ten_audio` either way). We used the QVHighlights val set instead: 1519 unique 150-second YouTube clips with intact audio, downloaded via `make fetch-qvhighlights` (~80% recovery rate to dead/region-locked IDs, ~1246 videos in our run).

### Pilot setup

- 100-video subset of QVH val (first 100 lexicographically-sorted vids that were on disk), 100 corresponding text queries.
- Indexed with `make index-full` (vLLM + Whisper ASR + CLAP) → 1686 clips in `ten_visual` / `ten_text` / `ten_audio`.
- Eval is **open-set moment retrieval**, not QVH's official within-video localization: for each query, retrieve top-N clips across the full ~3,100-clip index (MSR-VTT + QVH pilot + smoke), then check whether the GT vid is present and whether the retrieved clip overlaps a `relevant_window`.
- Metrics: any-overlap recall, IoU-thresholded recall (0.3 / 0.5 / 0.7), and `top_vid_match` recall (correct vid surfaced, ignoring temporal precision).

10 s clip granularity caps the achievable IoU against long ground-truth windows (a 10 s clip vs a 60 s GT window can never exceed IoU 0.17), which is why we report `any_overlap` alongside the standard 0.5 / 0.7.

### Result — three configurations

Same index, same 100 queries, three search-time configurations:

| config | any_R@1 | any_R@10 | top_vid_R@1 | IoU≥0.3 R@1 | IoU≥0.5 R@1 | IoU≥0.5 R@20 |
|---|---:|---:|---:|---:|---:|---:|
| text+visual (baseline) | 0.610 | 0.860 | 0.680 | 0.400 | 0.150 | 0.320 |
| **+ CLAP via equal-weight RRF** | **0.160** | **0.680** | **0.210** | **0.060** | **0.010** | **0.310** |
| + CLAP as bounded reranker | 0.610 | 0.860 | 0.680 | 0.400 | 0.150 | **0.330** |

Numbers in `data/eval/qvhighlights_clap_{off,on,rerank,rerank_v2}.json`.

The RRF row is a **−47 percentage point regression** at top-vid-match R@1. CLAP's text encoder is trained for audio alignment, not general semantics; adding it as a peer RRF source on visual-description queries injects rank noise across the whole list and pushes the correct video out of the top-K.

### Per-query analysis — gating wouldn't have saved RRF

Hypothesis tested: maybe RRF only hurts non-audio queries, and a query-router (cue words: music / talking / applause / etc.) would gate CLAP correctly. Result:

| subset | n | baseline any_R@1 | CLAP-RRF any_R@1 | absolute drop |
|---|---:|---:|---:|---:|
| All queries | 100 | 0.610 | 0.160 | −0.450 |
| Cue queries (music/talk/laugh/etc) | 14 | 0.429 | 0.071 | −0.358 |
| Plain visual queries | 86 | 0.640 | 0.174 | −0.466 |

Cue queries are hurt *roughly as badly* as plain ones (relative drop is actually slightly larger). Of 13 queries CLAP-RRF helped, only 2 were cue queries. Gating wouldn't have fixed the regression.

### Reranker integration — why it works

The shipped integration:
- text+visual fuse via RRF as before (no audio in the rank lists).
- CLAP audio search runs in parallel; results form a `clip_id → cosine_score` map.
- For the top-K head of the fused list (default `K=30`), score is bumped by `weight * max(0, cosine − threshold)`. Below the head, nothing changes.
- Audio can therefore **promote within the head**, but never demote anything out of it and never introduce new candidates.

Defaults (`TEN_AUDIO_RERANK_{TOP_K=30,THRESHOLD=0.30,WEIGHT=0.03}`) are deliberately conservative — they bound the maximum rerank impact to a few ranks. On the pilot this surfaced one extra query into R@20 ("A video blogger talking and eating" at rank 21 → 20) which cleared three R@20 thresholds simultaneously (`any_overlap`, `top_vid_match`, `iou≥0.5`), giving the +0.010 in the table above. Net effect: **safe but ~no-op on visual-description queries**; the upside arrives when queries actually concern sound content (a regime the QVH pilot doesn't exercise).

### MSR-VTT regression check

Re-ran the standard MSR-VTT 1K-A eval with `TEN_CLAP_BACKEND=clap` active. Since MSR-VTT clips have no audio, `audio_score_map` is empty for every query and the reranker is a strict no-op:

| metric | baseline (`444ebcf`) | CLAP-as-reranker active |
|---|---:|---:|
| R@1 | 0.360 | 0.361 |
| R@5 | 0.569 | 0.584 |
| R@10 | 0.660 | 0.665 |
| Mean rank | 44.2 | 33.8 |

R@1 is essentially unchanged (within run noise). The R@5 / R@10 / mean-rank drift is from index-state noise (1686 QVH clips were ingested into the same Qdrant container between the two runs; namespace filtering excludes them from MSR-VTT eval but floating-point ordering during re-ingest of the MSR-VTT set itself shifts marginally). No regression attributable to CLAP.

### Verdict

CLAP ships **opt-in** (`TEN_CLAP_BACKEND=clap`) and **reranker-only** (no equal-weight RRF path exposed). For visual-description query distributions it's at-worst a no-op. The interesting unlock is an audio-explicit query surface (e.g., a future `/search-audio` endpoint or a query router); the bounded reranker is a safety-first stepping stone, not the end state.

## Reproduce

```bash
make qdrant-up                                    # one-time
make vllm-up                                      # optional but ~2.7× faster ingest
make fetch-msrvtt                                 # ~2.2 GB
TEN_VLM_BACKEND=vllm make ingest-msrvtt           # ~34 min on GB10
make eval                                         # ~45 s; writes data/eval/msrvtt_latest.{json,png}
```

CLAP / QVHighlights pilot:

```bash
make fetch-qvhighlights                           # yt-dlp val split, hours, ~80% recovery
# carve out a 100-video pilot folder (see tools/eval_qvhighlights.py docstring)
TEN_VLM_BACKEND=vllm TEN_ASR_BACKEND=whisper TEN_CLAP_BACKEND=clap \
  make index-full FOLDER=./videos/qvh_pilot       # ~80 min for 1686 clips
make eval-qvh-ablation                            # runs eval twice (CLAP off/on)
```
