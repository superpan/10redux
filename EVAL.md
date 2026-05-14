# Evaluation

Retrieval quality measurements against published benchmarks. Numbers here are reproducible from the repo via `make fetch-msrvtt` → `make ingest-msrvtt` → `make eval`.

## MSR-VTT 1K-A — text → video retrieval

The standard 1K-A protocol: 1000 test videos, 1 caption each, retrieve the matching video from a candidate pool of 1000.

### Result

| metric | value |
|---|---|
| Recall@1   | **0.338** |
| Recall@5   | **0.559** |
| Recall@10  | **0.657** |
| Median rank | 4 |
| Mean rank   | 44.0 |
| Queries (n) | 1000 |
| Wall time   | 44 s |

(Earlier 16-frame run got 0.343 / 0.553 / 0.651 — within noise. We use 8 frames per clip by default; see "Frame budget" below.)

![MSR-VTT 1K-A: Recall@K + rank distribution](data/eval/msrvtt_latest.png)

Per-query results: [`data/eval/msrvtt_latest.json`](data/eval/msrvtt_latest.json).

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

### ASR caveats

- **Music → `¶¶¶…`** : Whisper transcribes instrumental sections as repeated note characters. The VAD filter helps but doesn't catch all of them.
- **Repetition loops** : on quiet / ambiguous audio, Whisper sometimes outputs the same phrase ("I'm sorry. I'm sorry…") for the entire clip. Known Whisper failure mode.
- For serious deployment with ASR on, post-process transcripts to detect and drop clips where >50% of tokens are repeated. Likely to recover some of the −0.013 R@1 regression.

## Reproduce

```bash
make qdrant-up                                    # one-time
make vllm-up                                      # optional but ~2.7× faster ingest
make fetch-msrvtt                                 # ~2.2 GB
TEN_VLM_BACKEND=vllm make ingest-msrvtt           # ~34 min on GB10
make eval                                         # ~45 s; writes data/eval/msrvtt_latest.{json,png}
```
