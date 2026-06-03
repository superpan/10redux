# From "personal experiment" to production

ten today is a single-DGX, single-process system: Qdrant + vLLM as Docker containers, one FastAPI server, one static Next.js bundle, all behind one port on a tailnet. This doc sketches what changes to take it to a real multi-tenant service, and — more importantly — *what to instrument so you can run it*.

The architecture is mostly mechanical decomposition. The interesting work is observability and cost discipline.

## Current state, briefly

- One DGX Spark (GB10), one machine, one GPU.
- vLLM serves Qwen3-VL-8B on `:8000`, capped at 35 % of unified memory.
- Qdrant on `:6333` with `ten_visual` / `ten_text` / `ten_audio` collections.
- `ten serve` (FastAPI) on `:8765` serves the REST API, the MCP endpoint at `/mcp/`, and the static UI at `/`. Lazy-loads V-JEPA, Qwen3-Embedding, Whisper, Silero VAD, CLAP, and the cross-encoder reranker on first use.
- Ingest is a separate one-shot process (`ten index`) that shares the same model wrappers.
- Source videos live in the filesystem. Thumbnails live under XDG. No tenant concept, no auth, Tailscale is the trust boundary.

## Production topology

### Services

| service | role | scale knob | why split |
|---|---|---|---|
| `ten-api` | FastAPI, CPU-only, stateless. Handles search, clip lookups, signed-URL minting, MCP. | HPA on QPS; 3–N replicas behind an L7 load balancer | user-facing surface; bursty and latency-sensitive |
| `ten-ingest-worker` | Python, GPU. Pulls jobs from a queue; runs V-JEPA + Whisper + CLAP; calls out to vLLM for captions. | HPA on queue depth; spot-tolerant; GPU node pool | bulk throughput, latency-tolerant |
| `ten-summary-worker` | Python, GPU. On-demand summarization (the "Summarize" button + `summarize_clip` MCP tool). | HPA on QPS; smaller, dedicated quota | latency-bound (≤ 5 s OK, ≥ 20 s feels broken); must not get queued behind an ingest |
| `vllm-caption` | vLLM (or NVIDIA NIM) cluster, batched. | tokens / s, GPU memory | shared by ingest + summary workers; continuous batching shines here |
| `vllm-summary` | Optionally a separate vLLM cluster for short summaries. | per-tenant fairness | keeps batch sizes sane on the user-visible latency path |

In current ten, the single `ten serve` process plays the API + summary roles, and a separate `ten index` invocation plays the ingest role. In production they all become independently scalable services so one heavy summarize call from one user can't tail-latency search for everyone else.

### Data plane

| concern | today | production |
|---|---|---|
| Metadata source of truth | Qdrant payloads | **Postgres** — videos, users, libraries, ACLs, ingest task state, billing-relevant counters |
| Vector index | Qdrant on local Docker | **Qdrant cluster** (multi-node, replicated) *or* managed (Qdrant Cloud / Pinecone / Vespa for hybrid retrieval) |
| Video bytes | Local `~/videos/...` | **S3** with lifecycle policies → Glacier after 90 days unaccessed |
| Thumbnails | Local XDG dir | **S3 + CDN** (CloudFront / Cloudflare) — heavily cached |
| Clip streams | FastAPI HTTP-Range from local disk | **CDN-fronted S3** with signed URLs; `ten-api` hands out short-TTL signed URLs |
| Ingest pipeline | Linear, in-process | **Kafka / SQS** with explicit stages: `upload → probe → chunk → caption → embed → upsert`. Each stage idempotent, retryable, observable. |
| Hot-path cache | None | **Redis** for query-embedding cache, recent results, hot payloads |

### Surface

| concern | today | production |
|---|---|---|
| Auth | Tailscale ACL (implicit) | OAuth (Google / Apple SSO for consumer; SAML for enterprise) → JWT bearer; MCP authenticated via same flow |
| Multi-tenancy | None (single `library` tag) | `tenant_id` in every Qdrant payload + Postgres row; all queries auto-scoped; library ACLs (owner / shared-with) |
| Quotas | None | per-tenant: storage GB, ingest-hours / month, search QPS, summarize calls / day |
| Secrets | env vars | Vault / AWS Secrets Manager — HF tokens, DB creds, signing keys, MCP server tokens |

### Deploy + operate

- **K8s** with separate GPU and CPU node pools.
- **GitHub Actions** for tests + image builds; **ArgoCD** for deploys.
- **Canary** for `ten-api` (5 % → 25 % → 100 % with auto-rollback on SLO breach).
- **Rolling** for ingest workers; version-tagged Kafka topics so old + new versions can co-exist during cutover.
- **OpenTelemetry** in code → **Prometheus + Tempo + Loki + Grafana**; **Alertmanager → PagerDuty**.

## Metrics — what to instrument

Four buckets, listed in priority order within each bucket.

### Quality — the thing users actually care about (and the hardest)

Eval-set numbers (see `EVAL.md`) are great for design decisions. *Production* quality is what users do, not what test sets say.

| metric | why | how |
|---|---|---|
| **CTR at position 1, 3, 10** per tenant | how often the top-K result is what the user wanted | log query + clicks; aggregate per tenant per day |
| **Median rank of clicked result** | retrieval ordering quality — if this is rising, the ranker is degrading | from the same click logs |
| **Re-query rate within 30 s** | dissatisfaction signal — high means top results weren't useful | sequence-of-queries from session |
| **Time-to-first-click** | how long users browse before finding something | session timing |
| **Offline R@K vs 7-day baseline** on a golden set | guardrail against pipeline regressions before users notice | nightly cron of QVHighlights / MSR-VTT eval; write to a results table; Grafana chart |
| **Quality per feature-flag bucket** | when A/B-ing a new captioner or reranker, per-bucket CTR is required | shadow traffic + offline replay |

### Latency — the thing users feel

| metric | illustrative target | notes |
|---|---|---|
| **p99 search end-to-end** | < 800 ms | budget: ~50 ms query embed, ~30 ms Qdrant, ~200 ms rerank, ~100 ms hydrate, ~50 ms CDN signing, rest = network |
| **p99 summarize-on-demand** | < 8 s | Qwen3-VL re-decodes clip + generates ~256 tokens; vLLM batching helps |
| **p95 end-to-end ingest** (upload → searchable) | < 5 min for 10-min source video | queue depth × per-clip cost determines this |
| **per-stage ingest latency** (decode / caption / embed / upsert) | each tracked separately | tells you which stage to optimize first |
| **TTFB on clip stream** | < 300 ms | CDN cache hit rate is the lever |

### Reliability — the thing that wakes oncall

| metric | target |
|---|---|
| **API availability** (5xx rate) | 99.9 % monthly |
| **Ingest pipeline lag** (oldest unprocessed task age) | < 10 min |
| **vLLM error rate** (the `RemoteProtocolError` is the canonical one) | < 0.5 % with auto-retry |
| **Qdrant query error rate** | < 0.1 % |
| **GPU OOM rate** | 0; alert on any |
| **Successful backups in last 24 h** | binary alert |
| **MTTR per incident class** | tracked + reviewed weekly |

### Cost — the thing finance asks about

| metric | why |
|---|---|
| **GPU hours / day** by service | shows where compute concentrates; informs reserved vs spot strategy |
| **Cost per minute-of-source-video ingested** | unit economics of ingest; tied to VAD skip rate (more skip = less Whisper) |
| **Cost per search query** | unit economics of serving; tied to cache hit rate, Qdrant cost, reranker on/off |
| **Storage growth rate** (S3 + Qdrant) | drives the cold-tier-after-90-days decision |
| **CDN egress GB / day** | clip streams dominate; per-user cap may be needed |
| **GPU utilization (compute + memory)** | sub-50 % means consolidate; sub-30 % sustained means right-size or batch harder |

### Operational signals specific to ten's design

These wouldn't show up in a generic checklist but come out of stuff that bit us during the build.

| signal | why |
|---|---|
| **VAD skip rate per library** | direct compute lever — outdoor / music libraries skip 40 %+ of Whisper; surveillance libraries skip 80 %+. Affects ingest cost per minute-of-source-video. |
| **`¶¶¶` / repetition-loop rate in transcripts** | leading indicator that VAD threshold drifted or a new corpus type slipped past the gate (see `tools/asr_contamination.py`) |
| **CLAP reranker firing rate** (how often `sources=['text','audio']` appears in returned hits) | tells you whether audio retrieval is contributing or sitting dormant |
| **Rotation-tag mismatch** (frame dimensions vs caption mentioning "vertical" / "horizontal") | catches new ingests with broken display-matrix handling before users notice sideways thumbs |
| **Per-library R@K** (offline eval scoped to one library) | per-tenant quality tracking; alerts on the *one* library that's degrading |
| **Captioner backend mix** (vLLM vs transformers fallback ratio) | if a tenant is hitting transformers a lot, vLLM is over-utilized or down |

## What's actually hard

A few things that look easy on a slide but bite in practice.

1. **GPU economics dominate.** vLLM running at 35 % memory utilization (what we set on the DGX) wastes most of the GPU. In production you pack more model replicas per node, share KV cache across tenants, and the captioner cluster becomes the largest line item. Continuous batching only helps if QPS is steady — bursty ingest patterns require queue shaping.
2. **Re-ingest is the unfixable cost.** Every captioner or VAD upgrade means deciding whether to re-process the historical corpus. We hit this with the rotation fix (had to re-ingest 593 clips). At 10 M clips that's a 10-day GPU job. Production has to budget for this from day one — versioned embeddings, dual-collection rollouts, or accepting a "old clips stay at v1 quality" policy.
3. **Quality metrics are deceptive.** CTR rewards click-bait thumbnails. Offline R@K rewards overfitting to the eval set. Time-to-first-click rewards short videos. The honest move is to instrument all of them, watch ratios, and use offline eval only as a guardrail — not the optimization target.
4. **The MCP surface becomes an API contract.** `tools/search` returns whatever payload feels right today. In production that's a stability problem — agents wired against today's schema break on any rename. Versioned tool surface (`search.v1`, `search.v2`) gets you out, but it's tedious work to introduce *after* clients exist.

## Summary

The architecture isn't exotic. The work is mostly mechanical decomposition + observability + cost discipline. The hard parts are:

- Getting *quality* metrics that aren't lying to you.
- Making the captioner re-process story sane from day one.
- Designing the MCP / API contracts so they're versionable before agents start consuming them.
