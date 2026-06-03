# ten — docs

Design and context docs that don't belong in the main `README.md` or `EVAL.md`. The top-level `README.md` covers what ten *is* and how to run it; `EVAL.md` covers measured quality. This folder is for the longer-form *why* and *what next*.

| doc | covers |
|---|---|
| [production.md](production.md) | What ten looks like if you take it from "personal experiment on one DGX" to a real multi-tenant service. Topology, data plane, observability stack, and the metrics that actually matter in operation. |
| [2016_sota.md](2016_sota.md) | What you would have built instead if asked to ship ten in mid-2016. Frames how much of the current stack is downstream of the 2022–2023 model convergence. |
