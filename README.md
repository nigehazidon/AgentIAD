# AgentIAD

Research workspace for anomaly-detection agent evaluation on industrial benchmarks (MVTec AD, MVTec LOCO, HeadCT, LAG, KAPUTT).

## Layout

- `AnomalyAgent/` — reference implementation of the AnomalyAgent LLM agent. See its own [`AnomalyAgent/README.md`](AnomalyAgent/README.md) for setup and run instructions. Core source lives under [`AnomalyAgent/src/react_agent/`](AnomalyAgent/src/react_agent/); per-dataset evaluation entrypoints are `AnomalyAgent/evaluate_*.py`.
- `test_patchcore_mvtec.py`, `eval_patchcore_full_mvtec.py`, `run_mvtec_2gpu.sh` — PatchCore baseline experiments on MVTec.
- `results/`, `data/`, `paper/`, and `anomalib/` are runtime artifacts / a third-party checkout and are intentionally **not** tracked by this repository.

## Getting started

Follow the setup and run steps in [`AnomalyAgent/README.md`](AnomalyAgent/README.md).
