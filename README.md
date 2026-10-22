# KVguard

Streaming harmfulness detection from a frozen language model's KV cache. A small head reads a running mean of keys, values, or hidden states while a response is replayed token by token, and raises an alarm at the first logit that crosses a calibration threshold.

The published comparison is on [FineHarm](https://huggingface.co/datasets/liyang-ict/FineHarm) with `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. It is a controlled comparison of native cache features and hidden states. It is not a reproduction of SIREN, ShieldHead, or Kelp, and the FineHarm numbers are teacher-forced replays, not free generations of this backbone.

A separate repository, KVguard_agent, trains pooled K, V, and KV probes on agent trajectories with the TACIT joint-training protocol. The two codebases do not share checkpoints or labels.

## Held-out FineHarm test

Layer index 11 was chosen on the pilot development set and then locked. Thresholds were fit only on safe calibration responses at a 5% response-level false-positive target. The official test split was scored once after `results/full/protocol_lock.json` and was not used to pick a layer, a threshold, or a filter. Numbers below are the mean over seeds 17, 42, and 2026. The full tables, intervals, and per-seed rows are in [results/full/REPORT.md](results/full/REPORT.md).

| Method | Dim | Parameters | Safe FPR | Premature alarm | Recall by end+32 | Recall by end+128 | Stream AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `h_mean` | 2560 | 327,937 | 0.044 | 0.278 | 0.650 | 0.692 | 0.996 |
| `v_mean` | 1024 | 131,329 | 0.046 | 0.253 | 0.645 | 0.702 | 0.995 |
| `kv_mean` | 2048 | 262,401 | 0.042 | 0.233 | 0.643 | 0.720 | 0.995 |

Read the false-positive rate and the premature-alarm rate before recall. `recall_by_end_plus_32` ignores alarms that fire before the proxy sentence boundary. The span `[a, b]` is the tokens overlapping the first harmful sentence. It is not a human annotation of the first harmful token, and delay after `b` is not a leaked-token count.

On the same locked test split, `Qwen/Qwen3Guard-Stream-4B` (official Unsafe label, not the 5% FineHarm threshold) has safe FPR 0.029 and recall by end+32 of 0.357, with a premature-alarm rate of 0.600. That row is an external baseline. It is not one of the trained heads. See `results/full/metrics/qwen3guard_stream_summary.csv`.

Peak memory of the detector itself is not distinguishable from run-to-run noise (about 1 MiB, `extra_peak_reliable=false`). Persistent state is 4–10 KiB per sample. Decode overhead on a 2048-token response is a few percent. The backbone KV cache is shared by every method.

The released layer-11 checkpoints, scalers, and thresholds are under `results/full/checkpoints/`.

## What this repository contains

```
configs/          smoke, pilot, full, and follow-up YAML
src/kv_safety/    prepare, extract, train, calibrate, evaluate, benchmark, report
scripts/          portable runners; set PYTHON if python is not the intended interpreter
tests/            CPU unit tests, a tiny-Qwen causality check, and an optional GPU alignment test
results/          published reports, metrics, figures, and the locked layer-11 checkpoints
```

Not in the repository, and not safe to commit:

- Hugging Face weights and datasets (`hf_cache/`, or `HF_HOME`)
- Prepared rows, feature tensors, and score dumps (`artifacts/`)
- Local alignment notes that quote response text (`local_checks/`)
- Virtualenvs, conda prefixes, and Weights & Biases logs

`results/` is the snapshot that belongs in git. A fresh run writes to `artifacts/` and does not overwrite `results/`.

## Install

Python 3.11 or 3.12. A CUDA GPU is required for extraction, training, and the memory benchmark. CPU tests do not need one.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -e . --no-deps
```

`requirements.lock.txt` is the lockfile from the development environment (torch 2.14.1+cu130, transformers 5.19.0). The GPU run that produced `results/full` used an RTX 4080 SUPER 32GB. Install a torch build that matches your CUDA if the lockfile wheel does not.

Weights and datasets stay in the Hugging Face cache:

```bash
export HF_HOME="$PWD/hf_cache"
python -m kv_safety.download --config configs/full.yaml
```

FineHarm is a gated dataset. Accept its terms on Hugging Face and log in before the download. `Qwen/Qwen3-4B` is Apache-2.0. `Qwen/Qwen3GuardTest` is CC-BY-NC-4.0. This repository does not redistribute those files.

Recorded revisions, file hashes, and licenses are in `results/resource_manifest.json`. A config with `revision: null` uses the SHA in that manifest. `offline: true` loads only the local snapshot.

## Reproduce

```bash
python -m kv_safety.environment --config configs/pilot.yaml
python -m kv_safety.data.prepare --config configs/pilot.yaml
pytest -q -m "not gpu"
pytest -q -m gpu          # CUDA and the pinned Qwen3-4B snapshot

# Published full run: cleaned train, dev, calibration, then the locked test.
bash scripts/run_layer11.sh
python -m kv_safety.report --config configs/full.yaml
```

`scripts/env.sh` resolves the repo root and `HF_HOME`. Override the interpreter with `PYTHON=/path/to/python`.

| Config | Role |
|---|---|
| `configs/smoke.yaml` | Overlapping-free prompt groups drawn from the official train split |
| `configs/pilot.yaml` | Development run used to choose block 11. Test scoring stays off |
| `configs/full.yaml` | Published train / dev / calibration / test comparison at block 11 |
| `configs/workshop.yaml` | Allowed only with a protocol lock |

Test scoring refuses to run unless `--protocol-lock` points at a lock file written after thresholds exist. Dev metrics were used for early stopping, so they are development feedback.

Weights & Biases is off in the checked-in configs. Set `wandb.enabled: true` and `wandb.entity` to your own account if you want it. A missing login does not stop the run.

## Method

**Input.** The prompt is `apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)`. The response is tokenized on its own and concatenated, so the boundary cannot fall inside a token. The final `<|im_end|>` is not an input and is not supervised. Rows longer than 2048 tokens (template plus response) are dropped.

**Span.** A fast tokenizer offset map aligns the first harmful sentence to tokens. Overlapping tokens form `[a, b]`, 1-based and inclusive. A token that crosses the sentence boundary is counted. Special tokens, empty offsets, and non-monotonic maps go to `alignment_errors.jsonl` and are not guessed.

**Labels.** Position `t < a` is 0, `a ≤ t < b` is ignored (`-100`), and `t ≥ b` is 1. A safe response is 0 at every position. Train and dev keep at most 32 supervised positions per response (`a-1`, `b`, `b+32`, then a uniform fill), with weight `1/n`. The loss is weighted BCE. The scaler is fit on train only, with equal weight per response. Early stopping uses the same 32 positions on dev. Streaming scores are a separate token-by-token replay. They do not reuse those 32 positions.

**Features.** `h` is the block output before the final norm. `k` is the post-k-norm, post-RoPE key written into the cache. `v` is the native value in the cache. Eight KV heads are flattened head-major. They are not averaged across heads and they are not the repeated query heads. The running mean covers response tokens only and is accumulated in FP32.

**Online state.** The score after the forward that consumes response token `t` belongs to prefix `t`. The last forward is the flush. KV features read the new cache positions as a view. Hidden-state features update a running sum inside the hook and do not keep the history. Persistent state does not grow with length. Monitors do not change logits, and state is reset between requests.

**Threshold.** On calibration, take `max_t logit` over each safe response and choose the lowest threshold whose response-level FPR is at most 5%. An alarm is a strict `>`. Ties are not alarms. Dev reports the measured FPR and does not refit the threshold.

**Benchmark.** Each method, batch, length, and repeat runs in its own process, with the same token ids, cache, SDPA, BF16, LM head, and flush schedule. There is no per-token `.item()`. OOM is recorded as OOM. The benchmark refuses to run without CUDA.

## Follow-ups

These runs do not edit `results/full/protocol_lock.json` or the response-head thresholds.

| Run | Command | What it changes | Where the numbers are |
|---|---|---|---|
| Prompt head | `python -m kv_safety.prompt_head --config configs/full.yaml` | Layer-11 V of the prompt only. Label is the FineHarm filename (`_HP_` / `_BP_`), not response harm | `results/full/prompt_head/` |
| Fused label | `python -m kv_safety.v_fuse --config configs/full.yaml` | Same response `v_mean`, but a harmful prompt labels every response token as positive | `artifacts/full/v_fuse/` after a local run |
| Early label | `python -m kv_safety.v_early --config configs/full.yaml` | Early positive labels only on harmful answers | `artifacts/full/v_early/` after a local run |
| Qwen3GuardTest | `python scripts/eval_qwen3guardtest.py` | Official stream protocol on `Qwen/Qwen3GuardTest` | `results/full/qwen3guardtest/summary.json` |
| Guard on FineHarm | `bash scripts/run_qwen3guard_after_phase_d.sh` | `Qwen3Guard-Stream-4B` on the locked FineHarm test split | `results/full/metrics/qwen3guard_stream_summary.csv` |

Prompt classification on the FineHarm test split, mean over the same three seeds: `prompt_v_last` macro-style F1 0.952 (AUROC 0.990, FPR 0.036) and `prompt_v_mean` F1 0.977 (AUROC 0.996, FPR 0.039). That task is whether the prompt is harmful. It is not the streaming response task in the table above.

If a follow-up is short on disk it exits. Set `KVGUARD_ALLOW_SCRATCH_DELETE=1` only if you accept deletion of recomputable feature shards, `wandb/`, and `.pip-cache` inside this repo. The script will not delete the model cache or the locked checkpoints.

## Tests

```bash
pytest -q -m "not gpu"
```

- `test_running_mean.py`, `test_labels.py`, `test_splits.py`, `test_alignment.py`, `test_metrics.py`, `test_classifier.py` are CPU unit tests.
- `test_causality.py` uses a randomly initialized tiny Qwen3, on the same transformers code path, to check post-RoPE keys, chunked versus tokenwise extraction, prefix invariance, online alignment, unchanged logits, constant state size, reset between requests, and benchmark forward counts. Those checks do not certify real-model numerics.
- `test_pipeline_synthetic.py` runs prepare, extract, train, evaluate, and report on synthetic data.
- `test_model_alignment.py` (`-m gpu`) repeats the checks on the real Qwen3-4B snapshot. The gate is mean relative error below 0.02 and cosine above 0.98. The max-element ratio is inflated by BF16 residual accumulation and is not the pass condition.

## Limits

- Sentence-boundary labels are a proxy for cumulative prefix harm. A sample of 100 rows is still marked pending human review in the data audit. Model scores are not that review.
- Dev selected the checkpoints. Only the locked test split is a held-out number.
- KV is 2048-d and V is 1024-d, so equal width is not equal parameter count. A gap is not, by itself, evidence of complementary information.
- The label audit and the alignment spot-check that quotes response text stay on the machine that produced them.

## License

Code is MIT. See [LICENSE](LICENSE). Model and dataset licenses stay with their publishers.
