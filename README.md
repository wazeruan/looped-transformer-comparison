# Looped Transformer Comparison

Headless, single-GPU language-model training from scratch on **WikiText-103**, targeting one NVIDIA H100 80GB. The default runner budgets **at most eight hours total for calibration and both models**, with measured step planning and a process watchdog. Download/tokenization and environment setup happen separately before the timed run. No GUI, notebook, pretrained weights or external logging account is required.

## Matched architecture

| Setting | Standard decoder | Looped decoder |
|---|---:|---:|
| Unique transformer blocks | 24 | 6 |
| Stack repetitions | 1 | 4 |
| Effective block applications | 24 | 24 |
| Width / attention heads | 1,088 / 17 | 1,088 / 17 |
| Parameters at vocabulary 8,192 | 350,451,328 | 94,508,032 |
| Context / target BPE vocabulary | 256 / 8,192 | 256 / 8,192 |
| Microbatch / accumulation | 4 / 16 | 4 / 16 |
| Training tokens per optimizer step | 16,384 | 16,384 |
| Main optimizer steps | Same measured plan | Same measured plan |

`loop_layers * loops == depth` is enforced. The looped model passes hidden states through the same stack repeatedly, with full gradient flow and no additional loop-specific parameters. Both use pre-norm LayerNorm, causal SDPA attention, GELU MLPs with 4× expansion, learned positions added once, tied token/output embeddings and zero dropout. Each head has 64 dimensions. There is no KV cache.

This matches effective depth, width, tokenizer, batches, optimizer, learning-rate schedule and training tokens. **Unique parameter counts differ.** Optimizer cost, kernels and wall time can differ too; giving each model exactly four hours would not guarantee equal training tokens. Shared parameters accumulate gradients from every use, and repeated depth still requires activation memory.

Both runs use identical seeded batch sequences. Token/position weights and the first six blocks share initial values at the same seed; remaining standard blocks initialize independently. Each model has its own optimizer. GPU execution is not claimed bitwise deterministic. Use multiple seeds for reliable research conclusions.

## H100 quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a compatible NVIDIA driver. Clone/update this repository and enter its root:

```bash
./scripts/setup.sh
./scripts/check.sh --require-h100
# Prepare the larger dataset BEFORE the timed GPU job:
./scripts/prepare.sh
# Calibration + both main runs, eight hours total:
./scripts/train.sh
```

Preparation now defaults to `wikitext-103-raw-v1` under `data/wikitext103`. Existing WikiText-2 data does not become WikiText-103: prepare the new directory once. Fresh larger-data training cannot resume the old tokenizer/checkpoints.

The lock selects PyTorch 2.14.0 with CUDA 13 Linux dependencies and glibc 2.28+ wheels. CUDA 13 normally needs an R580-or-newer NVIDIA driver; see [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html). `setup.sh` uses the project-local Python 3.12 environment and committed `uv.lock`. It does not change the host driver. The H100 preset requires BF16; verify the CUDA build and visible device using `check.sh`. On an older cluster runtime, resolve a compatible PyTorch build before both arms, rather than mixing runtimes between models.

`scripts/train.sh` begins with the same hardware check and exits before calibration or run-directory creation unless the selected CUDA device reports H100, exposes at least 75 GiB, and supports BF16. This catches SSH login nodes, CPU-only PyTorch installations, other GPU types, and restricted H100 partitions. On a scheduled cluster, run it inside an H100 allocation through `sbatch scripts/train.sbatch`.

```bash
# Keep running after an SSH disconnect:
nohup ./scripts/train.sh > training.log 2>&1 &
# Or submit from the repository root after preparation:
sbatch scripts/train.sbatch
# Explicit GPU outside Slurm:
CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh --output runs/wiki103-seed42-8h
# Shorter total budget:
./scripts/train.sh --hours 4 --output runs/wiki103-4h
```

Run one of these alternatives, not simultaneous jobs writing the same directory. The Slurm example requests `08:00:00`, one GPU, eight CPUs and 32GB host RAM; adapt the partition/account to your cluster. Scheduler setup/check time is also inside that allocation, so the scheduler remains the ultimate hard limit. The runner honors `CUDA_VISIBLE_DEVICES`.

## How the eight-hour budget works

1. The runner records an absolute deadline at startup. Default output: `runs/h100-350m-wiki103-8h`.
2. It trains a disposable copy of each architecture for 128 optimizer steps by default, measuring steady-state training time and the surrounding initialization, validation and checkpoint overhead. The longer calibration amortizes H100 startup and kernel warm-up that made the earlier eight-step calibration underfill the allocation. Its final timing pass uses validation again, **not the test set**. Calibration is included in the time budget but excluded from the reported main training tokens. Override with `--calibration-steps` only when you have a reason to trade planning accuracy for calibration time.
3. The planner chooses a shared optimizer-step count using both measured speeds, recurring evaluation costs, a 5% timing margin and a five-minute reserve. It caps steps at `configs/h100-8h.json`'s upper limit (1,000,000), writes `resolved-config.json`, and adjusts warmup to at most 5% of planned steps. Validation/checkpoint cadence is 500 steps. Both main arms restart from scratch at the same seed.
4. The standard model runs, followed by the looped model. Each finishes its identical planned schedule and evaluates its validation-selected checkpoint on the held-out test set. `comparison.json` is emitted only after both complete with matching token budgets.

The aim is to use most of the eight-hour allocation, typically with some headroom. **Actual duration/token throughput is not known until calibration on your H100.** We do not claim an eight-hour benchmark or guarantee completion: contention, thermal changes and slow checkpoint storage can invalidate an estimate. A separate parent process terminates a worker near the deadline and kills it if necessary. This protects the time cap at the cost of possibly stopping before a matched comparison completes. Slurm enforces the external eight-hour allocation. OS-level stalls can delay userspace process cleanup.

Progress is flushed to `training.log` (when redirected), per-model `standard.log` / `looped.log`, and each model's `metrics.jsonl`. `budget.json` stores measured timings, the planned tokens, the original deadline and completion/incomplete status. Use `tail -f runs/h100-350m-wiki103-8h/standard.log` to watch the active arm.

## Interruptions and resume

```bash
# Resume within the ORIGINAL eight-hour deadline, if checkpoints exist:
./scripts/train.sh --resume
```

Resume does not grant another eight hours; downtime counts against the original deadline. It checks settings and data identity. If calibration was interrupted, start with a new output directory to obtain valid timings. If a main model stopped before its first checkpoint, a new output is required. The latest periodic `last.pt` and selected `best.pt` remain available after a timeout; work since the last checkpoint is lost. A forced termination cannot promise a final save. No comparison is reported for incomplete or unequal runs.

After the deadline, you may explicitly continue in **a new allocation outside the original eight-hour budget**:

```bash
uv run python -m looped_transformer_comparison.cli compare \
  --config runs/h100-350m-wiki103-8h/resolved-config.json \
  --data data/wikitext103 --output runs/h100-350m-wiki103-8h --resume
```

This fixed-step command has no time watchdog; choose it only when intentionally extending the experiment. Its completed comparison does not rewrite the earlier budget record. Never change `resolved-config.json` for a resume. Model, optimizer, step, data RNG and PyTorch CPU/CUDA RNG states are restored, with exact same-device/runtime CPU continuation covered by tests. Checkpoint writes use atomic rename; keep both checkpoint files. Logs can repeat steps after a crash between logging and checkpointing.

## Data and evaluation

The [Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext) contains Wikipedia articles; WikiText-103 is the larger variant, with over 100 million source tokens (not our learned BPE-token count). We pin revision `f776294184f13b8ff2337b3841cf9269a6216d1e`. Observe the source dataset's attribution/license terms.

Byte-level BPE with a full byte alphabet trains **only on the training split**. The actual vocabulary size is saved and used by both models, so parameter counts may vary slightly when a tiny custom corpus cannot fill the vocabulary. Nonblank source rows are encoded with `<eos>` boundaries; packed attention can span rows. Train, validation and test streams remain separate. Preparation iterates dataset rows and writes token IDs in bounded chunks instead of retaining the whole tokenized corpus as a Python list; tokenizer training itself still needs memory.

The manifest records source revision, tokenizer hash and split checksums/counts. Training checks these before use. Training windows are sampled with replacement, so the resolved token budget is not an epoch count and can exceed the corpus size.

```bash
# Optional small original dataset:
./scripts/prepare.sh --dataset-config wikitext-2-raw-v1 --output data/wikitext2
# Or caller-provided train.txt, validation.txt and test.txt:
./scripts/prepare.sh --local-dir /path/to/text-splits --output data/custom
```

Validation covers every next-token target once in fixed non-overlapping windows, including the short tail, with context reset every 256 targets. It selects `best.pt`; held-out test evaluation follows main training. Perplexity is this tokenizer's **BPE-token perplexity**, not published word-level WikiText perplexity. Calibration does not select model settings by test performance. Do not tune using repeated inspection of test scores.

## Outputs and other commands

Each main architecture directory contains `metadata.json`, `metrics.jsonl`, `last.pt`, `best.pt` and, on completion, `result.json`. The parent holds `budget.json`, `resolved-config.json`, logs, disposable `calibration/` runs and, only after successful paired completion, `comparison.json` / `comparison.md`.

Results report parameter counts, held-out loss/perplexity, training tokens, training-only throughput and peak CUDA allocation. Training time synchronizes CUDA and includes batch construction/optimizer work, excluding evaluation and saving. Peak allocation includes evaluation, can vary after resume, and is not total process VRAM.

```bash
./scripts/evaluate.sh --checkpoint runs/h100-350m-wiki103-8h/standard/best.pt
./scripts/infer.sh --checkpoint runs/h100-350m-wiki103-8h/looped/best.pt --prompt 'The history of science'
# Fixed-step experiment (not time bounded):
uv run python -m looped_transformer_comparison.cli compare --config configs/h100.json --output runs/fixed350m
# One architecture only:
uv run python -m looped_transformer_comparison.cli train --architecture standard --output runs/single
# Debug synchronous CUDA errors:
./scripts/debug.sh --architecture looped --output runs/debug --stop-after 1
```

Standalone evaluation uses FP32 for portability; main H100 reports use configured BF16, so small numerical differences are expected. Generation checks tokenizer identity and rolls the context window.

**Resources:** standard FP32 parameters, gradients and Adam moments alone take about 5.6GB; activations, logits, attention workspace and temporary copies require more. H100 peak memory/runtime remain unmeasured. Allow roughly 40–50GB disk for CUDA dependencies/caches, the larger prepared dataset, calibration and main checkpoints (estimate). Model training uses microbatch 4 and accumulation 16. If adjusting memory use, change these inversely to retain 16,384 tokens/step in both arms. There is no multi-GPU implementation.

## Local verification and variants

```bash
uv run pytest -q
# Small numerical check via the FIXED-step command:
./scripts/prepare.sh --dataset-config wikitext-2-raw-v1 --output data/smoke --vocab-size 512
uv run python -m looped_transformer_comparison.cli compare --config configs/smoke.json --data data/smoke --output runs/smoke
```

`configs/smoke.json` is for CPU verification. `configs/h100.json` preserves the fixed 2,000-step 350M experiment, and `configs/h100-small.json` preserves the earlier 42M model. The timed default uses `configs/h100-8h.json`; its million-step value is an upper bound, not a promise to run that many steps. Pass `--config` to the timed runner to change architecture, seed or other settings, keeping a fresh output directory. Different loop factors must preserve depth, e.g. 24 = 12×2, 6×4 or 3×8. Use 24×1 for the architecture-equivalence control.

Implementation uses [PyTorch causal SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). Training/evaluation are local and headless after preparation. Git excludes data, checkpoints, environments, secrets and logs.
