# Independent verification

Verified 2026-09-05 by a separate testing agent on macOS, Python 3.12.12, PyTorch 2.14.0, CPU only. No H100 or CUDA execution was available.

## Automated suite

Command: `env -u VIRTUAL_ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run --no-sync pytest -q`

Result: **18 passed in 7.34 seconds**.

Coverage includes:

- Invalid depth/product, dimensions and dropout are rejected; forward hooks confirm actual block applications equal effective depth.
- Shared embedding and initial blocks match across architectures for a shared seed. One-loop architecture matches the standard model exactly.
- Future-token changes cannot affect earlier logits in either architecture.
- Shared block gradients match the sum of equivalent independently unrolled block gradients, verifying gradient flow through every recurrence without detachment.
- Altering held-out text does not alter tokenizer training. Prepared data validates token counts/checksums and rejects corruption and occupied destinations.
- Seeded batches match and targets are correctly shifted.
- Evaluation counts each target once, handles partial tails, weights by tokens, and restores model training/evaluation mode.
- Both architectures resume to bit-identical final weights and RNG states after an off-schedule pause with nonzero dropout and gradient accumulation. Changed configs and occupied outputs are rejected.
- Paired CLI training, standalone evaluation, greedy generation, report generation, completed-run resume, and mismatched comparison rejection succeed.
- Shell scripts and Slurm script pass `bash -n`; `uv lock --check` succeeds. CPU capability checks work and required CUDA check fails as intended without a GPU.
- Mocked tests verify `auto`, bare `cuda`, and explicit CUDA indices resolve correctly. These are control-flow tests, not GPU execution.

## Live WikiText verification

Successfully downloaded `Salesforce/wikitext`, config `wikitext-2-raw-v1`, pinned revision `f776294184f13b8ff2337b3841cf9269a6216d1e`, and prepared a 512-token BPE vocabulary using only its training split.

Prepared token counts: train **5,313,311**, validation **555,985**, test **628,817**. Tokenizer SHA-256: `a74a835d699f3c3ddb6ffc362f7f0b9885a70e522aad57746371afbbe857b9da`.

A deliberately bounded verification subset used the first 4,096 training tokens and first 1,025 tokens from each held-out split, retaining the train-only tokenizer. The subset manifest explicitly labels this truncation and has recomputed split checksums. Both architectures completed `configs/smoke.json` through `scripts/train.sh`: 4 optimizer steps, 512 training tokens, and 1,024 test targets per architecture.

| CPU verification model | Unique layers | Effective depth | Parameters | Test loss |
|---|---:|---:|---:|---:|
| Standard | 4 | 4 | 68,288 | 6.189627 |
| Looped | 2 | 4 | 42,880 | 6.193700 |

These tiny runs verify operation only; they do not support architecture-quality, speed, convergence, or H100 performance conclusions. Generated data and checkpoints remain under ignored `data/verify-wiki`, `data/verify-wiki-subset`, and `runs/verify-wiki-subset` directories and are not published to Git.

## Historical 42M H100 configuration inspection

Actual CPU instantiation of the original H100 configuration, now preserved as `configs/h100-small.json`, gives **42,155,008** parameters for the 12-layer standard model and **13,783,552** parameters for the 3-layer looped model repeated 4 times. Both have effective depth 12. This is depth/token-budget matching; parameter count is intentionally different.

The Linux lock selects CUDA 13 dependencies for PyTorch 2.14 and a manylinux 2.28 wheel: plan for an R580+ NVIDIA driver, glibc 2.28+, and roughly 10–15 GB for the environment/cache. Actual Linux dependency installation, H100 BF16 kernels, CUDA RNG resume, GPU memory use, throughput, multi-hour training, and Slurm scheduling remain unverified and must be checked on the deployment host. No GPU timing or memory benchmark is claimed.

## 350M default sizing update

Independently verified 2026-09-05 by a separate testing agent on the same CPU-only macOS environment. The automated suite command above now gives **24 passed in 7.83 seconds**.

The full production models were instantiated on PyTorch's `meta` device, which retains real module/tensor shapes without allocating the parameter storage. Parameter enumeration gives **350,451,328** for the standard model and **94,508,032** for the looped model. Forward hooks on a meta-device input confirm **24 block applications** for both architectures, using 24 distinct standard blocks versus 6 shared blocks repeated 4 times. Both output `[1, 8, 8192]` logits for an eight-token input. Width 1,088 and 17 attention heads give 64 dimensions per head. Meta execution verifies architecture and shape wiring; it does not verify numerical behavior, CUDA kernels, or GPU memory use.

New regression checks verify that microbatch 4 × accumulation 16 × context 256 retains **16,384 tokens per optimizer step**, and 2,000 steps retain **32,768,000 training tokens per model**. The historical small preset matches the former default configuration exactly, including microbatch 16 × accumulation 4; all other training settings are unchanged. Mocked CLI dispatch verifies the new `runs/h100-350m` default for training, paired comparison and reporting, and the production config/data defaults. Existing numerical training, resume, data, evaluation and CLI integration tests still pass on the small CPU configurations.

No new full-size training run, H100 execution, memory benchmark, or throughput measurement was performed. The earlier live WikiText smoke results remain the only live-corpus training evidence recorded here.

## WikiText-103 and eight-hour paired budget update

Independently verified 2026-09-05 by a separate testing agent using the project-local uv environment on macOS CPU. Full suite: **30 passed in 15.36 seconds**, using the same command recorded above.

The new end-to-end CLI test actually launches four tiny CPU training workers: two 2-step calibration runs followed by fresh 4-step standard and looped runs. The completed main arms each consume exactly **128 training tokens**, agree with the saved plan, and produce a comparison. Calibration results contain a validation timing pass and no test metric; main checkpoints identify themselves as non-calibration. Completed-run resume preserves the budget state; an expired incomplete run refuses to reset its original deadline. An injected timeout verifies that failure is persisted as incomplete and retry retains the original deadline.

The planner test independently computes its cost formula and checks both sides of the selected integer boundary: the selected common step count fits after reserve while one additional step does not. It also checks explicit maximum-step caps, insufficient budgets, nonfinite timings, and paired training/overhead totals. The default budget CLI, 1,000,000-step search ceiling, 500-step validation interval, retained production model dimensions, WikiText-103 preparation default, training wrapper and eight-hour Slurm allocation are verified.

A real short watchdog test launches a sleeping Python worker and its subprocess, then verifies that the deadline sends SIGTERM to both members of their process group. Nonzero worker exits are also detected. This is a local Unix watchdog check, not an eight-hour endurance test or a Slurm deployment test.

Preparation tests compare actual train-only BPE output against independently reconstructed line encodings, including CRLF, blank lines and a final line without a newline. A mocked encoder produces **1,200,003 tokens per split** to exercise the million-token flush boundary and verify exact token counts and EOS positions. This checks bounded output buffering without downloading the full corpus. WikiText-103 download, full-corpus preparation, H100 throughput/memory and the eight-hour production run remain unverified. Calibration supplies a conservative estimate; actual elapsed time can differ, and an overrun produces an incomplete comparison rather than unequal training budgets presented as a completed experiment.

## H100 startup guard

Independently verified 2026-09-05 on the same CPU-only macOS environment. Focused guard, wrapper, entry-point, shell and dispatch checks: **10 passed, 28 deselected in 3.24 seconds**. Full suite: **38 passed in 16.25 seconds**.

Mocked CUDA states verify that `check --require-h100` rejects no CUDA, missing BF16 support, a selected A100, and a selected H100 exposing only 74.9 GiB. It accepts a selected H100 at the 75 GiB boundary. A two-device case exposes a valid 80 GiB H100 as device 0 while selecting an 80 GiB A100 as device 1; rejection confirms that the guard checks `torch.cuda.current_device()` rather than accepting another visible H100.

An executable wrapper test substitutes `uv`, records invocations, and forces the hardware check to fail. `scripts/train.sh` propagates that failure after only the `check --require-h100` invocation, never invokes `budget`, and does not create the requested output path. The success path records the hardware check before budget dispatch. `scripts/train.sbatch` delegates to this wrapper, so the same guard runs inside the scheduler allocation. All shell and Slurm scripts pass `bash -n`, and `uv lock --check` succeeds.

Both the installed `looped-transformer-comparison check` entry point and `python -m looped_transformer_comparison.cli check` return valid JSON without a `RuntimeWarning`. The installed launcher initially reflected the prior package entry point until `uv sync --locked` regenerated it. Repository scripts invoke the module directly through the project environment, avoiding dependence on stale generated console-launcher contents after a source update.

All GPU capability outcomes above are mocked control-flow tests. No physical H100, CUDA kernel, BF16 computation, MIG allocation, Slurm scheduling, GPU memory measurement, throughput measurement, or production training run was available for this verification.

## Live training plots

Independently verified 2026-09-07 on macOS, Python 3.12, CPU only, with the noninteractive Matplotlib Agg backend. Four focused plotting tests pass, and the full suite passes (**42 passed in 19.21 seconds**).

Synthetic flushed `metrics.jsonl` logs for both architectures render nonempty PNG files at `standard/training.png` and `looped/training.png`. The reader ignores an incomplete final JSONL line, and writes each image through a temporary file before atomically replacing the published image. Missing logs return cleanly without creating run state. The generated four-panel figure contains training/validation loss, validation perplexity, learning-rate schedule, and cumulative token progress.

The `plot --once` CLI emits one valid JSON result, supports `--dpi`, and does not require a display. The `--watch` mode repeatedly rereads flushed metrics at the configured interval; `scripts/plot.sbatch` requests a CPU Slurm allocation and invokes it. Both plot launchers pass `bash -n`, and `uv lock --check` succeeds. No checkpoints or training data are modified by plotting. Live H100 rendering, multi-hour watch execution, and cluster filesystem behavior remain unverified.
