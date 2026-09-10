# Recorded H100 experiments

These lightweight records preserve the two completed WikiText-103 Muon runs. Checkpoints, tokenized data, Slurm logs, and full metric streams remain excluded from Git because they are large and reproducible from the recorded configuration.

| Experiment | Standard test loss / PPL | Looped test loss / PPL | Interpretation |
|---|---:|---:|---|
| Equal tokens, different parameter counts | 2.8011 / 16.4635 | **2.7832 / 16.1703** | The 94.5M looped model outperformed the 350.5M standard model at the same 604.0M training tokens. |
| Parameter- and measured-compute-matched | **2.7978 / 16.4079** | 2.8237 / 16.8394 | The 350.5M looped model received fewer steps/tokens within the measured budget and underperformed the 350.5M standard model. |

Both results use WikiText-103, an 8,192-token byte-BPE vocabulary, BF16, the mixed Muon/AdamW recipe, and one H100 80GB GPU. They are single-seed measurements, not statistical evidence.

The first run tests parameter efficiency under an equal-token budget. The second run controls parameter count to within 0.02% and allocates steps from calibration timing; it tests end-to-end compute efficiency. Read the JSON records for the exact settings and metrics.
