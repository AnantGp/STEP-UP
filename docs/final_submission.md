# Final Submission Recipe

This repository documents the final arithmetic for
`SUBMIT_TOMORROW_ms400_weight_115.zip`, reported as 1st place at 8.00% EER in the
[competition summary paper](https://arxiv.org/abs/2607.13905).

The final file was not a raw single-model output. It was a calibrated fusion of
two already-normalized S-norm score lists:

1. A primary Circle-loss S-norm ensemble score list.
2. A complementary top-400 mean S-norm score list from the secondary stride
   model.

Both score lists were converted to empirical ranks to remove calibration and
scale differences. The submitted score for each claim was:

```text
final_score = 0.885 * rank(primary) + 0.115 * rank(secondary)
```

The `code/fuse_rank_scores.py` utility reproduces this final step with:

```bash
python code/fuse_rank_scores.py \
  --primary path/to/primary_submission.zip \
  --secondary path/to/secondary_submission.zip \
  --secondary-weight 0.115 \
  --output runs/SUBMIT_TOMORROW_ms400_weight_115.zip
```

The public repository intentionally does not include private CodaBench score
files, submitted zips, dataset files, embeddings, or checkpoints.

## Provenance and Reproduction Boundary

The recorded primary input is `ensemble_2_models_heavy_w2.zip`. The secondary
input is `ms_balanced_submissions/submission_v2_snorm_top400_mean_raw.zip`.
These are historical artifact names, not bundled files. `MS400` is an experiment
label; it should not be treated as evidence of a multi-scale architecture.

Historical project checks recorded agreement with all 10,000 submitted scores
(maximum absolute difference `0.0`) and identical threshold contents (`0.5`).
This establishes final fusion arithmetic given the original inputs. It does
not establish reproduction of those inputs from raw data.

Winning checkpoints, full run configurations, and primary-ensemble construction
are absent. The public baseline and scoring code are reusable development
components, not an exact reconstruction of every historical model. The primary
input is already an ensemble, so two score streams do not mean exactly two
individual networks. See [method](method.md) and [reproduction](reproduction.md).
