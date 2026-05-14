# Final Submission Recipe

This repository documents the code path used for
`SUBMIT_TOMORROW_ms400_weight_115.zip`.

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
