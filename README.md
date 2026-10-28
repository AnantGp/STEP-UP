# StepUP 2026: Biometric Footstep Verification

**Verifying identity from paired footstep pressure recordings across changes in footwear and walking speed.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.13905-b31b1b)](https://arxiv.org/abs/2607.13905)
[![Result](https://img.shields.io/badge/StepUP_2026-1st_Place-18794e)](https://www.codabench.org/competitions/13840/)

[Method](docs/method.md) | [Reproduction](docs/reproduction.md) |
[Final submission](docs/final_submission.md) | [Paper](https://arxiv.org/abs/2607.13905)

Code for the ArogyaPandit Research Team submission to the 2nd International
StepUP Competition for Biometric Footstep Recognition at IJCB 2026.

The organizers' final validated standings place the team **1st with 8.00% EER**,
as reported in the [competition summary paper](https://arxiv.org/abs/2607.13905).
Our final submission is `SUBMIT_TOMORROW_ms400_weight_115.zip`.

## Research Question

Can a system verify an unfamiliar person's identity from a walking stride when
enrollment and query samples have different shoes or walking speeds?

The input is a left/right pair of time-varying pressure recordings and a claimed
identity. The output is a similarity score against that identity's reference
recordings. A higher score indicates stronger support for the claim. The task
requires generalization to people absent from training, with limited enrollment
data and distractor identities in the probe set.

Footwear changes the contact pattern recorded by the mat, so a model can learn
shoe appearance instead of identity. Our approach combines a learned stride
representation with cohort-based normalization and score fusion to address
this mismatch.

## Pipeline

```text
Left pressure  --> shared CNN --+
                                +--> concatenate --> FC fusion --> 512-D stride embedding
Right pressure --> shared CNN --+                                     |
                                                                L2 normalize
                                                                     |
                                                      probe/reference cosine scores
                                                                     |
                                                      training-cohort top-k S-norm
                                                                     |
                                                      reference-score aggregation
                                                                     |
                                                      weighted empirical-rank fusion
                                                                     |
                                                      claim verification score
```

## Team

- Anant Gupta
- Gourav Gupta

Affiliation: Arogyapandit Private Limited, India

## Method Summary

The system learns stride-level biometric embeddings from paired left and right
pressure footsteps. Each foot is processed by a shared 3D CNN backbone. The two
foot embeddings are concatenated, projected to a 512-dimensional stride
embedding, and L2-normalized for verification.

Training used the provided competition training data only. Participants 1-140
were used for training/cohort generation and participants 141-150 were reserved
for subject-disjoint validation. The documented approach uses Circle Loss
(`m=0.25`, `gamma=256`) and prototype scoring against each claimed user's
reference templates.

At inference, cosine claim scores were normalized with symmetric score
normalization (S-norm) using cohort embeddings from the training participants.
The final submitted file, `SUBMIT_TOMORROW_ms400_weight_115.zip`, used weighted
empirical-rank fusion of two S-norm score lists:

```text
final_score =
  0.885 * rank(primary_circle_snorm_ensemble)
+ 0.115 * rank(secondary_top400_mean_snorm_model)
```

No external biometric datasets or pretrained backbones were used.

The generator exports prototype-cosine and nearest-reference baselines
separately. Its S-norm mean candidates average normalized similarities over
individual references; that is different from cosine to a normalized prototype.
Right-foot volumes are mirrored along the width axis, resized to
`64 x 64 x 32` in `(time, height, width)` order, and min-max normalized per foot.

## Results and Interpretation

The summary paper reports **8.00% EER** and **42.88% FNMR at FMR = 1%**.
At the submitted threshold, FNMR is 0.16%, FMR is 32.80%, and balanced accuracy
is 83.52%. These are different operating points; EER is not classification
accuracy. This is a competition result, not evidence of deployment readiness.

Aggregate leaderboard feedback was available during development. The result
should therefore be read as competition performance rather than an untouched
test-set estimate. Empirical-rank fusion depends on the supplied batch of claims
and is not a calibrated per-probe probability.

## Release Scope

This release contains a baseline runner, scoring utilities, and the final fusion
rule. Original winning checkpoints, complete training configurations, the
construction of the primary ensemble, and original component scores are not
included. The commands below do **not** reproduce the 8.00% result by themselves.

Historical checks recorded exact agreement with all 10,000 final submission
scores given the two original inputs. Those inputs are absent from this
checkout, so that comparison cannot currently be rerun. See
[reproduction details](docs/reproduction.md) for the supported workflow and limits.

## Repository Layout

```text
code/stepup_runner.py          Train/infer/check CLI for stride embeddings
code/make_v2_submissions.py    Builds prototype, nearest, and S-norm submission candidates
code/fuse_rank_scores.py       Recreates SUBMIT_TOMORROW_ms400_weight_115.zip score fusion
code/snorm.py                  S-norm helper
docs/approach.md               Short paper-style approach description
docs/method.md                 Model, scoring equations, and limitations
docs/reproduction.md           Commands and checkpoint compatibility
CITATION.cff                   Citation metadata for the summary paper
```

Competition data, checkpoints, embeddings, and submitted score files are not
included.

## Expected Dataset Layout

```text
data/competition/
  1 - Training/
  2 - Reference/
    reference_data.npz
    reference_metadata.csv
  3 - Probe/
    probe_data.npz
    probe_metadata.csv
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default
package is not appropriate for your GPU.

## Basic Usage

Check the dataset:

```bash
python code/stepup_runner.py check --dataset-root data/competition
```

Train a Circle Loss baseline:

```bash
python code/stepup_runner.py train \
  --dataset-root data/competition \
  --output-dir runs/circle_baseline \
  --loss circle \
  --epochs 10 \
  --batch-size 128 \
  --num-workers 0
```

Generate S-norm submission candidates:

```bash
python code/make_v2_submissions.py \
  --dataset-root data/competition \
  --checkpoint runs/circle_baseline/best.pt \
  --output-dir runs/circle_baseline/submissions \
  --top-k 100 200 400 \
  --num-workers 0
```

Fuse two S-norm score lists with the final `SUBMIT_TOMORROW_ms400_weight_115.zip`
weighting:

```bash
python code/fuse_rank_scores.py \
  --primary path/to/primary_submission.zip \
  --secondary path/to/secondary_submission.zip \
  --secondary-weight 0.115 \
  --output runs/final_rank_fusion.zip
```

The two input lists must refer to the same claims in the same order. The
original lists must be supplied separately. Start with zero data-loader workers;
increase after checking throughput and memory on your machine.

## Paper and Citation

The team contribution appears in *The 2nd International StepUP Competition for
Biometric Footstep Recognition: From Steps to Strides*, by Robyn Larracy,
Anant Gupta, Gourav Gupta, and colleagues. The
[arXiv record](https://arxiv.org/abs/2607.13905) states acceptance to IJCB 2026.
Use [CITATION.cff](CITATION.cff) to cite the paper.

## Acknowledgments and Provenance

The runner adapts the public notebook workflow from the
[UNB StepUP organizer starter kit](https://github.com/UNB-StepUP/2nd_stepUP_competition).
We acknowledge the dataset creators, competition organizers, and the
[PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
implementation of Circle Loss. Git history is preserved from the
[earlier team release](https://github.com/AnantG5/STEP-UP).

No new license grant is added by this documentation update. A project license
is not currently included; consult the authors and upstream terms before
redistribution. Competition data is obtained separately under its applicable terms.
