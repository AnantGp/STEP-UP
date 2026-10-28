# Method: Pressure Strides to Verification Scores

## Task and Representation

Given a left/right footstep pair and a claimed identity, estimate whether the
stride belongs to that person. Enrollment is limited and query conditions can
differ from enrollment, especially in footwear.

The runner pairs consecutive opposite-side footsteps within each participant,
pass, footwear, and speed group. Right-foot volumes are mirrored along the
width axis. Each foot is resized to `(64, 64, 32)` in `(T, H, W)` order and
min-max normalized to `[0, 1]`. Constant-valued volumes map to zero.

`SpatioTemporalConv` factors a 3D kernel into spatial and temporal operations.
`CNN3D` uses four such blocks, instance normalization, ReLU, and adaptive max
pooling. Each foot produces 256 features in the released code. `FusionNet`
concatenates both vectors and projects them to a 512-D stride embedding.
Inference L2-normalizes the embedding.

The baseline optimizes Circle Loss (`m=0.25`, `gamma=256`) with Adam (`1e-4`)
and exponential decay (`0.95` per epoch). The best validation EER selects the
checkpoint. These defaults are not a complete record of every historical
ensemble member.

## Direct Scoring

For normalized probe `p` and references `r_j` for the claimed identity:

```text
prototype = normalize(mean_j(r_j))
prototype_score = (dot(p, prototype) + 1) / 2
```

Nearest-reference scoring takes the largest cosine similarity to that user's
references, also mapped to `[0, 1]`. The generator exports both separately.

## Top-k Symmetric Score Normalization

For each query embedding `x`, compute cosine similarities to the training-stride
cohort. Retain its highest `k` scores and compute their mean `mu_x` and standard
deviation `sigma_x`. For raw cosine `s(p, r)`:

```text
S_norm(p, r) = 0.5 * (
    (s(p, r) - mu_p) / (sigma_p + epsilon)
  + (s(p, r) - mu_r) / (sigma_r + epsilon)
)
```

The generator uses `epsilon=1e-6`. Probe and reference have their own top-k
cohort statistics. Cohort embeddings come from training participants 1-140.
The generator then averages or maximizes normalized scores across the claimed
user's references.

`top400_mean_raw` means **400 cohort stride embeddings**, **mean reference-score
aggregation**, and **no min-max rescaling** of that candidate. It is not cosine
to a normalized mean reference embedding; both scoring options exist.

## Final Fusion

The historical final score is:

```text
final = 0.885 * rank01(primary_ensemble_scores)
      + 0.115 * rank01(secondary_top400_mean_snorm_scores)
```

`rank01` assigns ascending ordinal ranks divided by the number of claims.
The utility uses a stable sort, so ties receive different ranks in their input
order. This behavior is preserved from the released historical utility.

The primary stream is itself an ensemble. Two final score streams do not mean
exactly two individual networks. The available release does not reconstruct
all component models or the construction of the primary input.

## Interpretation

S-norm adjusts scores relative to cohort similarity distributions; it does not
prove that embeddings become footwear-invariant. Aggregate competition results
do not isolate the contribution of each component.

Ranks are computed over the whole claim list. The final score is therefore
batch-dependent, not a probability or a calibrated streaming-system score.
A streaming application would need a fixed calibration reference and new evaluation.

Validation compares same/different identities across held-out strides and does
not exactly replicate the competition enrollment/probe protocol. Aggregate
leaderboard feedback was available during development. Both facts limit the
interpretation of generalization estimates.
