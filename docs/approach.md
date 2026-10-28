# Approach Description

This is the short write-up supplied for the competition paper. For the precise
behavior and scope of the current public code, see [method](method.md) and
[reproduction](reproduction.md). In particular, prototype cosine and mean S-norm
reference scoring are separate outputs; the final secondary stream uses the latter.

Our final submission used an ensemble of two stride-level metric-learning systems
trained only on the provided data. Participants 1-140 were used for
training/cohort generation and participants 141-150 were reserved for
subject-disjoint validation. Each sample was represented as a paired left-right
stride from the 64x64x32 pressure sequence. We normalized pressure volumes to
the common input shape and did not use external datasets or pretrained
backbones.

The backbone was a Siamese 3D CNN shared between left and right footsteps. The
left and right foot embeddings were concatenated and passed through a fusion
layer, then L2-normalized to obtain a 512-D stride embedding. We trained metric
embeddings with Circle Loss (m=0.25, gamma=256), using validation EER for model
selection. At inference, reference templates and probe strides were embedded
with the same network. Claims were scored by cosine similarity against the
claimed participant's reference prototype, computed as the normalized mean of
the available reference embeddings.

To reduce footwear/speed and cohort mismatch effects, we applied symmetric score
normalization (S-norm) using training-set cohort embeddings. We generated S-norm
variants using different cohort top-k values and reference aggregation rules.
The submitted system fused two complementary calibrated score lists: the
strongest Circle-loss S-norm ensemble and a complementary top-400 mean S-norm
model. Because raw score ranges differed across models, we converted each list
to empirical ranks and used weighted rank fusion: 88.5% from the primary
ensemble and 11.5% from the complementary top-400 mean S-norm model. The final
scores.txt contained the fused claim scores and a fixed threshold file.
