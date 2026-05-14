import numpy as np
from scipy.spatial.distance import cdist

def adaptive_s_norm(probe_emb, ref_emb, cohort_embs, top_k=200):
    raw_scores = 1 - cdist(probe_emb, ref_emb, metric='cosine')
    probe_cohort = 1 - cdist(probe_emb, cohort_embs, metric='cosine')
    ref_cohort = 1 - cdist(ref_emb, cohort_embs, metric='cosine')
    
    probe_topk = np.sort(probe_cohort, axis=1)[:, -top_k:]
    probe_mean = probe_topk.mean(axis=1, keepdims=True)
    probe_std = probe_topk.std(axis=1, keepdims=True) + 1e-8

    ref_topk = np.sort(ref_cohort, axis=1)[:, -top_k:]
    ref_mean = ref_topk.mean(axis=1, keepdims=True).T
    ref_std = ref_topk.std(axis=1, keepdims=True).T + 1e-8

    z_probe = (raw_scores - probe_mean) / probe_std
    z_ref = (raw_scores - ref_mean) / ref_std
    normalized_scores = 0.5 * (z_probe + z_ref)
    return normalized_scores
