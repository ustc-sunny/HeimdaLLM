"""DClip + public top-k+; adapted from InvisibleInk (GPL-3.0).

Upstream: cerai-iitm/invisibleink, commit 2141ee265094612e85202b9e2745962b548b11b5.
This pilot uses conservative REPLACE-ONE full-record accounting, not the
upstream replace-by-null epsilon. Floating-point sampler is research software.
"""
import math
import numpy as np

def calibration(epsilon=8., delta=1e-5, tokens=96, references=3, temperature=1.):
    # rho + 2 sqrt(rho log(1/delta)) = epsilon.
    rho = (math.sqrt(math.log(1/delta) + epsilon) - math.sqrt(math.log(1/delta)))**2
    # Replacement: sensitivity 2C/B, bounded range 4C/(B*tau).
    clip = references * temperature * math.sqrt(rho / (2 * tokens))
    return {'epsilon':epsilon, 'delta':delta, 'max_tokens':tokens,
            'references':references, 'temperature':temperature, 'rho':rho, 'clip':clip}

def support(public, topk, clip, references):
    # Upstream get_topk, with B defined as PRIVATE references, not B+1.
    threshold = np.partition(public, -topk)[-topk] - 2 * clip / references
    return public >= threshold

def scores(public, private, arm, clip):
    public = np.asarray(public, dtype=np.float64)
    if arm == 'PUBLIC': return public
    private = np.asarray(private, dtype=np.float64)
    if arm == 'NONPRIVATE': return private.mean(axis=0)
    if arm == 'DP_E8': return public + np.clip(private-public, -clip, clip).mean(axis=0)
    raise ValueError(arm)

def probabilities(logits, mask, temperature):
    ids = np.flatnonzero(mask)
    s = np.asarray(logits[ids], dtype=np.float64) / temperature
    if not np.isfinite(s).all(): raise RuntimeError('nonfinite logits; stop without reroll')
    p = np.exp(s-s.max()); p /= p.sum()
    return ids, p

def sample(ids, p, uniform):
    # Fresh OS entropy for DP; no seed/state is serialized. See audit limitations.
    cdf = np.cumsum(p, dtype=np.float64); cdf[-1] = 1.
    return int(ids[min(int(np.searchsorted(cdf, uniform, side='right')), len(ids)-1)])
