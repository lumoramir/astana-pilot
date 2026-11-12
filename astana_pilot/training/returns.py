"""Return targets for the critic warm-up (pure NumPy)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

GAMMA = 0.995


def normalized_returns(rew: np.ndarray, term: np.ndarray, ep_id: np.ndarray,
                       gamma: float = GAMMA) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Discounted returns of ``rew / std(running return)`` - what the critic sees under ``VecNormalize``.

    Episodes that ended by time limit (not ``term``) are treated as continuing with value 0 (slightly pessimistic).
    Returns ``(targets, (mean, var, count))``; ``var`` is the variance of the running discounted return.
    """
    ret, rets = 0.0, []
    for i in range(len(rew)):
        if i > 0 and ep_id[i] != ep_id[i - 1]:
            ret = 0.0
        ret = ret * gamma + float(rew[i])
        rets.append(ret)
    rets_a = np.asarray(rets, dtype=np.float64)
    var = float(rets_a.var()) + 1e-8
    std = np.sqrt(var)
    scaled = np.clip(rew.astype(np.float64) / std, -10.0, 10.0)
    tgt = np.zeros(len(rew))
    run = 0.0
    for i in range(len(rew) - 1, -1, -1):
        if i == len(rew) - 1 or ep_id[i] != ep_id[i + 1]:
            run = 0.0
        run = scaled[i] + gamma * run
        tgt[i] = run
    return tgt.astype(np.float32), (float(rets_a.mean()), var, float(len(rets_a)))
