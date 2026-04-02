from __future__ import annotations

from typing import Dict, List

import numpy as np


def _sample_mask_positions(num_attrs: int, ratio: float, rng: np.random.Generator) -> List[int]:
    k = max(1, int(round(num_attrs * ratio)))
    return sorted(rng.choice(np.arange(num_attrs), size=min(k, num_attrs), replace=False).tolist())


def generate_missing_masks(ground_truth: np.ndarray, ratio: float, scenario: str, seed: int = 42) -> Dict[str, List[int]]:
    if not (0 < ratio <= 1):
        raise ValueError("missing ratio must be in (0, 1]")
    if scenario not in {"mcar", "mar", "mnar", "random"}:
        raise ValueError("scenario must be one of mcar, mar, mnar, random")

    rng = np.random.default_rng(seed)
    num_samples, num_attrs = ground_truth.shape
    masks: Dict[str, List[int]] = {}

    for i in range(num_samples):
        base = _sample_mask_positions(num_attrs, ratio, rng)
        if scenario in {"mcar", "random"}:
            masks[str(i)] = base
        elif scenario == "mar":
            shifted = sorted({(x + 1) % num_attrs for x in base})
            masks[str(i)] = shifted
        elif scenario == "mnar":
            shifted = sorted({(x + 2) % num_attrs for x in base})
            masks[str(i)] = shifted
    return masks
