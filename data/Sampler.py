import random
import numpy as np

_DEV_SUBSAMPLE_BUDGET = {
    "speech":  1000,
    "sound":   1000,
    "singing": 1000,
    "music":   1000,
}


def _stratified_sample(rows, target_n, rng):
    """
    Sample up to *target_n* rows from *rows*, distributed as evenly as
    possible across all (label, generator) cells.

    - rows: list of (filename, class_type, label, generator) tuples
    - target_n: desired number of samples
    - rng: random.Random instance (for reproducibility)

    If len(rows) <= target_n the full list is returned unchanged.
    Small cells (fewer items than their fair share) contribute all their
    rows; the saved budget is redistributed to the remaining cells.
    """
    if len(rows) <= target_n:
        return list(rows)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r[2], r[3])].append(r)   # key = (label, generator)

    remaining_budget = target_n
    remaining_keys = sorted(groups.keys())
    selected = []

    # Iteratively absorb cells that are smaller than their fair share,
    # then redistribute the freed budget to the surviving larger cells.
    changed = True
    while changed and remaining_keys:
        changed = False
        fair_share = remaining_budget / len(remaining_keys)
        next_keys = []
        for k in remaining_keys:
            if len(groups[k]) <= fair_share:
                selected.extend(groups[k])
                remaining_budget -= len(groups[k])
                changed = True
            else:
                next_keys.append(k)
        remaining_keys = next_keys

    # All surviving cells are larger than their fair share — allocate evenly.
    if remaining_keys:
        per_cell = remaining_budget // len(remaining_keys)
        extra    = remaining_budget %  len(remaining_keys)
        for i, k in enumerate(remaining_keys):
            n = per_cell + (1 if i < extra else 0)
            selected.extend(rng.sample(groups[k], n))

    return selected