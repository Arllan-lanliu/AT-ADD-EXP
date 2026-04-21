import random
import numpy as np

_DEV_SUBSAMPLE_BUDGET = {
    "speech":  1000,
    "sound":   1000,
    "singing": 1000,
    "music":   1000,
}

def _stratified_sample(rows, target_n, rng):
    if len(rows) <= target_n:
        return list(rows)

    from collections import defaultdict

    # 第一层：按 label 分组
    label_groups = defaultdict(list)
    for r in rows:
        label_groups[r[2]].append(r)  # r[2] = label

    # 按原始比例分配 budget 给每个 label
    total = len(rows)
    label_budgets = {}
    for label, items in label_groups.items():
        label_budgets[label] = max(1, round(target_n * len(items) / total))

    # 修正 budget 总和（避免舍入误差）
    diff = target_n - sum(label_budgets.values())
    if diff != 0:
        # 把差值加到最大的 label 上
        largest = max(label_budgets, key=lambda k: label_budgets[k])
        label_budgets[largest] += diff

    selected = []

    # 第二层：在每个 label 内，按 generator 均匀采样
    for label, items in label_groups.items():
        budget = label_budgets[label]
        gen_groups = defaultdict(list)
        for r in items:
            gen_groups[r[3]].append(r)  # r[3] = generator

        # 复用原来的均匀分配逻辑
        selected.extend(_sample_from_groups(gen_groups, budget, rng))

    return selected

def _sample_from_groups(groups, budget, rng):
    if sum(len(v) for v in groups.values()) <= budget:
        return [r for v in groups.values() for r in v]

    remaining_budget = budget
    remaining_keys = sorted(groups.keys())
    selected = []

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

    if remaining_keys:
        per_cell = remaining_budget // len(remaining_keys)
        extra    = remaining_budget %  len(remaining_keys)
        for i, k in enumerate(remaining_keys):
            n = per_cell + (1 if i < extra else 0)
            selected.extend(rng.sample(groups[k], n))

    return selected