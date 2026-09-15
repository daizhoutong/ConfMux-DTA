"""No-fit metrics for released prediction tables, Python >= 3.9.

CI excludes equal observed labels and awards half to prediction ties.
Rm2 retains the historical predictive-R2 convention, not squared Pearson r.
"""
import math
from itertools import groupby


def concordance_index(y, pred):
    ranks = {v: i + 1 for i, v in enumerate(sorted(set(pred)))}
    tree = [0] * (len(ranks) + 1)

    def prefix(i):
        total = 0
        while i:
            total += tree[i]
            i -= i & -i
        return total

    seen = permissible = 0
    score = 0.0
    for _, group in groupby(sorted(zip(y, pred)), key=lambda pair: pair[0]):
        values = [ranks[p] for _, p in group]
        for i in values:
            lower = prefix(i - 1)
            score += lower + 0.5 * (prefix(i) - lower)
            permissible += seen
        for i in values:
            while i < len(tree):
                tree[i] += 1
                i += i & -i
        seen += len(values)
    return score / permissible if permissible else None


def regression_metrics(y, pred, include_ci=True):
    if len(y) != len(pred) or len(y) < 2:
        raise ValueError('Paired arrays must have equal length >= 2')
    if not all(math.isfinite(v) for a in (y, pred) for v in a):
        raise ValueError('Non-finite label/prediction: do not silently drop records')
    n = len(y)
    mean_y, mean_p = math.fsum(y) / n, math.fsum(pred) / n
    sst = math.fsum((v - mean_y) ** 2 for v in y)
    ssp = math.fsum((v - mean_p) ** 2 for v in pred)
    sse = math.fsum((a - b) ** 2 for a, b in zip(y, pred))
    if sst <= 0:
        raise ValueError('R2 is undefined for constant labels')
    r2 = 1 - sse / sst
    pp = math.fsum(v * v for v in pred)
    k = math.fsum(a * b for a, b in zip(y, pred)) / pp if pp else None
    r02 = 1 - math.fsum((a - k * b) ** 2 for a, b in zip(y, pred)) / sst if k is not None else None
    r2m = r2 * (1 - math.sqrt(abs(r2 - r02))) if r02 is not None else None
    cov = math.fsum((a - mean_y) * (b - mean_p) for a, b in zip(y, pred))
    result = {
        'sample_count': n, 'r2': r2, 'rmse': math.sqrt(sse / n),
        'mae': math.fsum(abs(a - b) for a, b in zip(y, pred)) / n,
        'pearson_r': cov / math.sqrt(sst * ssp) if ssp else None,
        'r2m': r2m, 'r2m_definition': 'historical_predictive_R2_convention',
        'absolute_error_le_1_count': sum(abs(a - b) <= 1 for a, b in zip(y, pred)),
    }
    result['absolute_error_le_1_fraction'] = result['absolute_error_le_1_count'] / n
    if include_ci:
        result['ci'] = concordance_index(y, pred)
    return result
