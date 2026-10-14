"""Rule selectors for structural adaptation of an additive ensemble."""

from fastogb.logic import Conjunction


def no_replacement(ensemble):
    """Keep every rule; return an empty list of zero-based ensemble indices."""
    return []


def smallest_absolute_weight(ensemble):
    """Select the weakest learned conjunction, using estimator base-term metadata when present."""
    base_count = getattr(ensemble, '_base_rule_count_', 0)
    indices = [index for index, rule in enumerate(ensemble) if index >= base_count and isinstance(rule.q, Conjunction)]
    return [min(indices, key=lambda index: abs(ensemble[index].y))] if indices else []
