"""Adapters for rules produced by external tree-rule libraries."""

from __future__ import annotations

from fastogb.logic import Conjunction, KeyValueProposition, constraint_from_op_string


def query_from_external_rule(rule, feature_names=None):
    """Convert an object exposing RuleFit-style conditions into a conjunction."""
    positions = {} if feature_names is None else {name: index for index, name in enumerate(feature_names)}
    propositions = []
    for condition in rule.conditions:
        constraint = constraint_from_op_string(condition.operator, condition.threshold)
        index = positions.get(condition.feature_name)
        propositions.append(KeyValueProposition(condition.feature_name, constraint, index))
    return Conjunction(propositions)


def patch_from_rule(rule, fc='blue', axes=None, x_min=-4, x_max=4, y_min=-4, y_max=4):
    """Create a Matplotlib rectangle for a two-dimensional external rule."""
    from matplotlib.patches import Rectangle
    axes = {'x1': 0, 'x2': 1} if axes is None else axes
    bounds = [[x_min, x_max], [y_min, y_max]]
    operator_bound = {'<=': 1, '<': 1, '>=': 0, '>': 0}
    operator_comp = {'<=': min, '<': min, '>=': max, '>': max}
    for condition in rule.conditions:
        axis = axes[condition.feature_name]
        bound = operator_bound[condition.operator]
        bounds[axis][bound] = operator_comp[condition.operator](condition.threshold, bounds[axis][bound])
    width = bounds[0][1] - bounds[0][0]
    height = bounds[1][1] - bounds[1][0]
    return Rectangle((bounds[0][0], bounds[1][0]), width, height, fill=True, color='black', lw=1, ls='-',
                     fc=fc, alpha=0.2)
