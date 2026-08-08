"""Shared compiled evaluation for additive rule ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastogb.logic import Conjunction, KeyValueProposition
from fastogb.parallel import numba_threads, resolve_numba_jobs
from fastogb.terms import LinearTerm


_OPERATION_CODES = {'<': 0, '<=': 1, '>': 2, '>=': 3, '==': 4, '!=': 5, 'missing': 6}
_PARALLEL_MIN_WORK = 100_000
_COLLECTION_PARALLEL_MIN_ROWS = 8_192


@dataclass(frozen=True)
class RuleEvaluationPlan:
    """Array descriptors for conjunction and linear rule queries."""

    offsets: np.ndarray
    columns: np.ndarray
    operations: np.ndarray
    operands: np.ndarray
    conjunction_rules: np.ndarray
    linear_rules: np.ndarray
    linear_columns: np.ndarray
    linear_locations: np.ndarray
    linear_scales: np.ndarray
    rule_count: int


@dataclass(frozen=True)
class RuleCollectionPlan:
    """Flattened descriptors and weights for several additive ensembles."""

    rules: RuleEvaluationPlan
    model_count: int
    conjunction_models: np.ndarray
    linear_models: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    linear_positive: np.ndarray
    linear_negative: np.ndarray


def build_rule_evaluation_plan(rules):
    """Build a compiled plan for package-native numeric queries, or return ``None`` for a custom query."""
    offsets = [0]
    columns, operations, operands, conjunction_rules = [], [], [], []
    linear_rules, linear_columns, linear_locations, linear_scales = [], [], [], []
    members = list(rules)
    for rule_index, rule in enumerate(members):
        query = rule.q
        if isinstance(query, LinearTerm) and query.proposition is None:
            linear_rules.append(rule_index)
            linear_columns.append(query.column_index)
            linear_locations.append(query.location)
            linear_scales.append(query.scale)
            continue
        propositions = [query.proposition] if isinstance(query, LinearTerm) else query
        if not isinstance(query, (Conjunction, LinearTerm)):
            return None
        for proposition in propositions:
            descriptor = _proposition_descriptor(proposition)
            if descriptor is None:
                return None
            column, operation, operand = descriptor
            columns.append(column)
            operations.append(operation)
            operands.append(operand)
        conjunction_rules.append(rule_index)
        offsets.append(len(columns))
    arrays = (
        (offsets, np.int64), (columns, np.int64), (operations, np.int8), (operands, np.float64),
        (conjunction_rules, np.int64), (linear_rules, np.int64), (linear_columns, np.int64),
        (linear_locations, np.float64), (linear_scales, np.float64),
    )
    values = [np.ascontiguousarray(items, dtype=dtype) for items, dtype in arrays]
    return RuleEvaluationPlan(*values, len(members))


def rule_structure_signature(rules):
    """Return a lightweight signature that changes when package-native query structures change."""
    return tuple(_query_signature(rule.q) for rule in rules)


def build_rule_collection_plan(ensembles):
    """Build one fused numeric evaluation plan for several ensembles."""
    members, rule_models = [], []
    for model_index, ensemble in enumerate(ensembles):
        model_rules = list(ensemble)
        members.extend(model_rules)
        rule_models.extend([model_index] * len(model_rules))
    rules = build_rule_evaluation_plan(members)
    if rules is None:
        return None
    rule_models = np.asarray(rule_models, dtype=np.int64)
    conjunction_models = np.ascontiguousarray(rule_models[rules.conjunction_rules])
    linear_models = np.ascontiguousarray(rule_models[rules.linear_rules])
    positive = np.ascontiguousarray([members[index].y for index in rules.conjunction_rules], dtype=np.float64)
    negative = np.ascontiguousarray([members[index].z for index in rules.conjunction_rules], dtype=np.float64)
    linear_positive = np.ascontiguousarray([members[index].y for index in rules.linear_rules], dtype=np.float64)
    linear_negative = np.ascontiguousarray([members[index].z for index in rules.linear_rules], dtype=np.float64)
    return RuleCollectionPlan(rules, len(ensembles), conjunction_models, linear_models, positive, negative,
                              linear_positive, linear_negative)


def evaluate_rule_ensemble(data, rules, plan=None, n_jobs=-1):
    """Evaluate rule outputs through an adaptive compiled path with a Python fallback for custom queries."""
    members = list(rules)
    values, fallback_data = _normalise_data(data)
    plan = build_rule_evaluation_plan(members) if plan is None else plan
    numeric = _numeric_values(values)
    if plan is None or numeric is None:
        return _python_scores(fallback_data, members)
    positive = np.ascontiguousarray([members[index].y for index in plan.conjunction_rules], dtype=np.float64)
    negative = np.ascontiguousarray([members[index].z for index in plan.conjunction_rules], dtype=np.float64)
    linear_positive = np.ascontiguousarray([members[index].y for index in plan.linear_rules], dtype=np.float64)
    linear_negative = np.ascontiguousarray([members[index].z for index in plan.linear_rules], dtype=np.float64)
    work = len(numeric) * max(len(plan.conjunction_rules) + len(plan.linear_rules) + len(plan.columns), 1)
    jobs = resolve_numba_jobs(n_jobs)
    parallel = jobs > 1 and work >= _PARALLEL_MIN_WORK
    kernel = _parallel_rule_scores if parallel else _serial_rule_scores
    manager = numba_threads(jobs) if parallel else _NullContext()
    with manager:
        return kernel(numeric, plan.offsets, plan.columns, plan.operations, plan.operands, positive, negative,
                      plan.linear_columns, plan.linear_locations, plan.linear_scales, linear_positive,
                      linear_negative)


def evaluate_rule_queries(data, rules, plan=None, n_jobs=-1):
    """Return the sample-by-rule query matrix through the shared compiled evaluator."""
    members = list(rules)
    values, fallback_data = _normalise_data(data)
    if not members:
        return np.empty((len(values), 0), dtype=np.float64)
    plan = build_rule_evaluation_plan(members) if plan is None else plan
    numeric = _numeric_values(values)
    if plan is None or numeric is None:
        columns = [np.asarray(rule.q(fallback_data), dtype=np.float64) for rule in members]
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    work = len(numeric) * max(plan.rule_count + len(plan.columns), 1)
    jobs = resolve_numba_jobs(n_jobs)
    parallel = jobs > 1 and work >= _PARALLEL_MIN_WORK
    kernel = _parallel_query_matrix if parallel else _serial_query_matrix
    manager = numba_threads(jobs) if parallel else _NullContext()
    with manager:
        return kernel(numeric, plan.rule_count, plan.offsets, plan.columns, plan.operations, plan.operands,
                      plan.conjunction_rules, plan.linear_rules, plan.linear_columns, plan.linear_locations,
                      plan.linear_scales)


def evaluate_rule_collection(data, plan, n_jobs=-1):
    """Evaluate several numeric ensembles in one adaptive compiled row traversal."""
    values, _ = _normalise_data(data)
    numeric = _numeric_values(values)
    if plan is None or numeric is None:
        return None
    rules = plan.rules
    work = len(numeric) * max(plan.model_count + rules.rule_count + len(rules.columns), 1)
    jobs = resolve_numba_jobs(n_jobs)
    parallel = jobs > 1 and len(numeric) >= _COLLECTION_PARALLEL_MIN_ROWS and work >= _PARALLEL_MIN_WORK
    kernel = _parallel_collection_scores if parallel else _serial_collection_scores
    manager = numba_threads(jobs) if parallel else _NullContext()
    with manager:
        return kernel(numeric, plan.model_count, rules.offsets, rules.columns, rules.operations, rules.operands,
                      plan.conjunction_models, plan.positive, plan.negative, plan.linear_models,
                      rules.linear_columns, rules.linear_locations, rules.linear_scales, plan.linear_positive,
                      plan.linear_negative)


def _proposition_descriptor(proposition):
    if not isinstance(proposition, KeyValueProposition) or proposition.column_index is None:
        return None
    operation = proposition.constraint.op
    if operation not in _OPERATION_CODES:
        return None
    try:
        operand = 0.0 if operation == 'missing' else float(proposition.constraint.value)
    except (TypeError, ValueError):
        return None
    return int(proposition.column_index), _OPERATION_CODES[operation], operand


def _query_signature(query):
    if isinstance(query, Conjunction):
        descriptors = tuple(_signature_descriptor(proposition) for proposition in query)
        return 'conjunction', id(query), descriptors
    if isinstance(query, LinearTerm):
        proposition = None if query.proposition is None else _signature_descriptor(query.proposition)
        return 'linear', id(query), query.column_index, query.location, query.scale, proposition
    return 'custom', id(query)


def _signature_descriptor(proposition):
    if not isinstance(proposition, KeyValueProposition):
        return id(proposition)
    value = proposition.constraint.value
    try:
        hash(value)
    except TypeError:
        value = repr(value)
    return id(proposition), proposition.column_index, proposition.constraint.op, value


def _normalise_data(data):
    values = data.to_numpy(copy=False) if hasattr(data, 'to_numpy') else np.asarray(data)
    if values.ndim == 1:
        values = values.reshape(1, -1)
        return values, values
    if values.ndim != 2:
        raise ValueError(f'Expected one- or two-dimensional feature data, received shape {values.shape}')
    return values, data


def _numeric_values(values):
    try:
        return np.ascontiguousarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _python_scores(data, rules):
    scores = np.zeros(len(data), dtype=np.float64)
    for rule in rules:
        scores += rule(data)
    return scores


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@njit(inline='always')
def _matches(value, operation, operand):
    if operation == 0:
        return value < operand
    if operation == 1:
        return value <= operand
    if operation == 2:
        return value > operand
    if operation == 3:
        return value >= operand
    if operation == 4:
        return value == operand
    if operation == 5:
        return value != operand
    return np.isnan(value)


@njit(inline='always')
def _rule_total(data, row, offsets, columns, operations, operands, positive, negative, linear_columns,
                linear_locations, linear_scales, linear_positive, linear_negative):
    total = 0.0
    for rule in range(len(positive)):
        satisfied = True
        for position in range(offsets[rule], offsets[rule + 1]):
            if not _matches(data[row, columns[position]], operations[position], operands[position]):
                satisfied = False
                break
        total += positive[rule] if satisfied else negative[rule]
    for rule in range(len(linear_positive)):
        value = (data[row, linear_columns[rule]] - linear_locations[rule]) / linear_scales[rule]
        value = value if np.isfinite(value) else 0.0
        total += value * linear_positive[rule] + (1.0 - value) * linear_negative[rule]
    return total


def _rule_scores(data, offsets, columns, operations, operands, positive, negative, linear_columns,
                 linear_locations, linear_scales, linear_positive, linear_negative):
    scores = np.empty(len(data), dtype=np.float64)
    for row in prange(len(data)):
        scores[row] = _rule_total(data, row, offsets, columns, operations, operands, positive, negative,
                                  linear_columns, linear_locations, linear_scales, linear_positive, linear_negative)
    return scores


def _query_matrix(data, rule_count, offsets, columns, operations, operands, conjunction_rules, linear_rules,
                  linear_columns, linear_locations, linear_scales):
    matrix = np.zeros((len(data), rule_count), dtype=np.float64)
    for row in prange(len(data)):
        for query in range(len(conjunction_rules)):
            satisfied = True
            for position in range(offsets[query], offsets[query + 1]):
                if not _matches(data[row, columns[position]], operations[position], operands[position]):
                    satisfied = False
                    break
            matrix[row, conjunction_rules[query]] = 1.0 if satisfied else 0.0
        for query in range(len(linear_rules)):
            value = (data[row, linear_columns[query]] - linear_locations[query]) / linear_scales[query]
            matrix[row, linear_rules[query]] = value if np.isfinite(value) else 0.0
    return matrix


def _collection_scores(data, model_count, offsets, columns, operations, operands, conjunction_models,
                       positive, negative, linear_models, linear_columns, linear_locations, linear_scales,
                       linear_positive, linear_negative):
    scores = np.zeros((len(data), model_count), dtype=np.float64)
    for row in prange(len(data)):
        for rule in range(len(positive)):
            satisfied = True
            for position in range(offsets[rule], offsets[rule + 1]):
                if not _matches(data[row, columns[position]], operations[position], operands[position]):
                    satisfied = False
                    break
            scores[row, conjunction_models[rule]] += positive[rule] if satisfied else negative[rule]
        for rule in range(len(linear_positive)):
            value = (data[row, linear_columns[rule]] - linear_locations[rule]) / linear_scales[rule]
            value = value if np.isfinite(value) else 0.0
            contribution = value * linear_positive[rule] + (1.0 - value) * linear_negative[rule]
            scores[row, linear_models[rule]] += contribution
    return scores


_serial_rule_scores = njit(cache=True, nogil=True)(_rule_scores)
_parallel_rule_scores = njit(cache=True, nogil=True, parallel=True)(_rule_scores)
_serial_query_matrix = njit(cache=True, nogil=True)(_query_matrix)
_parallel_query_matrix = njit(cache=True, nogil=True, parallel=True)(_query_matrix)
_serial_collection_scores = njit(cache=True, nogil=True)(_collection_scores)
_parallel_collection_scores = njit(cache=True, nogil=True, parallel=True)(_collection_scores)
