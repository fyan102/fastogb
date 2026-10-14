"""Array-only fastogb CPU kernels compiled with Numba."""

from __future__ import annotations

from functools import partial
from math import exp, inf, sqrt

import numpy as np
from numba import njit, prange


compile_kernel = partial(njit, cache=True, nogil=True)


def orthogonal_prefix_values_kernel(projected_gradient, basis, order, epsilon, regularisation):
    values = np.full(len(order), -inf, dtype=np.float64)
    projection = np.zeros(basis.shape[1], dtype=np.float64)
    gradient_sum = 0.0
    machine_epsilon = np.finfo(np.float64).eps
    for position in range(len(order)):
        row = order[position]
        gradient_sum += projected_gradient[row]
        projection_squared = 0.0
        for column in range(basis.shape[1]):
            projection[column] += basis[row, column]
            projection_squared += projection[column] * projection[column]
        query_squared = float(position + 1)
        residual = query_squared - projection_squared
        scale = max(query_squared, projection_squared, 1.0)
        tolerance = sqrt(32.0 * machine_epsilon) * scale
        if residual < -tolerance:
            raise FloatingPointError('Orthogonal projection exceeds prefix norm')
        if residual > tolerance or regularisation > 0:
            values[position] = abs(gradient_sum) / (sqrt(max(residual, 0.0) + regularisation) + epsilon)
    return values


def orthogonal_extent_norm_kernel(basis, extent, regularisation):
    projection = np.zeros(basis.shape[1], dtype=np.float64)
    for position in range(len(extent)):
        row = extent[position]
        for column in range(basis.shape[1]):
            projection[column] += basis[row, column]
    projection_squared = 0.0
    for column in range(len(projection)):
        projection_squared += projection[column] * projection[column]
    query_squared = float(len(extent))
    residual = query_squared - projection_squared
    tolerance = sqrt(32.0 * np.finfo(np.float64).eps) * max(query_squared, projection_squared, 1.0)
    if residual < -tolerance:
        raise FloatingPointError('Orthogonal projection exceeds query norm')
    return sqrt((residual if residual > tolerance else 0.0) + regularisation)


def squared_prefix_bound_kernel(gradient, hessian, extent, regularisation, scale):
    if len(extent) == 0:
        return -inf
    forward_gradient = 0.0
    reverse_gradient = 0.0
    forward_hessian = regularisation
    reverse_hessian = regularisation
    best = 0.0
    for position in range(len(extent)):
        forward = extent[position]
        reverse = extent[len(extent) - position - 1]
        forward_gradient += gradient[forward]
        reverse_gradient += gradient[reverse]
        forward_hessian += hessian[forward]
        reverse_hessian += hessian[reverse]
        if forward_hessian > 0:
            best = max(best, forward_gradient * forward_gradient / forward_hessian)
        if reverse_hessian > 0:
            best = max(best, reverse_gradient * reverse_gradient / reverse_hessian)
    return best / scale


def absolute_prefix_bound_kernel(gradient, extent, regularisation, scale):
    if len(extent) == 0:
        return -inf
    forward_gradient = 0.0
    reverse_gradient = 0.0
    best = 0.0
    for position in range(len(extent)):
        forward_gradient += gradient[extent[position]]
        reverse_gradient += gradient[extent[len(extent) - position - 1]]
        denominator = sqrt(position + 1.0 + regularisation)
        best = max(best, abs(forward_gradient) / denominator, abs(reverse_gradient) / denominator)
    return best / scale


def mask_intersection_extent_kernel(left, right):
    mask = np.empty(len(left), dtype=np.bool_)
    count = 0
    for index in range(len(left)):
        selected = left[index] and right[index]
        mask[index] = selected
        count += int(selected)
    extent = np.empty(count, dtype=np.int64)
    position = 0
    for index in range(len(mask)):
        if mask[index]:
            extent[position] = index
            position += 1
    return mask, extent


def is_subset_kernel(left, right):
    for index in range(len(left)):
        if left[index] and not right[index]:
            return False
    return True


def pack_masks_kernel(masks):
    word_count = (masks.shape[1] + 63) // 64
    packed = np.zeros((masks.shape[0], word_count), dtype=np.uint64)
    one = np.uint64(1)
    for mask in prange(masks.shape[0]):
        for row in range(masks.shape[1]):
            if masks[mask, row]:
                packed[mask, row // 64] |= one << np.uint64(row % 64)
    return packed


def packed_intersection_extent_kernel(left, right, row_count):
    packed = np.empty(len(left), dtype=np.uint64)
    count = 0
    one = np.uint64(1)
    for word_index in range(len(left)):
        word = left[word_index] & right[word_index]
        packed[word_index] = word
        while word:
            count += 1
            word &= word - one
    extent = np.empty(count, dtype=np.int64)
    position = 0
    for word_index in range(len(packed)):
        word = packed[word_index]
        bit = 0
        while word:
            while not word & one:
                word >>= one
                bit += 1
            row = 64 * word_index + bit
            if row < row_count:
                extent[position] = row
                position += 1
            word >>= one
            bit += 1
    return packed, extent[:position]


def packed_is_subset_kernel(left, right):
    for word in range(len(left)):
        if left[word] & ~right[word]:
            return False
    return True


def packed_to_extent_kernel(packed, row_count):
    count = 0
    one = np.uint64(1)
    for word in packed:
        while word:
            count += 1
            word &= word - one
    extent = np.empty(count, dtype=np.int64)
    position = 0
    for row in range(row_count):
        if packed[row // 64] & (one << np.uint64(row % 64)):
            extent[position] = row
            position += 1
    return extent


def prefix_objective_batch_kernel(parent, attributes, indices, gradient, hessian, regularisation, scale, mode):
    children = np.empty((len(indices), len(parent)), dtype=np.uint64)
    values = np.full(len(indices), -inf, dtype=np.float64)
    bounds = np.full(len(indices), -inf, dtype=np.float64)
    one = np.uint64(1)
    for output in prange(len(indices)):
        attribute = indices[output]
        count = 0
        gradient_sum = 0.0
        hessian_sum = 0.0
        forward_gradient = 0.0
        forward_hessian = regularisation
        best = 0.0
        for word in range(len(parent)):
            children[output, word] = parent[word] & attributes[attribute, word]
        for row in range(len(gradient)):
            if children[output, row // 64] & (one << np.uint64(row % 64)):
                count += 1
                gradient_sum += gradient[row]
                hessian_sum += hessian[row]
                forward_gradient += gradient[row]
                forward_hessian += hessian[row]
                if mode == 0 and forward_hessian > 0:
                    best = max(best, forward_gradient * forward_gradient / forward_hessian)
                elif mode == 1:
                    best = max(best, abs(forward_gradient))
                elif mode == 2:
                    best = max(best, abs(forward_gradient) / sqrt(count + regularisation))
        reverse_gradient = 0.0
        reverse_hessian = regularisation
        reverse_count = 0
        for row in range(len(gradient) - 1, -1, -1):
            if children[output, row // 64] & (one << np.uint64(row % 64)):
                reverse_count += 1
                reverse_gradient += gradient[row]
                reverse_hessian += hessian[row]
                if mode == 0 and reverse_hessian > 0:
                    best = max(best, reverse_gradient * reverse_gradient / reverse_hessian)
                elif mode == 1:
                    best = max(best, abs(reverse_gradient))
                elif mode == 2:
                    best = max(best, abs(reverse_gradient) / sqrt(reverse_count + regularisation))
        if count:
            if mode == 0:
                denominator = regularisation + hessian_sum
                if denominator > 0:
                    values[output] = gradient_sum * gradient_sum / (scale * denominator)
                bounds[output] = best / scale
            elif mode == 1:
                values[output] = abs(gradient_sum)
                bounds[output] = best
            else:
                values[output] = abs(gradient_sum) / (scale * sqrt(count + regularisation))
                bounds[output] = best / scale
    return children, values, bounds


def orthogonal_objective_batch_kernel(parent, attributes, indices, gradient, basis, regularisation, epsilon,
                                gradient_norm):
    children = np.empty((len(indices), len(parent)), dtype=np.uint64)
    values = np.full(len(indices), -inf, dtype=np.float64)
    bounds = np.full(len(indices), -inf, dtype=np.float64)
    one = np.uint64(1)
    for output in prange(len(indices)):
        attribute = indices[output]
        projection = np.zeros(basis.shape[1], dtype=np.float64)
        gradient_sum = 0.0
        absolute_gradient_sum = 0.0
        count = 0
        for word in range(len(parent)):
            children[output, word] = parent[word] & attributes[attribute, word]
        for row in range(len(gradient)):
            if children[output, row // 64] & (one << np.uint64(row % 64)):
                count += 1
                gradient_sum += gradient[row]
                absolute_gradient_sum += abs(gradient[row])
                for column in range(basis.shape[1]):
                    projection[column] += basis[row, column]
        projection_squared = 0.0
        for column in range(len(projection)):
            projection_squared += projection[column] * projection[column]
        residual = count - projection_squared
        tolerance = sqrt(32.0 * np.finfo(np.float64).eps) * max(count, projection_squared, 1.0)
        if residual < -tolerance:
            values[output] = np.nan
            bounds[output] = np.nan
        else:
            norm = sqrt(max(residual, 0.0) + regularisation)
            if count and (regularisation > 0 or norm > epsilon):
                values[output] = abs(gradient_sum) / (norm + epsilon)
            if count:
                lower_denominator = sqrt(regularisation) + epsilon
                bounds[output] = min(absolute_gradient_sum / lower_denominator, gradient_norm)
    return children, values, bounds


def orthogonal_greedy_values_kernel(parent, attributes, indices, gradient, basis, regularisation, epsilon):
    values = np.full(len(indices), -inf, dtype=np.float64)
    for output in prange(len(indices)):
        attribute = indices[output]
        projection = np.zeros(basis.shape[1], dtype=np.float64)
        gradient_sum = 0.0
        count = 0
        for row in range(len(parent)):
            if parent[row] and attributes[attribute, row]:
                count += 1
                gradient_sum += gradient[row]
                for column in range(basis.shape[1]):
                    projection[column] += basis[row, column]
        projection_squared = 0.0
        for column in range(len(projection)):
            projection_squared += projection[column] * projection[column]
        residual = count - projection_squared
        tolerance = sqrt(32.0 * np.finfo(np.float64).eps) * max(count, projection_squared, 1.0)
        if residual < -tolerance:
            values[output] = np.nan
        else:
            norm = sqrt(max(residual, 0.0) + regularisation)
            if count and (regularisation > 0 or norm > epsilon):
                values[output] = abs(gradient_sum) / (norm + epsilon)
    return values


def find_small_packed_critical_index_kernel(gen_index, extension, closure, attributes):
    for attribute in range(gen_index):
        if closure[attribute]:
            continue
        subset = True
        for word in range(len(extension)):
            if extension[word] & ~attributes[attribute, word]:
                subset = False
                break
        if subset:
            return attribute
    return len(closure)


def complete_packed_closure_kernel(gen_index, extension, closure, attributes):
    critical_index = len(closure)
    for attribute in range(gen_index + 1, len(closure)):
        if closure[attribute]:
            continue
        subset = True
        for word in range(len(extension)):
            if extension[word] & ~attributes[attribute, word]:
                subset = False
                break
        if subset:
            critical_index = min(critical_index, attribute)
            closure[attribute] = True
    return critical_index


def proposition_matrix_kernel(values, columns, operations, operands):
    matrix = np.empty((values.shape[0], len(columns)), dtype=np.bool_)
    for proposition in prange(len(columns)):
        column = columns[proposition]
        operation = operations[proposition]
        operand = operands[proposition]
        for row in range(values.shape[0]):
            value = values[row, column]
            if operation == 0:
                matrix[row, proposition] = value < operand
            elif operation == 1:
                matrix[row, proposition] = value <= operand
            elif operation == 2:
                matrix[row, proposition] = value > operand
            elif operation == 3:
                matrix[row, proposition] = value >= operand
            elif operation == 4:
                matrix[row, proposition] = value == operand
            elif operation == 5:
                matrix[row, proposition] = np.isnan(value)
            else:
                matrix[row, proposition] = value == -1.0
    return matrix


def rule_ensemble_scores_kernel(context, offsets, proposition_indices, positive, negative):
    scores = np.zeros(context.shape[0], dtype=np.float64)
    for row in prange(context.shape[0]):
        total = 0.0
        for rule in range(len(positive)):
            satisfied = True
            for position in range(offsets[rule], offsets[rule + 1]):
                if not context[row, proposition_indices[position]]:
                    satisfied = False
                    break
            total += positive[rule] if satisfied else negative[rule]
        scores[row] = total
    return scores


def loss_derivatives_kernel(target, scores, loss_code):
    gradient = np.empty(len(target), dtype=np.float64)
    hessian = np.empty(len(target), dtype=np.float64)
    for row in prange(len(target)):
        if loss_code == 0:
            gradient[row] = 2.0 * (scores[row] - target[row])
            hessian[row] = 2.0
        elif loss_code == 1:
            margin = target[row] * scores[row]
            if margin >= 0:
                exponential = exp(-margin)
                probability = exponential / (1.0 + exponential)
            else:
                exponential = exp(margin)
                probability = 1.0 / (1.0 + exponential)
            gradient[row] = -target[row] * probability
            hessian[row] = probability * (1.0 - probability)
        else:
            clipped = min(max(scores[row], -745.0), 709.0)
            mean = exp(clipped)
            gradient[row] = mean - target[row]
            hessian[row] = mean
    return gradient, hessian


_compiled_orthogonal_prefix_values = compile_kernel(orthogonal_prefix_values_kernel)
_compiled_orthogonal_extent_norm = compile_kernel(orthogonal_extent_norm_kernel)
_compiled_squared_prefix_bound = compile_kernel(squared_prefix_bound_kernel)
_compiled_absolute_prefix_bound = compile_kernel(absolute_prefix_bound_kernel)
_compiled_mask_intersection_extent = compile_kernel(mask_intersection_extent_kernel)
_compiled_is_subset = compile_kernel(is_subset_kernel)
_compiled_pack_masks = compile_kernel(pack_masks_kernel, parallel=True)
_compiled_packed_intersection_extent = compile_kernel(packed_intersection_extent_kernel)
_compiled_packed_is_subset = compile_kernel(packed_is_subset_kernel)
_compiled_packed_to_extent = compile_kernel(packed_to_extent_kernel)
_compiled_prefix_objective_batch = compile_kernel(prefix_objective_batch_kernel)
_compiled_parallel_prefix_objective_batch = compile_kernel(prefix_objective_batch_kernel, parallel=True)
_compiled_orthogonal_objective_batch = compile_kernel(orthogonal_objective_batch_kernel)
_compiled_parallel_orthogonal_objective_batch = compile_kernel(orthogonal_objective_batch_kernel, parallel=True)
_compiled_orthogonal_greedy_values = compile_kernel(orthogonal_greedy_values_kernel)
_compiled_parallel_orthogonal_greedy_values = compile_kernel(orthogonal_greedy_values_kernel, parallel=True)
_compiled_find_small_packed_critical_index = compile_kernel(find_small_packed_critical_index_kernel)
_compiled_complete_packed_closure = compile_kernel(complete_packed_closure_kernel)
_compiled_proposition_matrix = compile_kernel(proposition_matrix_kernel)
_compiled_parallel_proposition_matrix = compile_kernel(proposition_matrix_kernel, parallel=True)
_compiled_rule_ensemble_scores = compile_kernel(rule_ensemble_scores_kernel)
_compiled_parallel_rule_ensemble_scores = compile_kernel(rule_ensemble_scores_kernel, parallel=True)
_compiled_loss_derivatives = compile_kernel(loss_derivatives_kernel)
_compiled_parallel_loss_derivatives = compile_kernel(loss_derivatives_kernel, parallel=True)


def orthogonal_prefix_values(projected_gradient, basis, order, epsilon, regularisation=0.0):
    """Return Yang Algorithm 3 objective values for every ordered prefix."""
    gradient = np.ascontiguousarray(projected_gradient, dtype=np.float64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    order = np.ascontiguousarray(order, dtype=np.int64)
    return _compiled_orthogonal_prefix_values(gradient, basis, order, float(epsilon), float(regularisation))


def orthogonal_extent_norm(basis, extent, regularisation=0.0):
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    extent = np.ascontiguousarray(extent, dtype=np.int64)
    return _compiled_orthogonal_extent_norm(basis, extent, float(regularisation))


def squared_prefix_bound(gradient, hessian, extent, regularisation, scale):
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    hessian = np.ascontiguousarray(hessian, dtype=np.float64)
    extent = np.ascontiguousarray(extent, dtype=np.int64)
    return float(_compiled_squared_prefix_bound(gradient, hessian, extent, regularisation, scale))


def absolute_prefix_bound(gradient, extent, regularisation, scale):
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    extent = np.ascontiguousarray(extent, dtype=np.int64)
    return float(_compiled_absolute_prefix_bound(gradient, extent, regularisation, scale))


def mask_intersection_extent(left, right):
    left = np.ascontiguousarray(left, dtype=np.bool_)
    right = np.ascontiguousarray(right, dtype=np.bool_)
    return _compiled_mask_intersection_extent(left, right)


def is_subset(left, right):
    left = np.ascontiguousarray(left, dtype=np.bool_)
    right = np.ascontiguousarray(right, dtype=np.bool_)
    return bool(_compiled_is_subset(left, right))


def pack_masks(masks):
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    return _compiled_pack_masks(masks)


def full_packed_extent(row_count):
    words = np.full((row_count + 63) // 64, np.iinfo(np.uint64).max, dtype=np.uint64)
    remainder = row_count % 64
    if remainder:
        words[-1] = np.uint64((1 << remainder) - 1)
    return words


def packed_intersection_extent(left, right, row_count):
    left = np.ascontiguousarray(left, dtype=np.uint64)
    right = np.ascontiguousarray(right, dtype=np.uint64)
    return _compiled_packed_intersection_extent(left, right, int(row_count))


def packed_is_subset(left, right):
    left = np.ascontiguousarray(left, dtype=np.uint64)
    right = np.ascontiguousarray(right, dtype=np.uint64)
    return bool(_compiled_packed_is_subset(left, right))


def packed_to_extent(packed, row_count):
    packed = np.ascontiguousarray(packed, dtype=np.uint64)
    return _compiled_packed_to_extent(packed, int(row_count))


def packed_extent_signature(packed, row_count):
    byte_count = (row_count + 7) // 8
    return np.asarray(packed, dtype='<u8').view(np.uint8)[:byte_count].tobytes()


def find_small_packed_critical_index(gen_index, extension, closure, attributes):
    extension = np.ascontiguousarray(extension, dtype=np.uint64)
    closure = np.ascontiguousarray(closure, dtype=np.bool_)
    attributes = np.ascontiguousarray(attributes, dtype=np.uint64)
    return int(_compiled_find_small_packed_critical_index(gen_index, extension, closure, attributes))


def complete_packed_closure(gen_index, extension, closure, attributes):
    extension = np.ascontiguousarray(extension, dtype=np.uint64)
    attributes = np.ascontiguousarray(attributes, dtype=np.uint64)
    return int(_compiled_complete_packed_closure(gen_index, extension, closure, attributes))


def prefix_objective_batch(parent, attributes, indices, gradient, hessian, regularisation, scale, mode,
                           parallel=False):
    parent = np.ascontiguousarray(parent, dtype=np.uint64)
    attributes = np.ascontiguousarray(attributes, dtype=np.uint64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    hessian = np.ascontiguousarray(hessian, dtype=np.float64)
    kernel = _compiled_parallel_prefix_objective_batch if parallel else _compiled_prefix_objective_batch
    return kernel(parent, attributes, indices, gradient, hessian, float(regularisation), float(scale), int(mode))


def orthogonal_objective_batch(parent, attributes, indices, gradient, basis, regularisation, epsilon,
                               gradient_norm, parallel=False):
    parent = np.ascontiguousarray(parent, dtype=np.uint64)
    attributes = np.ascontiguousarray(attributes, dtype=np.uint64)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    kernel = _compiled_parallel_orthogonal_objective_batch if parallel else _compiled_orthogonal_objective_batch
    result = kernel(parent, attributes, indices, gradient, basis, float(regularisation), float(epsilon),
                    float(gradient_norm))
    if np.any(np.isnan(result[1])):
        raise FloatingPointError('Orthogonal projection exceeds query norm')
    return result


def orthogonal_greedy_values(parent, attributes, indices, gradient, basis, regularisation, epsilon, parallel=False):
    """Evaluate all greedy OGB refinements in one compiled candidate loop."""
    parent = np.ascontiguousarray(parent, dtype=np.bool_)
    attributes = np.ascontiguousarray(attributes, dtype=np.bool_)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    gradient = np.ascontiguousarray(gradient, dtype=np.float64)
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    kernel = _compiled_parallel_orthogonal_greedy_values if parallel else _compiled_orthogonal_greedy_values
    values = kernel(parent, attributes, indices, gradient, basis, float(regularisation), float(epsilon))
    if np.any(np.isnan(values)):
        raise FloatingPointError('Orthogonal projection exceeds query norm')
    return values


def proposition_matrix(values, columns, operations, operands, parallel=None):
    """Evaluate encoded propositions using a serial or column-parallel compiled kernel."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    columns = np.ascontiguousarray(columns, dtype=np.int64)
    operations = np.ascontiguousarray(operations, dtype=np.int8)
    operands = np.ascontiguousarray(operands, dtype=np.float64)
    use_parallel = values.shape[0] * len(columns) >= 100_000 if parallel is None else bool(parallel)
    kernel = _compiled_parallel_proposition_matrix if use_parallel else _compiled_proposition_matrix
    return kernel(values, columns, operations, operands)


def rule_ensemble_scores(context, offsets, proposition_indices, positive, negative, parallel=None):
    """Evaluate a rule ensemble using one compiled row loop."""
    context = np.ascontiguousarray(context, dtype=np.bool_)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    proposition_indices = np.ascontiguousarray(proposition_indices, dtype=np.int64)
    positive = np.ascontiguousarray(positive, dtype=np.float64)
    negative = np.ascontiguousarray(negative, dtype=np.float64)
    work = context.shape[0] * max(len(positive), len(proposition_indices), 1)
    use_parallel = work >= 100_000 if parallel is None else bool(parallel)
    kernel = _compiled_parallel_rule_ensemble_scores if use_parallel else _compiled_rule_ensemble_scores
    return kernel(context, offsets, proposition_indices, positive, negative)


def loss_derivatives(target, scores, loss_code, parallel=None):
    """Calculate the first two pointwise derivatives in one compiled pass."""
    target = np.ascontiguousarray(target, dtype=np.float64)
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    use_parallel = len(target) >= 100_000 if parallel is None else bool(parallel)
    kernel = _compiled_parallel_loss_derivatives if use_parallel else _compiled_loss_derivatives
    return kernel(target, scores, int(loss_code))


def numba_available():
    return True
