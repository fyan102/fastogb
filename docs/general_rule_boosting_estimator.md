# GeneralRuleBoostingEstimator interface

`GeneralRuleBoostingEstimator` is the single user-facing estimator for constructing additive rule ensembles. Its behaviour is configured by composing a query-selection objective, a search method, a loss and a weight-update method. Install the `fastogb` distribution from PyPI with `pip install fastogb`, then import its public API from `fastogb`. This document combines constructor and method contracts with task-based examples for initial training, adapting coefficients, replacing rules, choosing search algorithms and evaluating collections. The adaptation examples describe the 0.2.0 interface, so ensure that your installed version provides these methods before using them. Input matrices use NumPy-compatible values and positional feature alignment; categorical encoding and readable feature names are handled through the documented encoding configuration.

## Constructor

```python
GeneralRuleBoostingEstimator(num_rules=3, objective_function=GradientBoostingObjectiveXGB, weight_update_method=None,
                             loss='squared', reg=1.0, search='exhaustive', include_default_rule=False,
                             include_linear_terms=False, max_col_attr=10, search_params=None, verbose=False,
                             basis_rtol=None, objective_params=None, max_components=10_000, n_jobs=1)
```

### `num_rules=3`

`num_rules` is a non-negative integer giving the requested number of learned conjunction rules. The optional default
rule and optional linear terms are base terms and do not count towards this number. Training can finish with fewer
rules when no admissible independent query remains or when `max_components` is reached. The realised number of
rules can be inspected through `rules_`, `_base_rule_count_` and `stopping_reason_` after fitting.

### `objective_function=GradientBoostingObjectiveXGB`

`objective_function` is an objective class used to score candidate queries during search. Pass the class itself,
rather than an already constructed instance, because a fresh objective is built at every boosting iteration from the
current scores, gradients, rule ensemble and orthogonal basis.

`GradientBoostingObjectiveXGB` uses the second-order XGBoost loss-reduction criterion and is the default.
`GradientBoostingObjectiveMWG` maximises absolute gradient mass. `GradientBoostingObjectiveGPE` uses gradient
projection efficiency. `OrthogonalBoostingObjective` projects the gradient and candidate rule indicators away from
the span of existing rules; it is the objective used for Orthogonal Gradient Boosting.

A custom objective class must accept the training data, target and estimator-supplied keyword arguments. It must
implement `__call__(extent)` and `bound(extent)` for exact search. Depth-aware objectives may accept a second `depth`
argument. Optimised greedy search can additionally provide `greedy_values`, and orthogonal beam search requires
`prefix_values`.

### `weight_update_method=None`

`weight_update_method` accepts an instance of `KeepWeight`, `LineSearch`, `FullyCorrective` or a compatible custom
`WeightUpdateMethod`. `None` selects `KeepWeight()`. The estimator copies the updater for fitting and overwrites the
copy's `loss` and `reg` with the estimator settings, so users normally write `FullyCorrective()` without repeating
the loss or regularisation.

`KeepWeight()` retains the weight calculated when each rule is selected. `LineSearch()` reoptimises the newest rule
while holding earlier weights fixed. `FullyCorrective()` jointly reoptimises all active learned-rule weights after
every iteration. With squared loss, `FullyCorrective` solves a regularised linear system. With logistic or Poisson
loss, it uses a SciPy optimiser; its `solver` and `options` are configured on the updater instance.

`FullyCorrective(loss='squared', reg=1.0, solver='Newton-CG', options=None, learning_rate=1.0, fixed_prefix=None)`
preserves the positional order of all previously supported arguments. `LineSearch` and `KeepWeight` accept
`loss='squared', reg=1.0, learning_rate=1.0, fixed_prefix=None`.

`learning_rate` is finite and in `(0, 1]`. `FullyCorrective` and `LineSearch` interpolate from the previous weights
to their proposed weights using `previous + learning_rate * (proposed - previous)`. The rate applies once per
weight-calculation invocation; growing several rules can invoke the updater several times. A newly added rule has
previous coefficient zero for this interpolation. `KeepWeight` retains objective-derived coefficients and does not
apply damping. Custom updaters own their coefficient-step behaviour and require no new method or argument.

Initial fitting copies the updater and sets its learning rate to one, leaving the user's updater unchanged. Subsequent
growth, weight updates and replacement use its configured rate. `fixed_prefix` is `None` or a non-negative integer:
the first N coefficients remain fixed. Within the estimator, `None` fixes only the optional default rule. Explicit
zero optimises all coefficients. Direct updater calls interpret `None` as zero. Old serialized built-in updaters
receive compatible defaults when loaded.

### `loss='squared'`

`loss` accepts `'squared'`, `'logistic'`, `'poisson'`, the corresponding `SquaredLoss`, `LogisticLoss` or
`PoissonLoss` class, or an instance of one of these classes. A custom callable loss may also be supplied when it
provides the derivative and prediction operations required by the selected objective and updater.

Squared loss accepts finite real targets and returns raw scores from `predict`. Logistic loss requires every target
to be exactly `-1` or `1`; `predict` returns those labels and `predict_proba` returns columns for `-1` and `1` in that
order. Poisson loss requires finite non-negative targets and `predict` returns the exponentiated mean parameter.

### `reg=1.0`

`reg` is a non-negative finite floating-point L2 regularisation strength. It regularises learned weights in
`LineSearch` and `FullyCorrective`, and it contributes to objectives whose definitions include regularisation. The
MWG query objective ignores `reg`, although its weight update still uses it. The default OGB query objective is
unregularised unless `objective_params={'length_regularisation': True}` is supplied; its weight update still uses
the estimator's `reg` value.

### `search='exhaustive'`

`search` selects the algorithm that maximises the query objective. Accepted strings are `'exhaustive'`, `'greedy'`
and `'ogb'`. A compatible custom search class may also be supplied.

`'exhaustive'` uses exact branch-and-bound search over the core-query prefix tree. `'greedy'` repeatedly adds the
single proposition producing the greatest immediate objective improvement. `'ogb'` uses the orthogonal beam search
corresponding to Algorithm 2 and requires an objective with `prefix_values`, normally `OrthogonalBoostingObjective`.
The search-specific keys accepted through `search_params` are documented below.

### `include_default_rule=False`

When `include_default_rule` is true, the ensemble starts with a constant rule. Squared loss uses the target mean,
logistic loss uses the finite log odds of the positive class, and Poisson loss uses the logarithm of the target mean.
This term is a base rule, so it does not count towards `num_rules`. With the updater's default `fixed_prefix=None`,
its coefficient remains fixed during corrective optimisation. Explicit `FullyCorrective(fixed_prefix=0)` includes
it in coefficient optimisation. Its constant query is protected from structural replacement in either case.

### `include_linear_terms=False`

When `include_linear_terms` is true, numeric features contribute standardised linear terms based on their training
means and standard deviations. Categorical features contribute indicator terms corresponding to the fitted category
propositions. Constant or numerically degenerate numeric columns are skipped. Linear terms are base terms and do not
count towards `num_rules`; their initial coefficients and subsequent coefficients are handled by the configured
weight updater.

### `max_col_attr=10`

`max_col_attr` controls the numeric threshold search space. An integer limits the number of threshold propositions
generated for every numeric feature. `None` retains all distinct lower and upper threshold propositions. A mapping
may assign limits by feature name or zero-based column index; an omitted feature has no limit. When reduction is
needed, thresholds are chosen from quantiles, and both lower and upper comparisons are generated.

This parameter does not limit categorical cardinality. Use `max_categories` inside `search_params` for categorical
features. A `max_col_attr` key inside `search_params` overrides the constructor value.

### `search_params=None`

`search_params` is `None` or a dictionary containing feature-encoding options and options for the selected search
algorithm. Unrecognised keys may be accepted by `**kwargs` in a search implementation and should therefore be
avoided. The estimator defaults are `order='bestboundfirst'`, `apx=1.0`, `max_depth=None` and the constructor's
`max_col_attr` value. Only options relevant to the selected search have an effect.

The shared feature-encoding key `feature_names` accepts a sequence containing one name per input column. When it is
omitted for a NumPy array, names are generated as `x0`, `x1` and so forth. `load_csv` returns suitable names.

The `categorical` key accepts `None`, a sequence of feature names, a sequence of column indices, or a Boolean mask
containing one entry per feature. `None` enables inference: numeric NumPy arrays are numeric, string arrays are
categorical, and object arrays are inspected column by column. Explicit values are preferable for reproducible data
pipelines.

The `include_missing` key is a Boolean and defaults to true. It creates explicit `is missing` propositions for
features containing missing training values. The `min_category_count` key is a positive integer with default one;
categories occurring fewer times are omitted. The `max_categories` key is a positive integer or `None`; when a
limit is necessary, the most frequent categories are retained. The `without` key accepts feature names or indices
that should remain in the input array while being excluded from rule propositions.

The `n_jobs` key is retained as a compatibility alias for the top-level `n_jobs` constructor parameter. When both
are present, `search_params['n_jobs']` takes precedence for the fitted estimator. New code should normally use the
top-level parameter.

For `search='exhaustive'`, `order` accepts `'bestboundfirst'`, `'bestvaluefirst'`, `'breadthfirst'` or `'depthfirst'`.
`apx` must be positive; `1.0` performs exact pruning, while values between zero and one permit approximate early
termination. `max_depth` is a positive integer or `None` for unlimited conjunction depth. `parallel_min_candidates`
and `parallel_min_work` are positive integers controlling when candidate batches use Numba threads. Their defaults
are eight candidates and one million sample-candidate evaluations. CUDA is currently unavailable for exact search.

For `search='greedy'`, `backend` accepts `'auto'`, `'numba'` or `'cuda'`. The CPU modes use mandatory Numba kernels.
CUDA requires the optional CUDA installation and a supported NVIDIA GPU; supported candidate-statistics objectives
can then use it. `parallel_min_candidates` and `parallel_min_work` have the same meanings and defaults as in exact
search. Greedy conjunction depth is controlled by the objective when it offers a limit, such as OGB's
`maximum_query_length`; the generic `max_depth` search key does not truncate greedy search.

For `search='ogb'`, `beam_width` accepts a positive integer, `None` or positive infinity. The default is one;
`None` and infinity retain every candidate in the beam. `max_depth` is a positive integer or `None` for unlimited
depth. `parallel_min_families` is a positive integer controlling when independent numeric threshold families are
sent to the thread pool and defaults to four. This search is normally combined with `OrthogonalBoostingObjective`.

The legacy key `discretization` is accepted and ignored. New code should use `max_col_attr` and the categorical
options described above.

### `verbose=False`

`verbose` is a Boolean. When true, fitting prints the newest learned rule and the empirical mean loss after every
successful boosting iteration. The `fit` method also has a temporary `verbose` argument; output is enabled when
either value is true. Search algorithms may print additional diagnostics when verbosity reaches them.

### `basis_rtol=None`

`basis_rtol` is `None` or a positive floating-point relative tolerance used to identify zero and linearly dependent
rule indicators. `None` uses a scale-aware tolerance derived from double-precision machine epsilon. Increasing the
tolerance rejects nearly dependent rules more aggressively. Decreasing it admits smaller orthogonal residuals and
can reduce numerical stability. Rejected candidates are not divided by a near-zero norm.

### `objective_params=None`

`objective_params` is `None` or a dictionary passed to every newly constructed objective. The common
`hessian_floor` option is a finite positive float used to clamp extremely small Hessian entries before forming
second-order ratios. Its default is the smallest positive normal `float64` value.

For `GradientBoostingObjectiveXGB`, `length_regularisation` is a Boolean with default false. When true, the query
regularisation becomes `reg * (depth + 1)` instead of a depth-independent `reg`.

For `OrthogonalBoostingObjective`, `epsilon` is a positive floating-point stabiliser with default `1e-10`.
`maximum_query_length` is a non-negative integer or `None` and rejects queries deeper than that value.
`length_regularisation` is a Boolean with default false; when true, the OGB denominator receives
`reg * (depth + 1)`, while the default false setting leaves the OGB query objective unregularised.

MWG and GPE currently have no objective-specific options beyond common options such as `hessian_floor`.

### `max_components=10_000`

`max_components` is a positive integer safety cap on learned structural complexity. Adding a conjunction costs one
component for the rule and one component for every proposition in its query. Base default and linear terms do not
consume this budget. The check occurs between boosting iterations, so the final accepted rule may take the realised
total slightly beyond the nominal cap. `stopping_reason_` is set to `'component_limit'` when the next iteration is
prevented by this cap.

### `n_jobs=1`

`n_jobs` is a positive integer or `-1`. It controls Numba CPU workers for greedy and exact search and for sufficiently
large prediction batches. Orthogonal beam search uses the same value for its feature-family thread pool. `-1` uses
the maximum configured by Numba. Small workloads stay serial to avoid scheduling overhead.

`n_jobs` cannot exceed the Numba maximum established by `NUMBA_NUM_THREADS`. In Jupyter, change that environment
variable before importing `fastogb`, then restart the kernel. Omitting the environment variable lets Numba choose a
maximum from available CPUs. CUDA execution is configured through `search_params['backend']` and remains separate
from this CPU worker count.

## Training input

`fit(data, target)` expects `data` to be a two-dimensional NumPy array with shape `(n_samples, n_features)` and
`target` to contain one finite numeric value per row. A homogeneous numeric array should use a numeric NumPy dtype.
Mixed numeric and categorical columns may use `dtype=object`. Prediction arrays must retain the same column order
and compatible value types used for fitting.

`load_csv` returns `(data, target, feature_names, categorical)`. Pass the returned `feature_names` and `categorical`
through `search_params` so readable rule strings and column alignment are retained. Logistic labels should be mapped
to `-1.0` and `1.0` through `target_map`.

```python
from fastogb import FullyCorrective, GeneralRuleBoostingEstimator, LogisticLoss, OrthogonalBoostingObjective, load_csv

data, target, feature_names, categorical = load_csv('tic_tac_toe.csv', target_name='V10',
                                                  target_map={'negative': -1.0, 'positive': 1.0})
model = GeneralRuleBoostingEstimator(num_rules=10, objective_function=OrthogonalBoostingObjective,
                                    weight_update_method=FullyCorrective(), loss=LogisticLoss(), reg=1.0,
                                    search='greedy', n_jobs=4,
                                    search_params={'feature_names': feature_names, 'categorical': categorical},
                                    objective_params={'epsilon': 1e-4})
model.fit(data, target)
```

## Task-based examples

### Shared setup

The examples in this section use a small numerical regression dataset so they do not require pandas or external data files. `data` has shape `(12, 2)` and `target` has shape `(12,)`; the update batch uses the same feature matrix with changed responses. For real observations, use the arrays, feature names and categorical configuration returned by `load_csv` instead of the example's feature configuration, preserving column alignment in subsequent batches. Run this setup before each independent task unless the task explicitly continues the preceding example. It creates a fresh model with a default rule and a budget of two learned conjunctions. Search may stop below that budget when no admissible independent query remains. These examples have been reviewed against the implementation but have not been executed.

```python
import numpy as np
from fastogb import (Conjunction, FullyCorrective, GeneralRuleBoostingEstimator, GradientBoostingObjectiveXGB,
                     OrthogonalBoostingObjective, RuleEnsembleCollection, loss_function, no_replacement, smallest_absolute_weight)

data = np.column_stack((np.arange(12.0), np.tile([0.0, 1.0], 6)))
target = np.where(data[:, 0] < 6, -1.0, 1.0) + 0.5 * data[:, 1]
new_data = data.copy()
new_target = 1.3 * target + 0.2
settings = {'num_rules': 2, 'objective_function': OrthogonalBoostingObjective, 'loss': 'squared', 'reg': 0.1,
            'weight_update_method': FullyCorrective(learning_rate=0.1), 'include_default_rule': True, 'search': 'greedy',
            'n_jobs': 1, 'search_params': {'feature_names': ['measurement', 'enabled']}, 'objective_params': {'epsilon': 1e-4}}
model = GeneralRuleBoostingEstimator(**settings)
model.fit(data, target)
```

### Updating coefficients with a learning rate

`update_weights` recalculates coefficients while retaining every query, so it is the explicit operation for adapting a fixed model structure to new observations. The setup configures `FullyCorrective(learning_rate=0.1)`: initial fitting uses a copied updater with rate one, and this subsequent call uses rate 0.1. Each active coefficient moves from its current value towards the proposed optimum according to `previous + 0.1 * (proposed - previous)`. The default coefficient remains fixed with `fixed_prefix=None`; set `fixed_prefix=0` on the configured updater if it should also adapt. `LineSearch` changes only the last unprotected coefficient, while `KeepWeight` changes none and ignores its learning-rate setting. Change the configured `weight_update_method` for future operations; editing the fitted `weight_update_method_` copy does not configure the next call.

```python
previous_weights = np.asarray([rule.y for rule in model.rules_])
model.update_weights(new_data, new_target)
updated_weights = np.asarray([rule.y for rule in model.rules_])
print('Previous weights:', previous_weights)
print('Updated weights:', updated_weights)
print('Empirical loss before and after:', model.last_update_['loss_before'], model.last_update_['loss_after'])
```

To change the learning rate, solver or fixed prefix, replace the configured updater and then request another coefficient update. Built-in updaters do not implement nested `set_params`, so use the top-level assignment below rather than `weight_update_method__learning_rate`. The estimator supplies its own loss and regularisation to the copied updater at the next operation, avoiding duplicated configuration. Passing `fixed_prefix=0` allows the default coefficient and all other coefficients to participate in fully corrective optimisation; it does not make the default query eligible for replacement. The solver setting controls logistic and Poisson optimisation, while squared loss uses a linear-system solution. The interpolation rate applies on every coefficient calculation, so growing or replacing several rules can perform several damped steps within one call.

```python
model.set_params(weight_update_method=FullyCorrective(learning_rate=0.25, fixed_prefix=0, solver='BFGS'))
model.update_weights(new_data, new_target)
```

### Replacing rules and defining criteria

For a direct replacement request, call `replace_rules` with a selector such as `smallest_absolute_weight`. It selects one learned conjunction, removes it, and searches for replacement rules to restore at least the previous learned-rule count and the configured `num_rules`. It can select a rule even when the model is below its budget; in that case regrowth still targets the larger count. If search cannot restore that count, the call raises `RuntimeError` and restores the pre-replacement state. A selector passed to `fit` is consulted only when the budget has already been reached; initial fitting and undersized growth do not call it. The examples below are alternative entry points for the same built-in criterion, and a removed query can be rediscovered because replacement imposes no novelty requirement.

```python
model.replace_rules(new_data, new_target, replacement_selector=smallest_absolute_weight)
print('Removed indices:', model.last_update_['removed_indices'])
print(model.rules_)
```

```python
model.fit(new_data, new_target, replacement_selector=smallest_absolute_weight)
```

A custom selector must accept the ensemble snapshot and return a Python list of unique indices into that full snapshot. Base terms occupy the initial positions and cannot be selected, so indices must not be renumbered relative to a slice of learned rules. The snapshot includes `_base_rule_count_`, which the built-in selector also uses to identify the protected prefix. The following criterion selects up to two learned conjunctions by absolute weight, with ties retaining ensemble order. It accepts an optional `count`, while the estimator supplies only the ensemble argument and therefore uses the default count. To replace all eligible rules, the alternative uses `partial` to bind `count=None`, which retains the full sorted list. Base terms remain protected, and all-rule replacement still uses the adaptation learning rate. Treat the snapshot as read-only.

```python
def smallest_weights(ensemble, count=2):
    base_count = ensemble._base_rule_count_
    eligible = [i for i, rule in enumerate(ensemble) if i >= base_count and isinstance(rule.q, Conjunction)]
    return sorted(eligible, key=lambda i: abs(ensemble[i].y))[:count]

model.replace_rules(new_data, new_target, replacement_selector=smallest_weights)
```

```python
from functools import partial

model.replace_rules(new_data, new_target, replacement_selector=partial(smallest_weights, count=None))
```

Replacement can also be conditional, as shown by the weight threshold below. This callback reuses the built-in selector to retain its base-term protection, then decides whether the selected rule should actually be removed. An empty list prevents structural replacement and requests the ordinary coefficient update instead; `replace_rules` with its default `no_replacement` selector has that same meaning. Automatic `fit` also uses this default once its learned-rule budget has been reached. Consequently, keeping the rule structure does not necessarily keep its coefficients unchanged. Use `KeepWeight` when that is the intended coefficient policy, and distinguish that choice from returning an empty structural selection. Booleans, negative indices, duplicates, out-of-range indices, protected terms and `None` are invalid selector results.

```python
def replace_weakest_below_threshold(ensemble):
    indices = smallest_absolute_weight(ensemble)
    return indices if indices and abs(ensemble[indices[0]].y) < 0.05 else []

model.replace_rules(new_data, new_target, replacement_selector=replace_weakest_below_threshold)
```

```python
model.fit(new_data, new_target, replacement_selector=no_replacement)
```

### Selecting by empirical or regularised removal risk

A selector can capture data, targets and a loss function while retaining the required one-ensemble input interface. The example below measures the change in risk when each eligible rule is removed while all surviving coefficients stay unchanged, then selects the least damaging removal if its risk increase is within a chosen tolerance. Execute the `smallest_weights` definition above before this example; requesting `count=None` returns all eligible indices for evaluation. The captured targets are flattened to avoid broadcasting a single-column response array against one-dimensional scores. The loss receives raw additive scores, including for logistic and Poisson losses. This selection calculation describes deletion at the current coefficients; the subsequent replacement search and coefficient updates can produce a different final risk, and the estimator performs no separate final-risk acceptance test.

```python
criterion_loss = loss_function(model.loss)
include_regularisation = True
maximum_removal_increase = 0.01

def select_by_removal_risk(ensemble):
    eligible = smallest_weights(ensemble, count=None)
    if not eligible:
        return []
    response = np.asarray(new_target, dtype=np.float64).reshape(-1)
    scores = ensemble(new_data)
    weights = np.asarray([rule.y for rule in ensemble])
    penalty_strength = model.reg if include_regularisation else 0.0
    total_squared_weight = np.dot(weights, weights)
    baseline = float(np.mean(criterion_loss(response, scores))) + penalty_strength * total_squared_weight / (2 * len(response))
    changes = []
    for index in eligible:
        reduced_scores = scores - ensemble[index](new_data)
        reduced_risk = float(np.mean(criterion_loss(response, reduced_scores)))
        reduced_risk += penalty_strength * (total_squared_weight - weights[index] ** 2) / (2 * len(response))
        changes.append((reduced_risk - baseline, index))
    increase, index = min(changes)
    return [index] if increase <= maximum_removal_increase else []

model.replace_rules(new_data, new_target, replacement_selector=select_by_removal_risk)
```

Set `include_regularisation=False` to select using empirical mean loss alone, or leave it true to include the displayed L2 penalty. This example deliberately penalises every `rule.y` coefficient as part of the selector's criterion. Protected base terms remain present in both compared ensembles, so their unchanged penalty cancels; a selector intended to reproduce a particular updater's treatment of other fixed coefficients must use that updater's active prefix. The factor `1 / (2 * n_samples)` expresses the penalty on the same mean-loss scale as a summed loss plus `reg * squared_weight / 2`. No coefficient optimisation precedes this criterion. If recalculated coefficients should determine eligibility, the alternative below computes fully corrective proposals solely for selection and leaves the snapshot and fitted model unchanged until replacement begins.

```python
def select_by_recalculated_weight(ensemble):
    updater = FullyCorrective(loss=model.loss, reg=model.reg, fixed_prefix=model.weight_update_method_.fixed_prefix)
    response = np.asarray(new_target, dtype=np.float64).reshape(-1)
    proposed = updater.calc_weight(new_data, response, ensemble)
    eligible = smallest_weights(ensemble, count=None)
    return sorted(eligible, key=lambda i: abs(proposed[i]))[:1]

model.replace_rules(new_data, new_target, replacement_selector=select_by_recalculated_weight)
```

### Growing, resetting and configuring base terms

Increase `num_rules` and call ordinary `fit` to preserve the existing queries and search for additional learned rules. The automatic selector is ignored while that ensemble remains undersized, and coefficient calculations during growth honour the configured learning rate. Lowering the budget does not truncate an already larger ensemble. Replacement preserves at least its previous learned-rule count; obtaining a smaller model requires fresh fitting with the smaller budget. `fit_initial_rules` with its default operation discards previous queries, reconstructs the encoder and base terms, uses undamped initial fitting, and resets iteration and adaptation histories. These are separate tasks: run the relevant block after the shared setup rather than appending a reset to a replacement demonstration. Explicit legacy `has_origin_rules=False` is another reset entry point, while `True` retains grow-only continuation.

```python
model.set_params(num_rules=4)
model.fit(new_data, new_target)
print('Stopping reason:', model.stopping_reason_)
print(model.rules_)
```

```python
model.set_params(num_rules=1)
model.fit_initial_rules(new_data, new_target)
print(model.rules_)
```

Default and linear terms are configured before fresh fitting, because adaptation retains the already fitted base structure and numeric standardisation parameters. The first model below includes a default rule and permits its coefficient to adapt by selecting `fixed_prefix=0`; omitting the default rule requires `include_default_rule=False` in a fresh configuration. The second model adds standardised numeric linear terms while retaining the learned-rule budget. These base terms do not count towards `num_rules`, and their queries remain protected from replacement regardless of whether their coefficients are fixed. Fully corrective optimisation can update linear coefficients outside its fixed prefix. Numeric columns with no usable variation are skipped, and categorical linear terms use the encoder's category indicators. Changing these options on a fitted model should be followed by explicit fresh fitting when the base structure needs to change.

```python
default_model = GeneralRuleBoostingEstimator(**(settings | {'weight_update_method': FullyCorrective(learning_rate=0.1, fixed_prefix=0)}))
default_model.fit(data, target)
default_model.update_weights(new_data, new_target)

linear_model = GeneralRuleBoostingEstimator(**(settings | {'include_linear_terms': True}))
linear_model.fit(data, target)
print(linear_model.rules_)
```

### Choosing losses, objectives and search methods

The shared setup demonstrates squared-loss regression with an orthogonal objective, greedy search and fully corrective weights. These elements can be changed independently within their supported combinations. The classification example below constructs illustrative labels encoded as exactly `-1` and `1`, then exposes labels, raw scores and probabilities separately; real categorical target labels should be mapped during CSV loading. The Poisson example constructs non-negative counts and returns their predicted conditional means. Each new estimator performs fresh fitting, so its initial coefficient calculations ignore the configured adaptation rate. For these loss functions, risk calculations use `decision_function` scores rather than thresholded classification labels or exponentiated count predictions. The toy target transformations only illustrate the input domains and should be replaced by the actual response variables of a real application.

```python
labels = np.where(target > 0, 1.0, -1.0)
classifier = GeneralRuleBoostingEstimator(**(settings | {'loss': 'logistic'}))
classifier.fit(data, labels)
predicted_labels = classifier.predict(data)
raw_scores = classifier.decision_function(data)
probabilities = classifier.predict_proba(data)

counts = np.where(data[:, 0] < 6, 1.0, 4.0) + data[:, 1]
count_model = GeneralRuleBoostingEstimator(**(settings | {'loss': 'poisson'}))
count_model.fit(data, counts)
predicted_counts = count_model.predict(data)
```

For query selection, choose an objective class and a compatible search method, then fit a fresh estimator to compare their resulting structures. The following alternatives keep the shared data and encoding options: greedy OGB, exact branch-and-bound OGB, orthogonal beam search, and greedy XGB. Exact pruning uses `apx=1.0`; the example's `max_depth=3` restricts the exact search space. Beam search requires an objective providing ordered prefix values, so it is paired with `OrthogonalBoostingObjective`. XGB, MWG and GPE can use greedy or exact search. Changing the search or objective on a model already at its rule budget does not trigger query selection during default repeated `fit`; explicitly reset, grow or replace rules when those changes should affect structure. The dictionary merges preserve feature names while adding algorithm-specific options.

```python
greedy_model = GeneralRuleBoostingEstimator(**settings)
exact_params = settings['search_params'] | {'order': 'bestboundfirst', 'apx': 1.0, 'max_depth': 3}
exact_model = GeneralRuleBoostingEstimator(**(settings | {'search': 'exhaustive', 'search_params': exact_params}))
beam_params = settings['search_params'] | {'beam_width': 4, 'max_depth': 3}
beam_model = GeneralRuleBoostingEstimator(**(settings | {'search': 'ogb', 'search_params': beam_params}))
xgb_model = GeneralRuleBoostingEstimator(**(settings | {'objective_function': GradientBoostingObjectiveXGB, 'objective_params': {}}))
for candidate in (greedy_model, exact_model, beam_model, xgb_model):
    candidate.fit(data, target)
    print(candidate.search, candidate.stopping_reason_)
    print(candidate.rules_)
```

### Controlling feature use and parallel evaluation

To exclude a variable from learned rule conditions without changing the input schema, place its name or positional index in `search_params['without']`, as shown below for the second numerical feature. The array and its feature-name sequence still contain that column, so prediction and subsequent adaptation retain the same positional schema. Categorical columns can be specified through the same search dictionary and are encoded automatically as equality indicators; use an object array when combining numerical values with category strings. The current interface has no option requiring particular variables to appear in every conjunction. Exclusion controls candidate propositions; when numeric linear terms are enabled, they can still use excluded numeric columns. Remove a column from the original data before fresh fitting when it must be absent from every model component.

```python
filtered_params = settings['search_params'] | {'without': ['enabled']}
filtered_model = GeneralRuleBoostingEstimator(**(settings | {'search_params': filtered_params}))
filtered_model.fit(data, target)
```

```python
mixed_data = np.array([[18.0, 'cool'], [19.0, 'warm'], [20.0, None], [np.nan, 'cool'], [22.0, 'warm']], dtype=object)
mixed_target = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
mixed_params = {'feature_names': ['measurement', 'mode'], 'categorical': ['mode'], 'include_missing': True}
mixed_model = GeneralRuleBoostingEstimator(**(settings | {'search_params': mixed_params}))
mixed_model.fit(mixed_data, mixed_target)
print(mixed_model.rules_)
```

CPU parallelism is configured through `n_jobs`, independently of the choice of loss or coefficient learning rate. `n_jobs=-1` uses the configured Numba maximum, while explicit worker counts must remain within that limit. Large greedy and exact candidate batches use Numba workers, and orthogonal beam search can parallelise numerical threshold families. Prediction and package-native ensemble calls also select compiled parallel paths only when their workload thresholds are met; this small example may remain serial. `RuleEnsembleCollection` evaluates several models with columns corresponding to their list order, returning shape `(n_samples, n_models)`. Its fused numeric path can use CPU workers, while unsupported queries or non-numeric data may fall back to individual evaluation. The model list below continues the search-comparison example; run that example first.

```python
parallel_model = GeneralRuleBoostingEstimator(**(settings | {'n_jobs': -1}))
parallel_model.fit(data, target)
predictions = parallel_model.predict(new_data)
ensemble_scores = parallel_model.rules_(new_data)

models = [greedy_model, exact_model, beam_model, xgb_model]
evaluator = RuleEnsembleCollection(models, n_jobs=-1)
scores = evaluator.decision_function(new_data)
print('Score matrix shape:', scores.shape)
```

### Refreshing collections and inspecting adaptation

Collections cache the query descriptors and coefficients of their member ensembles. After fitting, reweighting or replacing rules in an existing member estimator, call `refresh` before evaluating the collection again; otherwise its fused path can still use the old coefficients or structures. If parallel training returns new estimator objects and the parent process installs them into the model list, call `set_members(models)` instead, after all assignments have completed. That method replaces the collection's copied member list and rebuilds its plan; calling `refresh` alone would still inspect the old estimator references. This distinction also applies to replacing list entries without multiprocessing. Each column of the score matrix remains associated with the corresponding member position, and transposing that matrix gives the action-by-observation layout used by action-selection code.

```python
models[0].update_weights(new_data, new_target)
evaluator.refresh()
scores = evaluator.decision_function(new_data)
action_values = scores.T
```

```python
replacement_model = GeneralRuleBoostingEstimator(**settings)
replacement_model.fit(new_data, new_target)
models[0] = replacement_model
evaluator.set_members(models)
scores = evaluator.decision_function(new_data)
```

For adaptation diagnostics, `last_update_` identifies the operation, records empirical loss before and after it, and reports removed indices for replacement. `update_history_` accumulates these records since the last initial fit, while `history_`, `loss_history_` and `time_` describe the latest training or adaptation call. Initial fitting clears adaptation history; a coefficient-only update records one iteration entry. The displayed losses are empirical means without the coefficient penalty, so they can differ from the regularised optimisation objective or a custom selector's criterion. Per-iteration elapsed times exclude preparation outside the timed loop, including initial encoding, and cold Numba compilation can affect observed timings. Use the code below after an adaptation example to inspect what actually happened rather than assuming that a replacement request necessarily produced a different query.

```python
print(model.last_update_['operation'], model.last_update_['removed_indices'], model.stopping_reason_)
print(model.last_update_['loss_before'], model.last_update_['loss_after'])
for update in model.update_history_:
    print(update['operation'], update['removed_indices'], update['loss_after'])
print(model.rules_)
```

## Public methods

### `fit(data, target, has_origin_rules=None, verbose=False, *, replacement_selector=no_replacement)`

`fit` learns the requested ensemble and returns the fitted `GeneralRuleBoostingEstimator`. Its complete signature is
`fit(data, target, has_origin_rules=None, verbose=False, *, replacement_selector=no_replacement)`.

With omitted `has_origin_rules`, an empty ensemble is fitted initially without damping. Existing ensembles below
`num_rules` retain their queries and grow towards the target. Once enough learned rules exist, the selector is
consulted. Its default empty list preserves every query and updates coefficients through the configured updater.
This intentionally changes repeated ordinary `fit` calls from reset-by-default to automatic adaptation. A callback
is not called during initial fitting or while an undersized ensemble grows. Existing ensembles above a newly lowered
`num_rules` are not silently truncated.

#### `data`

`data` is a two-dimensional NumPy array or NumPy-compatible array-like object with shape
`(n_samples, n_features)`. A numeric matrix may use an integer or floating-point dtype. A wholly categorical matrix
may use a string dtype, while mixed numeric and categorical columns should normally use `dtype=object`. Values must
be convertible to `float64` in columns treated as numeric. Categorical values must be hashable so that the fitted
encoder can construct category lookup tables. NumPy arrays are the recommended public input type. Objects exposing
`to_numpy` remain accepted for compatibility, although `fastogb` does not import or depend on pandas. SciPy sparse
matrices are not currently supported.

The row dimension gives the number of training observations and the column dimension gives the feature count.
Feature names supplied through `search_params['feature_names']` must contain exactly `n_features` entries. The
categorical mask or index collection supplied through `search_params['categorical']` must also refer to this same
positional column schema. Numeric `NaN` values and categorical `None` or `NaN` values are supported through missing
value propositions when `include_missing=True`.

#### `target`

`target` is a one-dimensional NumPy-compatible array with shape `(n_samples,)`; a single-column array with shape
`(n_samples, 1)` is also accepted and flattened. Values must be convertible to `float64`, every value must be finite,
and the number of values must equal the row count of `data`. Squared loss accepts any finite real-valued response.
Logistic loss accepts labels encoded exactly as `-1.0` and `1.0`. Poisson loss accepts finite values greater than or
equal to zero. A classification label mapping can be applied while reading a CSV with `load_csv(..., target_map=...)`.

#### `has_origin_rules=None`

`None` selects automatic dispatch based on the actual ensemble, including manually copied rules without a fitted
encoder. Explicit `False` preserves legacy fresh fitting; explicit `True` preserves legacy grow-only continuation,
which does not automatically reweight an ensemble already at its target. A fresh empty estimator is initialised
for either Boolean value. Existing positional calls retain their argument order. A custom selector cannot be combined
with an explicit Boolean legacy argument; use automatic dispatch for selection-controlled replacement.

#### `replacement_selector=no_replacement`

The keyword-only selector is a callable accepting a deep-copied `AdditiveRuleEnsemble` and returning a Python list
of unique zero-based integer indices into that same ensemble. NumPy integer scalars are accepted; Booleans,
negative indices, duplicates, out-of-range indices and protected default or linear terms are rejected. Return `[]`
to keep all rules; `None` is invalid. Selectors should treat the snapshot as read-only. No consolidation or coefficient
refit precedes selection. A callable object or closure can capture training data, a loss, regularisation, ages or
other information needed by a custom criterion.

`no_replacement` returns `[]`. `smallest_absolute_weight` selects one learned conjunction with minimum absolute
coefficient, breaking ties by original ensemble order. Estimator-produced ensembles and selector snapshots carry
base-term metadata so the selector skips protected terms. A constant learned query remains eligible when no default
rule is enabled. Standalone ensembles without that metadata treat all conjunctions as learned rules.
Both functions are available directly from `fastogb`.
The estimator imposes no separate loss-acceptance condition or `force_update` flag. A removed query may be found
again by search; there is no implicit novelty restriction beyond the existing surviving-rule extent checks.

#### `verbose=False`

The method-level `verbose` value is a Boolean that temporarily enables iteration output for this fit. Output is
enabled when either this value or the constructor's `verbose` value is true. Each completed iteration prints its new
rule and empirical mean loss. The setting does not alter fitted results.

`fit` raises `ValueError` for an invalid feature shape, inconsistent sample counts, non-finite targets, or targets
rejected by the selected loss. It raises `FloatingPointError` when an objective or weight updater produces a non-finite
coefficient. A successful call returns `self` and populates the fitted attributes described below.

```python
model.set_params(num_rules=20)
model.fit(data, target, has_origin_rules=True)
```

### `fit_initial_rules(data, target, verbose=False, *, operation='initial', target_rules=None)`

Starts again with a fresh ensemble, proposition encoder, default term and linear terms. `data` and `target` follow
the same types and shapes as `fit`, and the new feature schema may differ from the previous one. The updater's
configured learning rate is bypassed for the whole initial boosting procedure. Iteration and adaptation histories
are reset. Returns `self`. This is the explicit reset operation for new code.

This method contains the shared training implementation directly. Its keyword-only `operation` accepts `'initial'`,
`'grow'`, `'continue'`, `'weights'` or `'replace'`; `target_rules` accepts a non-negative integer or `None`, which uses
`num_rules`. The other fitting methods pass these options to reuse preparation and coefficient calculation without
duplicating the boosting loop. Ordinary explicit fresh fitting should leave both options at their defaults; use
`fit`, `update_weights` and `replace_rules` for automatic adaptation and validated replacement selection.

The supporting methods `prepare_base_terms`, `initialise_training_cache`, `initial_basis`, `initialise_base_weights`,
`find_independent_query`, `calculate_weights` and `build_prediction_plan` use ordinary names and perform separate
pieces of the training or prediction preparation. They assume the corresponding fitted metadata and staged state
already exist. Use the fitting and prediction entry points described here for normal model use. The previous
`_fit_rules`, `_new_objective`, `_append_training_mask`, `_check_fitted` and `_default_search_params` helpers have
been removed; their necessary work is performed directly at the relevant call sites.

### `update_weights(data, target, verbose=False)`

Keeps every query unchanged and invokes the selected weight updater on the supplied observations and responses.
`data` has shape `(n_samples, n_features_in_)`; `target` has shape `(n_samples,)` or `(n_samples, 1)` and obeys the
configured loss domain. Non-empty observations are required. Feature count, known feature names and categorical
roles must remain compatible. Existing linear-term locations and scales are retained. `FullyCorrective` updates
its active coefficients, `LineSearch` updates only the last unprotected coefficient, and `KeepWeight` changes none.
The updater owns its learning rate. Returns `self`; call on an empty unfitted estimator raises `RuntimeError`.

### `replace_rules(data, target, replacement_selector=no_replacement, verbose=False)`

Uses the same data and target contract as `update_weights`. Calls the selector once, validates its indices, removes
the selected learned rules and regrows the ensemble to at least its previous learned-rule count and the configured
`num_rules`. Empty selection delegates to `update_weights`. Default and linear terms remain structurally protected.
Removing every learned conjunction remains adaptation; it does not initialise new base terms or disable damping.
The updater's configured rate applies during coefficient calculations after adding replacement rules.

Selection does not trigger a preliminary coefficient refit or automatic duplicate-query consolidation. Search and
weight calculation reuse the ordinary training code, including empty-extent, duplicate-extent and independence
checks. If search cannot restore the requested count, `RuntimeError` is raised and the pre-replacement ensemble,
metadata, caches and histories are restored. Other fitting errors also restore the prior replacement state.
No risk-based rejection is imposed. Returns `self` on completion.

```python
from fastogb import smallest_absolute_weight

model.replace_rules(data, target, replacement_selector=smallest_absolute_weight)
```

Task examples for automatic replacement, custom selectors and learning-rate updates appear in [Task-based examples](#task-based-examples).

External `RuleEnsembleCollection` objects cache member descriptors and coefficients. Call `refresh()` after any
member update, or `set_members(models)` after replacing estimator objects returned by parallel training.

### `decision_function(data)`

`data` is a two-dimensional NumPy-compatible array with shape `(n_samples, n_features_in_)`. It may contain a
different number of rows from the training data, including an empty batch, but its column count, positional column
order, numeric-versus-categorical roles and compatible value types must match the fitted input schema. Feature names
on the prediction object are not used to reorder columns. Unseen categorical values match no fitted equality
proposition and therefore receive the ordinary false branch of relevant rules.

The method returns a one-dimensional `float64` NumPy array with shape `(n_samples,)`. Each element is the raw sum of
the default term, linear terms and rule contributions before applying a loss-specific link or threshold. It uses the
fitted encoding schema and compiled prediction plan, and large batches use up to `n_jobs_` Numba workers. It raises
`RuntimeError` before fitting and `ValueError` for an invalid feature shape or column count.

### `predict(data)`

`data` follows the same type, shape and positional schema as the `decision_function` input. The method returns a
one-dimensional NumPy array with shape `(n_samples,)`. Squared loss returns the raw real-valued `float64` scores.
Logistic loss returns label `-1` for a score below zero and label `1` for a score greater than or equal to zero.
Poisson loss returns a non-negative `float64` conditional mean after applying the exponential link. Invalid input
shape and fitted-state errors are inherited from `decision_function`.

### `predict_proba(data)`

`data` follows the same type, shape and positional schema as the `decision_function` input. With logistic loss, the
method returns a `float64` NumPy array with shape `(n_samples, 2)`. Column zero is the probability of label `-1`,
column one is the probability of label `1`, and each row sums to one apart from floating-point rounding. Squared and
Poisson loss raise `AttributeError` because they do not define class probabilities. Invalid input shape and
fitted-state errors are inherited from `decision_function`.

### `get_params(deep=True)`

`deep` is a Boolean and defaults to true. The method returns a `dict[str, object]` containing every constructor
parameter and its current value. It supports estimator inspection and parameter-search tools following the
scikit-learn naming convention. When `deep=True` and a configured child object exposes `get_params`, the result also
contains nested keys of the form `parent__child`. When `deep=False`, only the direct constructor parameters are
returned. This method does not require a fitted estimator and does not copy mutable parameter values.
Built-in weight updaters do not expose `get_params`, so their individual options do not appear as nested keys.

### `set_params(**params)`

`params` consists of keyword arguments whose names correspond to constructor parameters and whose values follow the constructor contracts documented above. Nested names of the form `parent__child` require the configured child object to implement `set_params`. Built-in weight updaters do not provide that method: use `model.set_params(weight_update_method=FullyCorrective(solver='BFGS', learning_rate=0.1))` to replace their configuration. The method returns `self` without changing the fitted ensemble immediately. Call `update_weights` to apply coefficient settings to existing queries. Objective and search settings affect subsequently added queries; an ordinary `fit` on a full ensemble updates only coefficients by default, so use growth, selected replacement or `fit_initial_rules` when new query selection is required. Configure default and linear terms before fresh fitting.

### `set_reg(reg)`

`reg` is a non-negative finite real number representing the new L2 regularisation strength. The method assigns the
value to the constructor-level `reg` parameter and returns `self`. It does not retroactively recalculate fitted
weights, loss histories or selected queries, so the estimator must be fitted again before the changed value affects
the model. `set_reg(value)` is equivalent to `set_params(reg=value)` and is retained as a concise compatibility
method.

## Fitted attributes

`rules_` is an `AdditiveRuleEnsemble` containing base terms followed by learned conjunction rules. `print(model.rules_)`
prints one readable weighted rule per line. Individual rules expose the query as `rule.q`, the satisfied weight as
`rule.y`, and the alternative weight as `rule.z`.

`history_` is a list containing a copied ensemble after every successful learned-rule iteration. `history` is an
alias. `loss_history_` contains the corresponding unregularised empirical mean loss. `time_` contains elapsed seconds
for each search-and-weight-update step, and `time` is an alias. Initial data loading, initial encoding and work before
the first iteration timer are outside these per-iteration values.

These iteration histories describe the latest call; a coefficient-only update records one entry. `last_update_`
records `operation` (`'initial'`, `'grow'`, `'continue'`, `'weights'` or `'replace'`), coefficient arrays before and
after the operation, empirical losses, elapsed time, removed indices and the stopping reason. Replacement adds
the removed query strings. `update_history_` accumulates successful adaptation records and is reset by initial
fitting. A rejected or failed operation leaves the existing fitted state intact.

`stopping_reason_` explains why fitting ended. `'maximum_rules'` means the requested learned-rule count was reached.
`'component_limit'` means the structural safety cap stopped further iterations. `'no_query'` means search returned no
candidate. `'empty_query_extent'` indicates an empty selected subgroup. `'duplicate_query_extent'` indicates an
already represented subgroup, and `'linearly_dependent_query_extent'` indicates dependence on the active rule basis.
`'weights_updated'` identifies a coefficient-only adaptation call.

`feature_names_in_` contains the fitted column names, and `n_features_in_` contains the input column count.
`encoder_` is the fitted `PropositionEncoder`, while `context_matrix_` is the Boolean training proposition matrix.
`training_rule_matrix_` contains base and rule outputs on the training rows. `orth_basis_` contains the final
orthonormal learned-rule basis. These arrays are useful for diagnostics and should generally be treated as read-only.

`weight_update_method_` is the fitted copy of the configured updater with the estimator loss and regularisation.
`search_params_` contains the resolved default and user-supplied search dictionary. `n_jobs_` is the effective worker
count used by the fitted model.

## Reading rules and iteration histories

```python
print(model.rules_)

for iteration, (ensemble, loss, elapsed) in enumerate(zip(model.history_, model.loss_history_, model.time_), start=1):
    print(f'Iteration {iteration}: loss={loss:.8g}, time={elapsed:.6f} s')
    print(ensemble)
```

Rule strings use the feature names supplied during fitting. Numeric propositions are rendered as comparisons such as
`age<=42.0`, categorical propositions as equalities such as `colour==blue`, and conjunctions join propositions with
`&`. The displayed coefficient is the rule's current fitted contribution when the conjunction is satisfied.



## Citation

Research using this estimator or its orthogonal gradient boosting implementation should cite:

```bibtex
@inproceedings{yang2024orthogonal,
  title={Orthogonal gradient boosting for simpler additive rule ensembles},
  author={Yang, Fan and Le Bodic, Pierre and Kamp, Michael and Boley, Mario},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1117--1125},
  year={2024},
  organization={PMLR}
}
```

See [the citation documentation](citation.md) and the repository-level `CITATION.cff` for the same preferred
citation in human-readable and machine-readable forms.
