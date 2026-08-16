# GeneralRuleBoostingEstimator interface

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

`GeneralRuleBoostingEstimator` is the single user-facing estimator for constructing additive rule ensembles. Its
behaviour is configured by composing a query-selection objective, a search method, a loss and a weight-update method.
Install the `fastogb` distribution from PyPI with `pip install fastogb`, then import its public API from `fastogb`.

## Constructor

```python
GeneralRuleBoostingEstimator(
    num_rules=3, objective_function=OrthogonalBoostingObjective, weight_update_method=FullyCorrective,
    loss='squared', reg=1.0, search='greedy', include_default_rule=False,
    include_linear_terms=False, max_col_attr=10, search_params=None, verbose=False, basis_rtol=None,
    objective_params=None, max_components=10_000, n_jobs=1)
```

### `num_rules=3`

`num_rules` is a non-negative integer giving the requested number of learned conjunction rules. The optional default
rule and optional linear terms are base terms and do not count towards this number. Training can finish with fewer
rules when no admissible independent query remains or when `max_components` is reached. The realised number of
rules can be inspected through `rules_`, `_base_rule_count_` and `stopping_reason_` after fitting.

### `objective_function=OrthogonalBoostingObjective`

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

### `weight_update_method=FullyCorrective`

`weight_update_method` accepts an instance of `KeepWeight`, `LineSearch`, `FullyCorrective` or a compatible custom
`WeightUpdateMethod`. `None` selects `KeepWeight()`. The estimator copies the updater for fitting and overwrites the
copy's `loss` and `reg` with the estimator settings, so users normally write `FullyCorrective()` without repeating
the loss or regularisation.

`KeepWeight()` retains the weight calculated when each rule is selected. `LineSearch()` reoptimises the newest rule
while holding earlier weights fixed. `FullyCorrective()` jointly reoptimises all active learned-rule weights after
every iteration. With squared loss, `FullyCorrective` solves a regularised linear system. With logistic or Poisson
loss, it uses a SciPy optimiser; its `solver` and `options` are configured on the updater instance.

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

### `search='greedy'`

`search` selects the algorithm that maximises the query objective. Accepted strings are `'exhaustive'`, `'greedy'`
and `'ogb'`. A compatible custom search class may also be supplied.

`'exhaustive'` uses exact branch-and-bound search over the core-query prefix tree. `'greedy'` repeatedly adds the
single proposition producing the greatest immediate objective improvement. `'ogb'` uses the orthogonal beam search
corresponding to Algorithm 2 and requires an objective with `prefix_values`, normally `OrthogonalBoostingObjective`.
The search-specific keys accepted through `search_params` are documented below.

### `include_default_rule=False`

When `include_default_rule` is true, the ensemble starts with a constant rule. Squared loss uses the target mean,
logistic loss uses the finite log odds of the positive class, and Poisson loss uses the logarithm of the target mean.
This term is a base rule, so it does not count towards `num_rules`. The default rule remains fixed when the updater
reoptimises learned rules.

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
from fastogb import (FullyCorrective, GeneralRuleBoostingEstimator, LogisticLoss,
                     OrthogonalBoostingObjective, load_csv)

data, target, feature_names, categorical = load_csv(
    'tic_tac_toe.csv', target_name='V10', target_map={'negative': -1.0, 'positive': 1.0})
model = GeneralRuleBoostingEstimator(
    num_rules=10, objective_function=OrthogonalBoostingObjective,
    weight_update_method=FullyCorrective(), loss=LogisticLoss(), reg=1.0, search='greedy', n_jobs=4,
    search_params={'feature_names': feature_names, 'categorical': categorical},
    objective_params={'epsilon': 1e-4})
model.fit(data, target)
```

## Public methods

### `fit(data, target, has_origin_rules=False, verbose=False)`

`fit` learns the requested ensemble and returns the fitted `GeneralRuleBoostingEstimator`. Its complete signature is
`fit(data, target, has_origin_rules=False, verbose=False)`.

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

#### `has_origin_rules=False`

`has_origin_rules` is a Boolean. Its default value, false, starts a new ensemble and discards terms from an earlier
fit. When it is true on an already fitted estimator, the existing base terms and learned rules are retained.
Continuing usually requires increasing `num_rules` first, because `num_rules` is the desired total learned-rule count
rather than the number of additional rules. The histories are reset and record only newly completed iterations from
the continuing fit. On a fresh estimator, true has the same effect as false because no fitted origin rules exist.

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

### `set_params(**params)`

`params` consists of keyword arguments whose names correspond to constructor parameters and whose values follow the
constructor contracts documented above. Nested names of the form `parent__child` are forwarded to a configured child
object exposing `set_params`; for example, `weight_update_method__solver='BFGS'` addresses a solver parameter on a
compatible updater. The method returns `self`. Changing configuration does not mutate the already learned rules,
weights, prediction plan or histories, so users must call `fit` to apply training-related changes.

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

`stopping_reason_` explains why fitting ended. `'maximum_rules'` means the requested learned-rule count was reached.
`'component_limit'` means the structural safety cap stopped further iterations. `'no_query'` means search returned no
candidate. `'empty_query_extent'` indicates an empty selected subgroup. `'duplicate_query_extent'` indicates an
already represented subgroup, and `'linearly_dependent_query_extent'` indicates dependence on the active rule basis.

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

for iteration, (ensemble, loss, elapsed) in enumerate(
        zip(model.history_, model.loss_history_, model.time_), start=1):
    print(f'Iteration {iteration}: loss={loss:.8g}, time={elapsed:.6f} s')
    print(ensemble)
```

Rule strings use the feature names supplied during fitting. Numeric propositions are rendered as comparisons such as
`age<=42.0`, categorical propositions as equalities such as `colour==blue`, and conjunctions join propositions with
`&`. The displayed coefficient is the rule's current fitted contribution when the conjunction is satisfied.
