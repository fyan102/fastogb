# fastogb

`fastogb` is a NumPy-first package for constructing additive rule ensembles. It provides configurable
query-selection objectives, search algorithms, losses and weight-update methods, with mandatory Numba CPU
acceleration and optional CUDA acceleration for supported NVIDIA systems.

Install the current release from PyPI with:

```bash
pip install fastogb
```

The main public estimator is `GeneralRuleBoostingEstimator`. Its complete constructor, nested configuration,
methods and fitted attributes are described in the
[GeneralRuleBoostingEstimator interface][estimator-interface].

[estimator-interface]: https://github.com/fyan102/fastogb/blob/main/docs/general_rule_boosting_estimator.md

```python
from fastogb import FullyCorrective, GeneralRuleBoostingEstimator, OrthogonalBoostingObjective, load_csv

data, target, feature_names, categorical = load_csv('data.csv', target_name='target',
                                                  target_map={'negative': -1.0, 'positive': 1.0})
model = GeneralRuleBoostingEstimator(num_rules=10, objective_function=OrthogonalBoostingObjective,
                                    weight_update_method=FullyCorrective(), loss='logistic', search='greedy', n_jobs=4,
                                    search_params={'feature_names': feature_names, 'categorical': categorical},
                                    objective_params={'epsilon': 1e-4})
model.fit(data, target)

print(model.rules_)
predictions = model.predict(data)
probabilities = model.predict_proba(data)
```

## Adapting a trained ensemble

Repeated `fit` calls preserve existing rules automatically: an undersized ensemble grows towards `num_rules`, while an ensemble already at its target updates its coefficients by default. To adapt coefficients without changing queries, configure a weight updater and call `update_weights` with the new observations and targets. `FullyCorrective(learning_rate=0.1)` moves ten per cent of the way towards its proposed coefficient optimum on each update; initial fitting always uses an undamped step. The new arrays must preserve the original feature names, column order and numeric or categorical roles. The following example continues the logistic model above and reads a second CSV with the same schema. The configured updater receives the estimator's loss and regularisation automatically.

```python
model.set_params(weight_update_method=FullyCorrective(learning_rate=0.1))
new_data, new_target, new_feature_names, _ = load_csv('new_data.csv', target_name='target',
                                                    target_map={'negative': -1.0, 'positive': 1.0})
if tuple(new_feature_names) != tuple(feature_names):
    raise ValueError('The update CSV must preserve feature names and column order')
model.update_weights(new_data, new_target)
print(model.rules_)
```

Rule replacement is requested explicitly through `replace_rules` or through a selector passed to automatic `fit`. The built-in `smallest_absolute_weight` selector chooses one eligible learned conjunction with the smallest absolute coefficient, leaving default and linear terms structurally protected. A selector receives a deep-copied ensemble and returns a list of unique zero-based indices into the full ensemble. No preliminary coefficient refit is performed by replacement, and the estimator imposes no additional loss-acceptance condition. The direct call below requests selection immediately; a selector supplied to `fit` is consulted only after the ensemble has reached its requested learned-rule count. The configured learning rate applies to coefficient calculations during replacement, with each newly added rule starting its interpolation from coefficient zero.

```python
from fastogb import smallest_absolute_weight

model.replace_rules(new_data, new_target, replacement_selector=smallest_absolute_weight)
print(model.last_update_['removed_indices'])
print(model.rules_)
```

Custom criteria can impose an additional condition on the eligible rule, select several rules, or evaluate removal risk using captured data and a loss function. The alternative below replaces the weakest rule only when its absolute coefficient is below a chosen threshold. Returning `[]` keeps all queries and still requests a coefficient update through the configured updater; `None` is invalid. To discard the existing structure deliberately, use the separate `model.fit_initial_rules(new_data, new_target)` reset operation. Explicit legacy `has_origin_rules=False` also resets the model, while `True` retains grow-only continuation. The [task-based examples][estimator-tasks] cover multiple-rule and loss-based criteria, model growth, loss and search choices, linear terms, feature exclusion, diagnostics and collection refresh.

```python
def replace_weakest_rule(ensemble):
    indices = smallest_absolute_weight(ensemble)
    return indices if indices and abs(ensemble[indices[0]].y) < 0.05 else []

model.replace_rules(new_data, new_target, replacement_selector=replace_weakest_rule)
```

[estimator-tasks]: https://github.com/fyan102/fastogb/blob/main/docs/general_rule_boosting_estimator.md#task-based-examples

## Citation

If you use `fastogb` or its implementation of orthogonal gradient boosting in academic work, please cite:

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

The citation is also available in [the citation documentation](docs/citation.md) and `CITATION.cff`.
