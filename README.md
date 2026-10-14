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

Repeated `fit` calls now preserve existing rules automatically. An undersized ensemble grows towards `num_rules`;
an ensemble already at its target updates only its coefficients by default. `fit_initial_rules` explicitly starts
again. Explicit legacy `has_origin_rules=False` still resets the model, and `True` retains its grow-only meaning.

The weight updater owns the adaptation learning rate. Initial fitting always uses undamped coefficient calculations;
`FullyCorrective(learning_rate=0.1)` moves towards the joint coefficient optimum during subsequent updates. Its
`fixed_prefix=None` default preserves the fixed default rule; explicit `fixed_prefix=0` includes that coefficient in
optimisation. Replacement is opt-in through a function accepting an ensemble and returning rule indices:

```python
from fastogb import smallest_absolute_weight

model.update_weights(data, target)
model.fit(data, target, replacement_selector=smallest_absolute_weight)
model.fit_initial_rules(data, target)
```

The default `no_replacement` function returns `[]`. Custom selectors own their selection criteria, including any
loss-based calculations, and the estimator adds no separate loss-acceptance filter. Default and linear terms are
protected from replacement. Refresh external `RuleEnsembleCollection` objects after models change, using
`set_members` when parallel training returns replacement estimator objects.


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
