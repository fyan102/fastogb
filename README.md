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

Package maintainers can follow the [release guide][release-guide] for TestPyPI and production PyPI uploads.

[release-guide]: https://github.com/fyan102/fastogb/blob/main/docs/releasing.md

```python
from fastogb import (FullyCorrective, GeneralRuleBoostingEstimator, OrthogonalBoostingObjective,
                     load_csv)

data, target, feature_names, categorical = load_csv(
    'data.csv', target_name='target', target_map={'negative': -1.0, 'positive': 1.0})
model = GeneralRuleBoostingEstimator(
    num_rules=10, objective_function=OrthogonalBoostingObjective,
    weight_update_method=FullyCorrective(), loss='logistic', search='greedy', n_jobs=4,
    search_params={'feature_names': feature_names, 'categorical': categorical},
    objective_params={'epsilon': 1e-4})
model.fit(data, target)

print(model.rules_)
predictions = model.predict(data)
probabilities = model.predict_proba(data)
```
