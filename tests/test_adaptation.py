import numpy as np
import pytest

from fastogb import (Constraint, FullyCorrective, GeneralRuleBoostingEstimator, KeepWeight, LinearTerm, LineSearch,
                     LogisticLoss, OrthogonalBoostingObjective, PoissonLoss, PropositionEncoder, SquaredLoss,
                     no_replacement, smallest_absolute_weight)


DATA = np.arange(8.0).reshape(-1, 1)
TARGET = np.array([-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0])


@pytest.mark.parametrize('loss,target', [(SquaredLoss(), TARGET), (LogisticLoss(), np.sign(TARGET)),
                                        (PoissonLoss(), TARGET + 3.0)])
def test_initial_fit_ignores_adaptation_rate_and_retains_the_configured_updater(loss, target):
    full = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), loss=loss, search='greedy',
                                       include_default_rule=True).fit(DATA, target)
    updater = FullyCorrective(learning_rate=0.1)
    damped = GeneralRuleBoostingEstimator(1, weight_update_method=updater, loss=loss, search='greedy',
                                         include_default_rule=True).fit(DATA, target)

    assert [str(rule.q) for rule in damped.rules_] == [str(rule.q) for rule in full.rules_]
    np.testing.assert_allclose([rule.y for rule in damped.rules_], [rule.y for rule in full.rules_])
    assert updater.learning_rate == 0.1
    assert updater.fixed_prefix is None
    assert damped.weight_update_method_.learning_rate == 1.0
    assert damped.weight_update_method_.fixed_prefix == 1


def test_automatic_fit_grows_retained_rules_before_calling_the_selector():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    first_query = str(model.rules_[0].q)
    model.num_rules = 2

    def unexpected_selector(ensemble):
        raise AssertionError('A selector must not run before the rule target is reached')

    model.fit(DATA, TARGET, replacement_selector=unexpected_selector)

    assert len(model.rules_) == 2
    assert str(model.rules_[0].q) == first_query
    assert model.last_update_['operation'] == 'grow'
    np.testing.assert_allclose(model._training_scores_, model.rules_(DATA))


def test_default_selector_performs_damped_coefficient_adaptation_on_fixed_queries():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(learning_rate=0.25), reg=0.2,
                                        search='greedy', include_default_rule=True).fit(DATA, TARGET)
    queries = [str(rule.q) for rule in model.rules_]
    previous = np.asarray([rule.y for rule in model.rules_])
    matrix = model.rules_.query_matrix(DATA)
    optimum = FullyCorrective(reg=0.2, fixed_prefix=1).calc_weight_from_matrix(TARGET * 2, model.rules_, matrix)

    model.fit(DATA, TARGET * 2)

    assert [str(rule.q) for rule in model.rules_] == queries
    np.testing.assert_allclose([rule.y for rule in model.rules_], previous + 0.25 * (optimum - previous))
    assert model.last_update_['operation'] == 'weights'
    assert model.last_update_['removed_indices'] == []
    assert model.stopping_reason_ == 'weights_updated'
    np.testing.assert_allclose(model.decision_function(DATA), model._training_scores_)


def test_legacy_reset_continuation_and_explicit_initial_fit_keep_their_roles():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    weights = np.asarray([rule.y for rule in model.rules_])
    model.fit(DATA, TARGET * 3, has_origin_rules=True)
    np.testing.assert_allclose([rule.y for rule in model.rules_], weights)

    reference = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET * 3)
    model.fit(DATA, TARGET * 3, has_origin_rules=False)
    np.testing.assert_allclose(model.predict(DATA), reference.predict(DATA))
    model.fit_initial_rules(DATA, TARGET)
    np.testing.assert_allclose([rule.y for rule in model.rules_], weights)
    assert model.update_history_ == []


def test_rule_only_estimator_can_adapt_without_discarding_copied_rules():
    original = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    restored = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy')
    restored.rules_ = original.rules_.copy()
    queries = [str(rule.q) for rule in restored.rules_]

    restored.fit(DATA, TARGET * 2)

    assert [str(rule.q) for rule in restored.rules_] == queries
    assert restored.last_update_['operation'] == 'weights'
    np.testing.assert_allclose(restored.predict(DATA), original.predict(DATA) * 2)


def test_full_learned_rule_replacement_remains_adaptation_and_preserves_the_default_term():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(learning_rate=0.25), search='greedy',
                                        include_default_rule=True).fit(DATA, TARGET)
    previous = np.asarray([rule.y for rule in model.rules_])

    model.fit(DATA, TARGET, replacement_selector=smallest_absolute_weight)

    assert len(model.rules_) == 2
    assert model.rules_[0].y == previous[0]
    np.testing.assert_allclose(model.rules_[1].y, previous[1] * 0.25)
    assert model.weight_update_method_.learning_rate == 0.25
    assert model.last_update_['operation'] == 'replace'
    assert model.last_update_['removed_indices'] == [1]
    np.testing.assert_allclose(model._training_scores_, model.rules_(DATA))


@pytest.mark.parametrize('indices,error', [(None, TypeError), ([True], TypeError), ([1.5], TypeError),
                                          ([-1], ValueError), ([99], ValueError), ([1, 1], ValueError), ([0], ValueError)])
def test_invalid_replacement_indices_do_not_change_the_estimator(indices, error):
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy',
                                        include_default_rule=True).fit(DATA, TARGET)
    ensemble, encoder = model.rules_, model.encoder_
    predictions = model.predict(DATA)

    with pytest.raises(error):
        model.replace_rules(DATA, TARGET, replacement_selector=lambda rules: indices)

    assert model.rules_ is ensemble
    assert model.encoder_ is encoder
    np.testing.assert_array_equal(model.predict(DATA), predictions)


def test_failed_replacement_restores_rules_metadata_and_iteration_histories():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy',
                                        include_default_rule=True).fit(DATA, TARGET)
    ensemble, encoder, history = model.rules_, model.encoder_, model.history_
    predictions = model.predict(DATA)

    with pytest.raises(RuntimeError, match='could not restore'):
        model.replace_rules(np.ones_like(DATA), TARGET, smallest_absolute_weight)

    assert model.rules_ is ensemble
    assert model.encoder_ is encoder
    assert model.history_ is history
    np.testing.assert_array_equal(model.predict(DATA), predictions)


def test_multiple_replacement_indices_refer_to_the_original_ensemble():
    model = GeneralRuleBoostingEstimator(2, weight_update_method=FullyCorrective(), search='greedy',
                                        include_default_rule=True).fit(DATA, TARGET)
    previous = [str(rule.q) for rule in model.rules_]

    model.replace_rules(DATA, TARGET, replacement_selector=lambda rules: [2, 1])

    assert len(model.rules_) == 3
    assert model.last_update_['removed_indices'] == [2, 1]
    assert model.last_update_['removed_rules'] == [previous[2], previous[1]]
    np.testing.assert_allclose(model._training_scores_, model.rules_(DATA))


def test_linear_base_terms_are_protected_from_replacement():
    model = GeneralRuleBoostingEstimator(0, weight_update_method=FullyCorrective(), include_default_rule=True,
                                        include_linear_terms=True).fit(DATA, TARGET)
    assert isinstance(model.rules_[1].q, LinearTerm)

    with pytest.raises(ValueError, match='linear terms'):
        model.replace_rules(DATA, TARGET, replacement_selector=lambda rules: [1])


def test_selector_receives_a_snapshot_and_is_not_preceded_by_coefficient_recalculation():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=KeepWeight(), search='greedy').fit(DATA, TARGET)
    previous = np.asarray([rule.y for rule in model.rules_])
    queries = [str(rule.q) for rule in model.rules_]

    def selector(ensemble):
        np.testing.assert_array_equal([rule.y for rule in ensemble], previous)
        ensemble[0].y = 999.0
        ensemble[0].q.props.clear()
        return []

    model.fit(DATA, TARGET * 5, replacement_selector=selector)

    np.testing.assert_array_equal([rule.y for rule in model.rules_], previous)
    assert [str(rule.q) for rule in model.rules_] == queries
    assert len(model.rules_[0].q)


def test_explicit_zero_prefix_allows_the_default_coefficient_to_adapt():
    fixed = GeneralRuleBoostingEstimator(0, weight_update_method=FullyCorrective(), reg=0,
                                        include_default_rule=True).fit(DATA, TARGET)
    free = GeneralRuleBoostingEstimator(0, weight_update_method=FullyCorrective(fixed_prefix=0), reg=0,
                                       include_default_rule=True).fit(DATA, TARGET)

    fixed.fit(DATA, TARGET + 5)
    free.fit(DATA, TARGET + 5)

    assert fixed.rules_[0].y == np.mean(TARGET)
    np.testing.assert_allclose(free.rules_[0].y, np.mean(TARGET + 5))
    assert no_replacement(free.rules_) == []
    assert smallest_absolute_weight(free.rules_) == []


def test_constant_learned_query_is_eligible_when_no_default_rule_is_enabled():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, np.ones(8))

    assert len(model.rules_) == 1
    assert not len(model.rules_[0].q)
    assert smallest_absolute_weight(model.rules_) == [0]


@pytest.mark.parametrize('updater', [FullyCorrective, LineSearch])
def test_direct_builtin_updaters_own_the_coefficient_learning_rate(updater):
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    matrix = model.rules_.query_matrix(DATA)
    previous = np.asarray([rule.y for rule in model.rules_])
    optimum = updater(reg=model.reg).calc_weight_from_matrix(TARGET * 2, model.rules_, matrix)
    actual = updater(reg=model.reg, learning_rate=0.2).calc_weight_from_matrix(TARGET * 2, model.rules_, matrix)

    np.testing.assert_allclose(actual, previous + 0.2 * (optimum - previous))


def test_old_updater_state_acquires_compatible_defaults():
    updater = FullyCorrective.__new__(FullyCorrective)
    updater.__setstate__({'loss': 'squared', 'reg': 1.0, 'fixed_prefix': 0, 'solver': 'Newton-CG', 'options': None})

    assert updater.learning_rate == 1.0
    assert updater.fixed_prefix is None


def test_custom_updater_without_new_parameters_remains_usable():
    class LegacyUpdater:
        def calc_weight(self, data, target, rules):
            return np.asarray([rule.y + 0.5 for rule in rules])

    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    previous = np.asarray([rule.y for rule in model.rules_])
    model.weight_update_method = LegacyUpdater()
    model.update_weights(DATA, TARGET)

    np.testing.assert_allclose([rule.y for rule in model.rules_], previous + 0.5)


def test_schema_mismatch_leaves_an_existing_model_intact():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    ensemble, encoder = model.rules_, model.encoder_

    with pytest.raises(ValueError, match='columns'):
        model.fit(np.column_stack((DATA, DATA)), TARGET)

    assert model.rules_ is ensemble
    assert model.encoder_ is encoder


def test_unnamed_update_arrays_retain_existing_feature_names():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy',
                                        search_params={'feature_names': ['temperature']}).fit(DATA, TARGET)
    model.search_params = None

    model.fit(DATA, TARGET * 2)

    assert list(model.feature_names_in_) == ['temperature']
    assert 'temperature' in str(model.rules_[0].q)


def test_invalid_custom_coefficient_update_does_not_commit_partial_state():
    class InvalidUpdater:
        def calc_weight(self, data, target, rules):
            return np.full(len(rules), np.nan)

    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(), search='greedy').fit(DATA, TARGET)
    ensemble, encoder = model.rules_, model.encoder_
    predictions = model.predict(DATA)
    model.weight_update_method = InvalidUpdater()

    with pytest.raises(FloatingPointError, match='invalid coefficients'):
        model.update_weights(DATA + 1, TARGET)

    assert model.rules_ is ensemble
    assert model.encoder_ is encoder
    np.testing.assert_array_equal(model.predict(DATA), predictions)


@pytest.mark.parametrize('search', ['greedy', 'exhaustive', 'ogb'])
def test_initial_rules_contains_training_and_supports_all_search_entry_points(search):
    model = GeneralRuleBoostingEstimator(1, objective_function=OrthogonalBoostingObjective,
                                        weight_update_method=FullyCorrective(), include_default_rule=True, search=search)
    result = model.fit_initial_rules(DATA, TARGET)

    assert result is model
    assert not hasattr(model, '_fit_rules')
    assert len(model.rules_) == 2
    assert model.last_update_['operation'] == 'initial'
    assert model.update_history_ == []
    np.testing.assert_allclose(model.decision_function(DATA), model.rules_(DATA))


def test_shared_training_operation_preserves_weight_adaptation():
    model = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(learning_rate=0.25), search='greedy')
    reference = GeneralRuleBoostingEstimator(1, weight_update_method=FullyCorrective(learning_rate=0.25), search='greedy')
    model.fit_initial_rules(DATA, TARGET)
    reference.fit_initial_rules(DATA, TARGET)
    queries = [str(rule.q) for rule in model.rules_]

    model.fit_initial_rules(DATA, TARGET * 2, operation='weights')
    reference.update_weights(DATA, TARGET * 2)

    assert [str(rule.q) for rule in model.rules_] == queries
    assert model.last_update_['operation'] == 'weights'
    np.testing.assert_allclose(model.predict(DATA), reference.predict(DATA))


@pytest.mark.parametrize('options', [{'operation': 'unknown'}, {'target_rules': -1}, {'target_rules': True}, {'target_rules': 1.5}])
def test_shared_training_options_are_validated_before_changing_state(options):
    model = GeneralRuleBoostingEstimator()
    ensemble = model.rules_

    with pytest.raises(ValueError):
        model.fit_initial_rules(DATA, TARGET, **options)

    assert model.rules_ is ensemble
    assert not hasattr(model, 'encoder_')


@pytest.mark.parametrize('method,operation,expected', [('less_equals', '<=', [True, True, False]),
    ('less', '<', [True, False, False]), ('greater_equals', '>=', [False, True, True]), ('greater', '>', [False, False, True]),
    ('equals', '==', [False, True, False]), ('not_equals', '!=', [True, False, True])])
def test_comparison_factories_bind_to_subclasses_without_forwarding_methods(method, operation, expected):
    class CustomConstraint(Constraint):
        pass

    condition = getattr(CustomConstraint, method)(1.0)

    assert isinstance(condition, CustomConstraint)
    assert condition.op == operation
    assert format(condition, 'temperature') == f'temperature{operation}1.0'
    np.testing.assert_array_equal(condition(np.array([0.0, 1.0, 2.0])), expected)


def test_encoder_and_query_conditions_share_missing_value_handling():
    data = np.array([[None], ['a'], [np.nan], ['b']], dtype=object)
    encoder = PropositionEncoder(feature_names=['category'])
    context = encoder.fit_transform(data)

    assert encoder.categorical_indices_ == {0}
    for index, proposition in enumerate(encoder.propositions_):
        np.testing.assert_array_equal(context[:, index], proposition(data))
    np.testing.assert_array_equal(Constraint.missing()(data[:, 0]), [True, False, True, False])


@pytest.mark.parametrize('loss,target', [(SquaredLoss(), TARGET), (LogisticLoss(), np.sign(TARGET)), (PoissonLoss(), TARGET + 3.0)])
def test_consolidated_loss_methods_preserve_derivatives_and_keyword_prediction_calls(loss, target):
    scores = TARGET / 2
    gradient, hessian = loss.derivatives(target=target, scores=scores)
    predictions = loss.predictions(scores=scores)

    np.testing.assert_allclose(gradient, loss.g(target, scores))
    np.testing.assert_allclose(hessian, loss.h(target, scores))
    assert predictions.shape == scores.shape
    if isinstance(loss, PoissonLoss):
        np.testing.assert_allclose(predictions, hessian)
