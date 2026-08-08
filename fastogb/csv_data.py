"""CSV loading without pandas."""

import csv

import numpy as np


def _parse_feature(values):
    values = [value.strip() for value in values]
    try:
        return [np.nan if value == '' else float(value) for value in values], False
    except ValueError:
        return [None if value == '' else value for value in values], True


def load_csv(path, target_name=None, drop=(), target_map=None):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        field_names = tuple(reader.fieldnames or ())
        records = list(reader)

    if not field_names or len(field_names) != len(set(field_names)):
        raise ValueError('The CSV must have a non-empty header containing unique column names')
    target_name = field_names[-1] if target_name is None else target_name
    if target_name not in field_names:
        raise ValueError(f'Unknown target column {target_name!r}')
    if not records:
        raise ValueError('The CSV contains no observations')

    excluded = set(drop) | {target_name}
    feature_names = tuple(name for name in field_names if name not in excluded)
    X = np.empty((len(records), len(feature_names)), dtype=object)
    categorical = []

    for index, name in enumerate(feature_names):
        column, is_categorical = _parse_feature([record[name] for record in records])
        X[:, index] = column
        if is_categorical:
            categorical.append(name)

    raw_target = [record[target_name].strip() for record in records]
    if any(value == '' for value in raw_target):
        raise ValueError('The target column contains missing values')
    if target_map is None:
        y = np.asarray(raw_target, dtype=np.float64)
    else:
        y = np.asarray([target_map[value] for value in raw_target], dtype=np.float64)

    return X, y, feature_names, tuple(categorical)
