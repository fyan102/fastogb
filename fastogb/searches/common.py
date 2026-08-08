"""Shared fastogb search introspection utilities."""

import inspect


def accepts_depth(function):
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [parameter for parameter in parameters
                  if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}]
    return len(positional) >= 2
