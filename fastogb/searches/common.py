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


def accepts_keyword(function, name):
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    if parameter is not None and parameter.kind != parameter.POSITIONAL_ONLY:
        return True
    return any(item.kind == item.VAR_KEYWORD for item in parameters.values())
