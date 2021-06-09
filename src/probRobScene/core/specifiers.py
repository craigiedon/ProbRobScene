"""Specifiers and associated objects."""
from typing import Any

from probRobScene.core.distributions import to_distribution
from probRobScene.core.lazy_eval import (DelayedArgument, toDelayedArgument, requiredProperties,
                                         needs_lazy_evaluation, LazilyEvaluable)
from probRobScene.core.utils import RuntimeParseError


## Specifiers themselves

class Specifier:
    """Specifier providing a value for a property given dependencies.

    Any optionally-specified properties are evaluated as attributes of the primary value.
    """

    def __init__(self, prop, value, deps=None, optionals=None):
        if deps is None:
            deps = set()

        if isinstance(value, LazilyEvaluable):
            deps |= value.required_properties()
        if optionals is None:
            optionals = {}
        self.property = prop
        self.value = toDelayedArgument(value)
        if prop in deps:
            raise RuntimeParseError(f'specifier for property {prop} depends on itself')
        self.requiredProperties = deps
        self.optionals = optionals

    def __str__(self):
        return f'<Specifier of {self.property}>'


## Support for property defaults

class PropertyDefault:
    """A default value, possibly with dependencies."""

    def __init__(self, requiredProperties, attributes, value):
        self.requiredProperties = requiredProperties
        self.value = value

    def resolve_for(self, prop):
        """Create a Specifier for a property from this default and any superclass defaults."""
        return Specifier(prop, DelayedArgument(self.requiredProperties, self.value))


def pd_for_value(value: Any) -> PropertyDefault:
    if isinstance(value, PropertyDefault):
        return value
    else:
        return PropertyDefault(set(), set(), lambda self: value)
