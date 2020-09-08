"""Specifiers and associated objects."""

from scenic3d.core.distributions import to_distribution
from scenic3d.core.lazy_eval import (DelayedArgument, toDelayedArgument, requiredProperties,
                                     needs_lazy_evaluation)
from scenic3d.core.utils import RuntimeParseError


## Specifiers themselves

class Specifier:
    """Specifier providing a value for a property given dependencies.

    Any optionally-specified properties are evaluated as attributes of the primary value.
    """

    def __init__(self, prop, value, deps=None, optionals=None):
        if deps is None:
            deps = set()
        deps |= requiredProperties(value)
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

        def enabled(thing, default):
            if thing in attributes:
                attributes.remove(thing)
                return True
            else:
                return default

        self.isAdditive = enabled('additive', False)
        for attr in attributes:
            raise RuntimeParseError(f'unknown property attribute "{attr}"')

    @staticmethod
    def forValue(value):
        if isinstance(value, PropertyDefault):
            return value
        else:
            return PropertyDefault(set(), set(), lambda self: value)

    def resolveFor(self, prop, overriddenDefs):
        """Create a Specifier for a property from this default and any superclass defaults."""
        if self.isAdditive:
            allReqs = self.requiredProperties
            for other in overriddenDefs:
                allReqs |= other.requiredProperties

            def concatenator(context):
                allVals = [self.value(context)]
                for other in overriddenDefs:
                    allVals.append(other.value(context))
                return tuple(allVals)

            val = DelayedArgument(allReqs, concatenator)
        else:
            val = DelayedArgument(self.requiredProperties, self.value)
        return Specifier(prop, val)
