from __future__ import annotations
from multimethodTypeTest import greet


class Foo:
    pass


class Bar:
    pass


def something_else():
    greet(Foo())
    greet(Bar())
