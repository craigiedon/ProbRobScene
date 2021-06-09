from __future__ import annotations
from multimethod import multimethod

from mmTypes import Foo, Bar


@multimethod
def greet(x: Foo):
    print("Foo")


@multimethod
def greet(x: Bar):
    print("Bar")


print("All is well")
