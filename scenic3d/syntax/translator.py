"""Translator turning Scenic programs into Scenario objects.

The top-level interface to Scenic is provided by two functions:

* `scenarioFromString` -- compile a string of Scenic code;
* `scenarioFromFile` -- compile a Scenic file.

These output a `Scenario` object, from which scenes can be generated.
See the documentation for `Scenario` for details.

When imported, this module hooks the Python import system so that Scenic
modules can be imported using the ``import`` statement. This is primarily for the
translator's own use, but you could import Scenic modules from Python to
inspect them. Because Scenic uses Python's import system, the latter's rules
for finding modules apply, including the handling of packages.

Scenic is compiled in two main steps: translating the code into Python, and
executing the resulting Python module to generate a Scenario object encoding
the objects, distributions, etc. in the scenario. For details, see the function
`compileStream` below.
"""

import ast
import builtins
import importlib
import importlib.abc
import importlib.util
import inspect
import io
import itertools
import os
import sys
import time
import tokenize
import traceback
import types
from ast import Load, Store, Name, Call, Tuple, BinOp, MatMult, BitAnd, BitOr, BitXor, LShift
from ast import RShift, Starred, Lambda, AnnAssign, Set, Str, Num, Subscript, Index
from ast import parse, NodeVisitor, NodeTransformer, copy_location, fix_missing_locations
from collections import namedtuple
from contextlib import contextmanager
from tokenize import INDENT, DEDENT, STRING
from tokenize import LEFTSHIFT, RIGHTSHIFT, VBAR, AMPER, TILDE, CIRCUMFLEX, STAR
from tokenize import LEFTSHIFTEQUAL, RIGHTSHIFTEQUAL, VBAREQUAL, AMPEREQUAL, CIRCUMFLEXEQUAL
from tokenize import LPAR, RPAR, LSQB, RSQB, COMMA, DOUBLESLASH, DOUBLESLASHEQUAL
from tokenize import NAME, NL, NEWLINE, ENDMARKER, NUMBER, COLON, COMMENT, ENCODING

import scenic3d.core.pruning as pruning
import scenic3d.syntax.relations as relations
import scenic3d.syntax.veneer as veneer
from scenic3d.core.distributions import Samplable, needs_sampling
from scenic3d.core.lazy_eval import needs_lazy_evaluation
from scenic3d.core.object_types import Constructible
from scenic3d.core.regions import Region
from scenic3d.core.scenarios import Scenario
from scenic3d.core.utils import ParseError, RuntimeParseError, InvalidScenarioError


### THE TOP LEVEL: compiling a Scenic program

def scenarioFromString(string, filename='<string>', cacheImports=False):
    """Compile a string of Scenic code into a `Scenario`.

    The optional **filename** is used for error messages."""
    stream = io.BytesIO(string.encode())
    return scenarioFromStream(stream, filename=filename, cacheImports=cacheImports)


def scenarioFromFile(path, cacheImports=False):
    """Compile a Scenic file into a `Scenario`.

    Args:
        path (str): path to a Scenic file
        cacheImports (bool): Whether to cache any imported Scenic modules.
          The default behavior is to not do this, so that subsequent attempts
          to import such modules will cause them to be recompiled. If it is
          safe to cache Scenic modules across multiple compilations, set this
          argument to True. Then importing a Scenic module will have the same
          behavior as importing a Python module.

    Returns:
        A `Scenario` object representing the Scenic scenario.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fullpath = os.path.realpath(path)
    head, extension = os.path.splitext(fullpath)
    if not extension or extension[1:] not in scenicExtensions:
        ok = ', '.join(scenicExtensions)
        err = f'Scenic scenario does not have valid extension ({ok})'
        raise RuntimeError(err)
    directory, name = os.path.split(head)

    with open(path, 'rb') as stream:
        return scenarioFromStream(stream, filename=fullpath, path=path,
                                  cacheImports=cacheImports)


def scenarioFromStream(stream, filename='<stream>', path=None, cacheImports=False):
    """Compile a stream of Scenic code into a `Scenario`."""
    # Compile the code as if it were a top-level module
    old_modules = list(sys.modules.keys())
    try:
        with topLevelNamespace(path) as namespace:
            compileStream(stream, namespace, filename=filename)
    finally:
        if not cacheImports:
            to_remove = []
            for name, module in sys.modules.items():
                if name not in old_modules and getattr(module, '_isScenicModule', False):
                    to_remove.append(name)
            for name in to_remove:
                del sys.modules[name]
    # Construct a Scenario from the resulting namespace
    return constructScenarioFrom(namespace)


@contextmanager
def topLevelNamespace(path=None):
    """Creates an environment like that of a Python script being run directly.

    Specifically, __name__ is '__main__', __file__ is the path used to invoke
    the script (not necessarily its absolute path), and the parent directory is
    added to the path so that 'import blobbo' will import blobbo from that
    directory if it exists there.
    """
    directory = os.getcwd() if path is None else os.path.dirname(path)
    namespace = {'__name__': '__main__'}
    if path is not None:
        namespace['__file__'] = path
    sys.path.insert(0, directory)
    try:
        yield namespace
    finally:
        del sys.path[0]


def compileStream(stream, namespace, filename='<stream>'):
    """Compile a stream of Scenic code and execute it in a namespace.

    The compilation procedure consists of the following main steps:

        1. Tokenize the input using the Python tokenizer.
        2. Partition the tokens into blocks separated by import statements.
           This is done by the `partitionByImports` function.
        3. Translate Scenic constructions into valid Python syntax.
           This is done by the `TokenTranslator`.
        4. Parse the resulting Python code into an AST using the Python parser.
        5. Modify the AST to achieve the desired semantics for Scenic.
           This is done by the `translateParseTree` function.
        6. Compile and execute the modified AST.
        7. After executing all blocks, extract the global state (e.g. objects).
           This is done by the `storeScenarioStateIn` function.
    """
    if verbosity >= 2:
        veneer.verbosePrint(f'  Compiling Scenic module from {filename}...')
        startTime = time.time()
    # Tokenize input stream
    try:
        tokens = list(tokenize.tokenize(stream.readline))
    except tokenize.TokenError as e:
        line = e.args[1][0] if isinstance(e.args[1], tuple) else e.args[1]
        raise TokenParseError(line, 'file ended during multiline string or expression')
    # Partition into blocks with all imports at the end (since imports could
    # pull in new constructor (Scenic class) definitions, which change the way
    # subsequent tokens are transformed)
    blocks = partitionByImports(tokens)
    veneer.activate()
    newSourceBlocks = []
    try:
        # Execute preamble
        exec(compile(preamble, '<veneer>', 'exec'), namespace)
        # Execute each block
        for blockNum, block in enumerate(blocks):
            # Find all custom constructors defined so far (possibly imported)
            constructors = findConstructorsIn(namespace)
            # Translate tokens to valid Python syntax
            startLine = max(1, block[0][2][0])
            translator = TokenTranslator(constructors)
            newSource, allConstructors = translator.translate(block)
            trimmed = newSource[2 * (startLine - 1):]  # remove blank lines used to align errors
            newSourceBlocks.append(trimmed)
            if dumpTranslatedPython:
                print(f'### Begin translated Python from block {blockNum} of {filename}')
                print(newSource)
                print('### End translated Python')
            # Parse the translated source
            tree = parseTranslatedSource(newSource, filename)
            # Modify the parse tree to produce the correct semantics
            newTree, requirements = translateParseTree(tree, allConstructors)
            if dumpFinalAST:
                print(f'### Begin final AST from block {blockNum} of {filename}')
                print(ast.dump(newTree, include_attributes=True))
                print('### End final AST')
            # Compile the modified tree
            code = compileTranslatedTree(newTree, filename)
            # Execute it
            executeCodeIn(code, namespace, filename)
        # Extract scenario state from veneer and store it
        storeScenarioStateIn(namespace, requirements, filename)
    finally:
        veneer.deactivate()
    if verbosity >= 2:
        totalTime = time.time() - startTime
        veneer.verbosePrint(f'  Compiled Scenic module in {totalTime:.4g} seconds.')
    allNewSource = ''.join(newSourceBlocks)
    return code, allNewSource


### TRANSLATION PHASE ZERO: definitions of language elements not already in Python

## Options

showInternalBacktrace = False
dumpTranslatedPython = False
dumpFinalAST = False
verbosity = 0
usePruning = True

## Preamble
# (included at the beginning of every module to be translated;
# imports the implementations of the public language features)
preamble = """\
from scenic3d.syntax.veneer import *
"""

## Get Python names of various elements
## (for checking consistency between the translator and the veneer)

api = set(veneer.__all__)

## Functions used internally

rangeConstructor = 'Range'
createDefault = 'PropertyDefault'
internalFunctions = {rangeConstructor, createDefault}

# sanity check: these functions actually exist
for imp in internalFunctions:
    assert imp in api, imp

## Statements implemented by functions

requireStatement = 'require'
paramStatement = 'param'
functionStatements = {requireStatement, paramStatement, 'mutate'}

# sanity check: implementations of statements actually exist
for imp in functionStatements:
    assert imp in api, imp

## Built-in functions

builtinFunctions = {'resample', 'verbosePrint'}

# sanity check: implementations of built-in functions actually exist
for imp in builtinFunctions:
    assert imp in api, imp

## Constructors and specifiers

# statement defining a new constructor (Scenic class);
# we still recognize 'constructor' for backwards-compatibility
constructorStatements = ('class', 'constructor')

Constructor = namedtuple('Constructor', ('name', 'parent', 'specifiers'))

point3dSpecifiers = {
    ('at',): 'At3D',
    ('in',): 'In3D',
    ('on',): 'OnTopOf',
    ('completely', 'on'): 'OnTopOfStrict',
    ('beyond',): 'Beyond3D',
    ('offset', 'by'): 'OffsetBy3D',
    ('facing',): 'Facing3D',
    ('facing', 'toward'): 'FacingToward3D',
    ('left', 'of'): 'LeftRough',
    ('right', 'of'): 'RightRough',
    ('ahead', 'of'): 'AheadRough',
    ('behind',): 'Behind3D',
    ('above',): 'Above3D',
    ('below',): 'Below3D',
    ('following',): 'Following3D',
}

orientedPoint3DSpecifiers = {

}


objectSpecifiers = {
}

# sanity check: implementations of specifiers actually exist
for imp in objectSpecifiers.values():
    assert imp in api, imp
for imp in point3dSpecifiers.values():
    assert imp in api, imp

builtinConstructors = {
    'Point3D': Constructor('Point3D', None, point3dSpecifiers),
    'OrientedPoint3D': Constructor('OrientedPoint3D', 'Point3D', orientedPoint3DSpecifiers),
    'Object': Constructor('Object', 'OrientedPoint3D', objectSpecifiers)
}
functionStatements.update(builtinConstructors)

# sanity check: built-in constructors actually exist
for const in builtinConstructors:
    assert const in api, const

# Prefix operators

prefixOperators = {
    ('relative', 'position'): 'RelativePosition',
    ('relative', 'heading'): 'RelativeHeading',
    ('apparent', 'heading'): 'ApparentHeading',
    ('distance', 'from'): 'DistanceFrom',
    ('distance', 'to'): 'DistanceFrom',
    ('front', 'left'): 'FrontLeft',
    ('front', 'right'): 'FrontRight',
    ('back', 'left'): 'BackLeft',
    ('back', 'right'): 'BackRight',
    ('front',): 'Front',
    ('back',): 'Back',
    ('left',): 'Left',
    ('right',): 'Right',
    ('top', 'front'): 'TopFront',
    ('top', 'back'): 'TopBack',
    ('top',): 'Top',
    ('bottom',): 'Bottom'
}
assert all(1 <= len(op) <= 2 for op in prefixOperators)
prefixIncipits = {op[0] for op in prefixOperators}
assert not any(op in functionStatements for op in prefixIncipits)

# sanity check: implementations of prefix operators actually exist
for imp in prefixOperators.values():
    assert imp in api, imp

## Infix operators

# pseudo-operator for encoding argument packages for (3+)-ary ops
packageToken = (RIGHTSHIFT, '>>')
packageNode = RShift

InfixOp = namedtuple('InfixOp', ('syntax', 'implementation', 'arity', 'token', 'node'))
infixOperators = (
    # existing Python operators with new semantics
    InfixOp('@', 'Vector', 2, None, MatMult),

    # operators not in Python (in decreasing precedence order)
    InfixOp('at', 'FieldAt', 2, (LEFTSHIFT, '<<'), LShift),
    InfixOp('relative to', 'RelativeTo', 2, (AMPER, '&'), BitAnd),
    InfixOp('offset by', 'RelativeTo', 2, (AMPER, '&'), BitAnd),
    InfixOp('offset along', 'OffsetAlong', 3, (CIRCUMFLEX, '^'), BitXor),
    InfixOp('can see', 'CanSee', 2, (VBAR, '|'), BitOr),

    # just syntactic conveniences, not really operators
    InfixOp('from', None, 2, (COMMA, ','), None),
    InfixOp('for', None, 2, (COMMA, ','), None),
    InfixOp('to', None, 2, (COMMA, ','), None),
    InfixOp('by', None, 2, packageToken, None)
)

infixTokens = {}
infixImplementations = {}
infixIncipits = set()
for op in infixOperators:
    # if necessary, set up map from Scenic to Python syntax
    if op.token is not None:
        tokens = tuple(op.syntax.split(' '))
        assert 1 <= len(tokens) <= 2, op
        assert tokens not in infixTokens, op
        infixTokens[tokens] = op.token
        incipit = tokens[0]
        assert incipit not in functionStatements, op
        infixIncipits.add(incipit)
    # if necessary, set up map from Python to Scenic semantics
    imp = op.implementation
    if imp is not None:
        assert imp in api, op
        node = op.node
        if node in infixImplementations:  # two operators may have the same implementation
            oldArity, oldName = infixImplementations[node]
            assert op.arity == oldArity, (op, oldName)
            assert imp == oldName, (op, oldName)
        else:
            infixImplementations[node] = (op.arity, imp)

allIncipits = prefixIncipits | infixIncipits

## Direct syntax replacements

replacements = {  # TODO police the usage of these? could yield bizarre error messages
    'of': tuple(),
    'deg': ((STAR, '*'), (NUMBER, '0.01745329252')),
}

## Illegal and reserved syntax

illegalTokens = {
    LEFTSHIFT, RIGHTSHIFT, VBAR, AMPER, TILDE, CIRCUMFLEX,
    LEFTSHIFTEQUAL, RIGHTSHIFTEQUAL, VBAREQUAL, AMPEREQUAL, CIRCUMFLEXEQUAL,
    DOUBLESLASH, DOUBLESLASHEQUAL
}

# sanity check: stand-in tokens for infix operators must be illegal
for token in infixTokens.values():
    ttype = token[0]
    assert (ttype is COMMA or ttype in illegalTokens), token

keywords = (set(constructorStatements)
            | internalFunctions | functionStatements
            | replacements.keys())

### TRANSLATION PHASE ONE: handling imports

## Meta path finder and loader for Scenic files

scenicExtensions = ('sc', 'scenic')


class ScenicMetaFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, paths, target):
        if paths is None:
            paths = sys.path
            modname = name
        else:
            modname = name.rpartition('.')[2]
        for path in paths:
            for extension in scenicExtensions:
                filename = modname + '.' + extension
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath):
                    filepath = os.path.abspath(filepath)
                    spec = importlib.util.spec_from_file_location(name, filepath,
                                                                  loader=ScenicLoader(filepath, filename))
                    return spec
        return None


class ScenicLoader(importlib.abc.InspectLoader):
    def __init__(self, filepath, filename):
        self.filepath = filepath
        self.filename = filename

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Read source file and compile it
        with open(self.filepath, 'r') as stream:
            source = stream.read()
        with open(self.filepath, 'rb') as stream:
            code, pythonSource = compileStream(stream, module.__dict__, filename=self.filepath)
        # Mark as a Scenic module
        module._isScenicModule = True
        # Save code, source, and translated source for later inspection
        module._code = code
        module._source = source
        module._pythonSource = pythonSource

    def is_package(self, fullname):
        return False

    def get_code(self, fullname):
        module = importlib.import_module(fullname)
        assert module._isScenicModule, module
        return module._code

    def get_source(self, fullname):
        module = importlib.import_module(fullname)
        assert module._isScenicModule, module
        return module._pythonSource


# register the meta path finder
sys.meta_path.insert(0, ScenicMetaFinder())


## Post-import hook to inherit objects, etc. from imported Scenic modules

def hooked_import(*args, **kwargs):
    """Version of __import__ hooked by Scenic to capture Scenic modules."""
    module = original_import(*args, **kwargs)
    if getattr(module, '_isScenicModule', False):
        if veneer.isActive():
            veneer.allObjects.extend(module._objects)
            veneer.globalParameters.update(module._params)
            veneer.externalParameters.extend(module._externalParams)
            veneer.inheritedReqs.extend(module._requirements)
    return module


original_import = builtins.__import__
builtins.__import__ = hooked_import


## Miscellaneous utilities

def partitionByImports(tokens):
    """Partition the tokens into blocks ending with import statements."""
    blocks = []
    currentBlock = []
    duringImport = False
    haveImported = False
    finishLine = False
    parenLevel = 0
    for token in tokens:
        startNewBlock = False
        if token.exact_type == LPAR:
            parenLevel += 1
        elif token.exact_type == RPAR:
            parenLevel -= 1
        if finishLine:
            if token.type in (NEWLINE, NL) and parenLevel == 0:
                finishLine = False
                if duringImport:
                    duringImport = False
                    haveImported = True
        else:
            assert not duringImport
            finishLine = True
            if token.type == NAME and token.string == 'import' or token.string == 'from':
                duringImport = True
            elif token.type in (NEWLINE, NL, COMMENT, ENCODING):
                finishLine = False
            elif haveImported:
                # could use new constructors; needs to be in a new block
                startNewBlock = True
        if startNewBlock:
            blocks.append(currentBlock)
            currentBlock = [token]
            haveImported = False
        else:
            currentBlock.append(token)
    blocks.append(currentBlock)  # add last block
    return blocks


def findConstructorsIn(namespace):
    """Find all constructors (Scenic classes) defined in a namespace."""
    constructors = []
    for name, value in namespace.items():
        if inspect.isclass(value) and issubclass(value, Constructible):
            if name in builtinConstructors:
                continue
            parent = None
            for base in value.__bases__:
                if issubclass(base, Constructible):
                    assert parent is None
                    parent = base
            constructors.append(Constructor(name, parent.__name__, {}))
    return constructors


### TRANSLATION PHASE TWO: translation at the level of tokens

class TokenParseError(ParseError):
    """Parse error occurring during token translation."""

    def __init__(self, tokenOrLine, message):
        line = tokenOrLine.start[0] if hasattr(tokenOrLine, 'start') else tokenOrLine
        self.lineno = line
        super().__init__('Parse error in line ' + str(line) + ': ' + message)


class Peekable:
    """Utility class to allow iterator lookahead."""

    def __init__(self, gen):
        self.gen = iter(gen)
        self.current = next(self.gen, None)

    def __iter__(self):
        return self

    def __next__(self):
        cur = self.current
        if cur is None:
            raise StopIteration
        self.current = next(self.gen, None)
        return cur

    def peek(self):
        return self.current


def peek(thing):
    return thing.peek()


class TokenTranslator:
    """Translates a Scenic token stream into valid Python syntax.

    This is a stateful process because constructor (Scenic class) definitions
    change the way subsequent code is parsed.
    """

    def __init__(self, constructors=()):
        self.functions = set(functionStatements)
        self.constructors = dict(builtinConstructors)
        for constructor in constructors:
            name = constructor.name
            assert name not in self.constructors
            self.constructors[name] = constructor
            self.functions.add(name)

    def createConstructor(self, name, parent, specs=None):
        if specs is None:
            specs = {}
        if parent is None:
            parent = 'Object'  # default superclass
        self.constructors[name] = Constructor(name, parent, specs)
        self.functions.add(name)
        return parent

    def specifiersForConstructor(self, const):
        name, parent, specs = self.constructors[const]
        if parent is None:
            return specs
        else:
            ps = dict(self.specifiersForConstructor(parent))
            ps.update(specs)
            return ps

    def translate(self, tokens):
        """Do the actual translation of the token stream."""
        tokens = Peekable(tokens)
        newTokens = []
        functionStack = []
        inConstructor = False  # inside a constructor or one of its specifiers
        specifiersIndented = False
        parenLevel = 0
        row, col = 0, 0  # position of next token to write out
        orow, ocol = 0, 0  # end of last token in the original source
        startOfLine = True  # TODO improve hack?
        functions = self.functions
        constructors = self.constructors

        for token in tokens:
            ttype = token.exact_type
            tstring = token.string
            skip = False
            endToken = token  # token to advance past in column count
            movedUpTo = False

            def injectToken(tok, spaceAfter=0):
                """Add a token to the output stream, trying to preserve spacing."""
                nonlocal row, col, movedUpTo
                if not movedUpTo:
                    moveUpTo(token)
                    moveBeyond(token)
                    movedUpTo = True
                ty, string = tok[:2]
                if len(tok) >= 3:
                    moveBeyond(tok)
                    srow, scol = tok[2]
                    erow, ecol = tok[3]
                    width = ecol - scol
                    height = erow - srow
                else:
                    width = len(string)
                    height = 0
                ncol = ecol if height > 0 else col + width
                newToken = (ty, string, (row, col), (row + height, ncol), '')
                newTokens.append(newToken)
                if ty in (NEWLINE, NL):
                    row += 1
                    col = 0
                elif height > 0:
                    row += height
                    col = ncol
                else:
                    col += width + spaceAfter

            def moveUpTo(tok):
                nonlocal row, col, orow, ocol
                nrow, ncol = tok[2]
                if nrow > orow:
                    row = nrow
                    col = ncol
                else:
                    gap = ncol - ocol
                    assert gap >= 0, (tok, row, col, ocol)
                    col += gap

            def moveBeyond(tok):
                nonlocal orow, ocol
                nrow, ncol = tok[3]
                if nrow > orow or (nrow == orow and ncol > ocol):
                    orow = nrow
                    ocol = ncol

            def advance(skip=True):
                nextToken = next(tokens)
                if skip:
                    moveBeyond(nextToken)
                else:
                    injectToken(nextToken)
                return nextToken

            def callFunction(function, argument=None):
                nonlocal skip, matched, functionStack
                functionStack.append((function, parenLevel))
                injectToken((NAME, function))
                injectToken((LPAR, '('))
                if argument is not None:
                    injectToken((NAME, argument))
                    injectToken((COMMA, ','))
                skip = True
                matched = True

            # Catch Python operators that can't be used in Scenic
            if ttype in illegalTokens:
                raise TokenParseError(token, f'illegal operator "{tstring}"')

            # Determine which operators are allowed in current context
            context, startLevel = functionStack[-1] if functionStack else (None, None)
            inConstructorContext = (context in constructors and parenLevel == startLevel)
            if inConstructorContext:
                inConstructor = True
                allowedPrefixOps = self.specifiersForConstructor(context)
                allowedInfixOps = dict()
            else:
                allowedPrefixOps = prefixOperators
                allowedInfixOps = infixTokens

            # Parse next token
            if ttype == LPAR or ttype == LSQB:  # keep track of nesting level
                parenLevel += 1
            elif ttype == RPAR or ttype == RSQB:  # ditto
                parenLevel -= 1
            elif ttype == STRING:
                # special case for global parameters with quoted names:
                # transform "name"=value into "name", value
                if (len(functionStack) > 0 and functionStack[-1][0] == paramStatement
                        and peek(tokens).string == '='):
                    next(tokens)  # consume '='
                    injectToken(token)
                    injectToken((COMMA, ','))
                    skip = True
            elif ttype == NAME:  # the interesting case: almost all new syntax falls in here
                # try to match 2-word language constructs
                matched = False
                nextToken = peek(tokens)  # lookahead so we can give 2-word ops precedence
                if nextToken is not None:
                    endToken = nextToken  # tentatively; will be overridden if no match
                    nextString = nextToken.string
                    twoWords = (tstring, nextString)
                    if startOfLine and tstring == 'for':  # TODO improve hack?
                        matched = True
                        endToken = token
                    elif startOfLine and tstring in constructorStatements:  # class definition
                        if nextToken.type != NAME or nextString in keywords:
                            raise TokenParseError(nextToken,
                                                  f'invalid class name "{nextString}"')
                        nextToken = next(tokens)  # consume name
                        parent = None
                        pythonClass = False
                        if peek(tokens).exact_type == LPAR:  # superclass specification
                            next(tokens)
                            nextToken = next(tokens)
                            parent = nextToken.string
                            if nextToken.exact_type != NAME or parent in keywords:
                                raise TokenParseError(nextToken,
                                                      f'invalid superclass "{parent}"')
                            if parent not in self.constructors:
                                if tstring != 'class':
                                    raise TokenParseError(nextToken,
                                                          f'superclass "{parent}" is not a Scenic class')
                                # appears to be a Python class definition
                                pythonClass = True
                            else:
                                nextToken = next(tokens)
                                if nextToken.exact_type != RPAR:
                                    raise TokenParseError(nextToken,
                                                          'malformed class definition')
                        injectToken((NAME, 'class'), spaceAfter=1)
                        injectToken((NAME, nextString))
                        injectToken((LPAR, '('))
                        if pythonClass:  # pass Python class definitions through unchanged
                            while nextToken.exact_type != COLON:
                                injectToken(nextToken)
                                nextToken = next(tokens)
                            injectToken(nextToken)
                        else:
                            if peek(tokens).exact_type != COLON:
                                raise TokenParseError(nextToken, 'malformed class definition')
                            parent = self.createConstructor(nextString, parent)
                            injectToken((NAME, parent))
                            injectToken((RPAR, ')'))
                        skip = True
                        matched = True
                        endToken = nextToken
                    elif twoWords in allowedPrefixOps:  # 2-word prefix operator
                        callFunction(allowedPrefixOps[twoWords])
                        advance()  # consume second word
                    elif not startOfLine and twoWords in allowedInfixOps:  # 2-word infix operator
                        injectToken(allowedInfixOps[twoWords])
                        advance()  # consume second word
                        skip = True
                        matched = True
                    elif inConstructorContext and tstring == 'with':  # special case for 'with' specifier
                        callFunction('With', argument=f'"{nextString}"')
                        advance()  # consume property name
                    elif tstring == requireStatement and nextString == '[':  # special case for require[p]
                        next(tokens)  # consume '['
                        nextToken = next(tokens)
                        if nextToken.exact_type != NUMBER:
                            raise TokenParseError(nextToken,
                                                  'soft requirement must have constant probability')
                        prob = nextToken.string
                        nextToken = next(tokens)
                        if nextToken.exact_type != RSQB:
                            raise TokenParseError(nextToken, 'malformed soft requirement')
                        callFunction(requireStatement, argument=prob)
                        endToken = nextToken
                if not matched:
                    # 2-word constructs don't match; try 1-word
                    endToken = token
                    oneWord = (tstring,)
                    if oneWord in allowedPrefixOps:  # 1-word prefix operator
                        callFunction(allowedPrefixOps[oneWord])
                    elif not startOfLine and oneWord in allowedInfixOps:  # 1-word infix operator
                        injectToken(allowedInfixOps[oneWord])
                        skip = True
                    elif inConstructorContext:  # couldn't match any 1- or 2-word specifier
                        raise TokenParseError(token, f'unknown specifier "{tstring}"')
                    elif tstring in functions:  # built-in function
                        callFunction(tstring)
                    elif tstring in replacements:  # direct replacement
                        for tok in replacements[tstring]:
                            injectToken(tok)
                        skip = True
                    elif startOfLine and tstring == 'from':  # special case to allow 'from X import Y'
                        pass
                    elif tstring in keywords:  # some malformed usage
                        raise TokenParseError(token, f'unexpected keyword "{tstring}"')
                    else:
                        pass  # nothing matched; pass through unchanged to Python

            # Detect the end of function argument lists
            if len(functionStack) > 0:
                context, startLevel = functionStack[-1]
                while parenLevel < startLevel:  # we've closed all parens for the current function
                    functionStack.pop()
                    injectToken((RPAR, ')'))
                    context, startLevel = (None, 0) if len(functionStack) == 0 else functionStack[-1]
                if inConstructor and parenLevel == startLevel and ttype == COMMA:  # starting a new specifier
                    while functionStack and context not in constructors:
                        functionStack.pop()
                        injectToken((RPAR, ')'))
                        context, startLevel = (None, 0) if len(functionStack) == 0 else functionStack[-1]
                    # allow the next specifier to be on the next line, if indented
                    injectToken(token)  # emit comma immediately
                    skip = True
                    nextToken = peek(tokens)
                    specOnNewLine = False
                    while nextToken.exact_type in (NEWLINE, NL, COMMENT, ENDMARKER):
                        specOnNewLine = True
                        if nextToken.exact_type == COMMENT:
                            advance(skip=False)  # preserve comment
                            nextToken = peek(tokens)
                        if nextToken.exact_type not in (NEWLINE, NL):
                            raise TokenParseError(nextToken, 'comma with no specifier following')
                        advance(skip=False)  # preserve newline
                        nextToken = peek(tokens)
                    if specOnNewLine and not specifiersIndented:
                        nextToken = next(tokens)  # consume indent
                        if nextToken.exact_type != INDENT:
                            raise TokenParseError(nextToken,
                                                  'expected indented specifier (extra comma on previous line?)')
                        injectToken(nextToken)
                        specifiersIndented = True
                elif ttype == NEWLINE or ttype == ENDMARKER or ttype == COMMENT:  # end of line
                    inConstructor = False
                    if parenLevel != 0:
                        raise TokenParseError(token, 'unmatched parens/brackets')
                    while len(functionStack) > 0:
                        functionStack.pop()
                        injectToken((RPAR, ')'))

            # Output token unchanged, unless handled above
            if not skip:
                injectToken(token)
            else:
                moveBeyond(endToken)
            startOfLine = (ttype in (ENCODING, NEWLINE, NL, INDENT, DEDENT))

        rewrittenSource = tokenize.untokenize(newTokens)
        if not isinstance(rewrittenSource, str):  # TODO improve?
            rewrittenSource = str(rewrittenSource, encoding='utf-8')
        return rewrittenSource, self.constructors


### TRANSLATION PHASE THREE: parsing of Python resulting from token translation

class PythonParseError(SyntaxError, ParseError):
    """Parse error occurring during Python parsing or compilation."""

    @classmethod
    def fromSyntaxError(cls, exc):
        msg, (filename, lineno, offset, line) = exc.args
        try:  # attempt to recover line from original file
            with open(filename, 'r') as f:
                line = list(itertools.islice(f, lineno - 1, lineno))
            assert len(line) == 1
            line = line[0]
            offset = min(offset, len(line))  # TODO improve?
        except FileNotFoundError:
            pass
        newExc = cls(msg, (filename, lineno, offset, line))
        return newExc.with_traceback(exc.__traceback__)


def parseTranslatedSource(source, filename):
    try:
        tree = parse(source, filename=filename)
        return tree
    except SyntaxError as e:
        cause = e if showInternalBacktrace else None
        raise PythonParseError.fromSyntaxError(e) from cause


### TRANSLATION PHASE FOUR: modifying the parse tree

noArgs = ast.arguments(
    args=[], vararg=None,
    kwonlyargs=[], kw_defaults=[],
    kwarg=None, defaults=[])
selfArg = ast.arguments(
    args=[ast.arg(arg='self', annotation=None)], vararg=None,
    kwonlyargs=[], kw_defaults=[],
    kwarg=None, defaults=[])

if sys.version_info >= (3, 8):  # TODO cleaner way to handle this?
    noArgs.posonlyargs = []
    selfArg.posonlyargs = []


class AttributeFinder(NodeVisitor):
    """Utility class for finding all referenced attributes of a given name."""

    @staticmethod
    def find(target, node):
        af = AttributeFinder(target)
        af.visit(node)
        return af.attributes

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.attributes = set()

    def visit_Attribute(self, node):
        val = node.value
        if isinstance(val, Name) and val.id == self.target:
            self.attributes.add(node.attr)
        self.visit(val)


class ASTParseError(ParseError):
    """Parse error occuring during modification of the Python AST."""

    def __init__(self, line, message):
        self.lineno = line
        super().__init__('Parse error in line ' + str(line) + ': ' + message)


class ASTSurgeon(NodeTransformer):
    def __init__(self, constructors):
        super().__init__()
        self.constructors = set(constructors.keys())
        self.requirements = []

    def parseError(self, node, message):
        raise ASTParseError(node.lineno, message)

    def unpack(self, arg, expected, node):
        """Unpack arguments to ternary (and up) infix operators."""
        assert expected > 0
        if isinstance(arg, BinOp) and isinstance(arg.op, packageNode):
            if expected == 1:
                raise self.parseError(node, 'gave too many arguments to infix operator')
            else:
                return self.unpack(arg.left, expected - 1, node) + [self.visit(arg.right)]
        elif expected > 1:
            raise self.parseError(node, 'gave too few arguments to infix operator')
        else:
            return [self.visit(arg)]

    def visit_BinOp(self, node):
        """Convert infix operators to calls to the corresponding Scenic operator implementations."""
        left = node.left
        right = node.right
        op = node.op
        if isinstance(op, packageNode):  # unexpected argument package
            raise self.parseError(node, 'unexpected keyword "by"')
        elif type(op) in infixImplementations:  # an operator with non-Python semantics
            arity, impName = infixImplementations[type(op)]
            implementation = Name(impName, Load())
            copy_location(implementation, node)
            assert arity >= 2
            args = [self.visit(left)] + self.unpack(right, arity - 1, node)
            newNode = Call(implementation, args, [])
        else:  # all other operators have the Python semantics
            newNode = BinOp(self.visit(left), op, self.visit(right))
        return copy_location(newNode, node)

    def visit_Tuple(self, node):
        """Convert pairs into uniform distributions."""
        if isinstance(node.ctx, Store):
            return self.generic_visit(node)
        if len(node.elts) != 2:
            raise self.parseError(node, 'interval must have exactly two endpoints')
        newElts = [self.visit(elt) for elt in node.elts]
        return copy_location(Call(Name(rangeConstructor, Load()), newElts, []), node)

    def visit_Call(self, node):
        """Wrap require statements with lambdas and unpack any argument packages."""
        func = node.func
        if isinstance(func, Name) and func.id == requireStatement:  # Require statement
            # Soft reqs have 2 arguments, including the probability, which is given as the
            # first argument by the token translator; so we allow an extra argument here and
            # validate it later on (in case the user wrongly gives 2 arguments to require).
            if not (1 <= len(node.args) <= 2):
                raise self.parseError(node, 'require takes exactly one argument')
            if len(node.keywords) != 0:
                raise self.parseError(node, 'require takes no keyword arguments')
            cond = node.args[-1]
            if isinstance(cond, Starred):
                raise self.parseError(node, 'argument unpacking cannot be used with require')
            req = self.visit(cond)
            reqID = Num(len(self.requirements))  # save ID number
            self.requirements.append(req)  # save condition for later inspection when pruning
            closure = Lambda(noArgs, req)  # enclose requirement in a lambda
            lineNum = Num(node.lineno)  # save line number for error messages
            copy_location(closure, req)
            copy_location(lineNum, req)
            newArgs = [reqID, closure, lineNum]
            if len(node.args) == 2:  # get probability for soft requirements
                prob = node.args[0]
                if not isinstance(prob, Num):
                    raise self.parseError(node, 'malformed requirement '
                                                '(should be a single expression)')
                newArgs.append(prob)
            return copy_location(Call(func, newArgs, []), node)
        else:  # Ordinary function call
            newFunc = self.visit(func)
            newArgs = []
            # Translate arguments, unpacking any argument packages
            for arg in node.args:
                if isinstance(arg, BinOp) and isinstance(arg.op, packageNode):
                    newArgs.extend(self.unpack(arg, 2, node))
                else:
                    newArgs.append(self.visit(arg))
            newKeywords = [self.visit(kwarg) for kwarg in node.keywords]
            return copy_location(Call(newFunc, newArgs, newKeywords), node)

    def visit_ClassDef(self, node):
        """Process property defaults for Scenic classes."""
        if node.name in self.constructors:  # Scenic class definition
            newBody = []
            for child in node.body:
                child = self.visit(child)
                if isinstance(child, AnnAssign):  # default value for property
                    origValue = child.annotation
                    target = child.target
                    # extract any attributes for this property
                    metaAttrs = []
                    if isinstance(target, Subscript):
                        sl = target.slice
                        if not isinstance(sl, Index):
                            self.parseError(sl, 'malformed attributes for property default')
                        sl = sl.value
                        if isinstance(sl, Name):
                            metaAttrs.append(sl.id)
                        elif isinstance(sl, Tuple):
                            for elt in sl.elts:
                                if not isinstance(elt, Name):
                                    self.parseError(elt,
                                                    'malformed attributes for property default')
                                metaAttrs.append(elt.id)
                        else:
                            self.parseError(sl, 'malformed attributes for property default')
                        newTarget = Name(target.value.id, Store())
                        copy_location(newTarget, target)
                        target = newTarget
                    # find dependencies of the default value
                    properties = AttributeFinder.find('self', origValue)
                    # create default value object
                    args = [
                        Set([Str(prop) for prop in properties]),
                        Set([Str(attr) for attr in metaAttrs]),
                        Lambda(selfArg, origValue)
                    ]
                    value = Call(Name(createDefault, Load()), args, [])
                    copy_location(value, origValue)
                    newChild = AnnAssign(
                        target=target, annotation=value,
                        value=None, simple=True)
                    child = copy_location(newChild, child)
                newBody.append(child)
            node.body = newBody
            return node
        else:  # ordinary Python class
            # it's impossible at the moment to define a Python class in a Scenic file,
            # but we'll leave this check here for future-proofing
            for base in node.bases:
                name = None
                if isinstance(base, Call):
                    name = base.func.id
                elif isinstance(base, Name):
                    name = base.id
                if name is not None and name in self.constructors:
                    self.parseError(node,
                                    f'Python class {node.name} derives from Scenic class {name}')
            return self.generic_visit(node)


def translateParseTree(tree, constructors):
    """Modify the Python AST to produce the desired Scenic semantics."""
    surgeon = ASTSurgeon(constructors)
    tree = fix_missing_locations(surgeon.visit(tree))
    return tree, surgeon.requirements


### TRANSLATION PHASE FIVE: AST compilation

def compileTranslatedTree(tree, filename):
    try:
        return compile(tree, filename, 'exec')
    except SyntaxError as e:
        cause = e if showInternalBacktrace else None
        raise PythonParseError.fromSyntaxError(e) from cause


### TRANSLATION PHASE SIX: Python execution

def generateTracebackFrom(exc, sourceFile):
    """Trim an exception's traceback to the last line of Scenic code."""
    # find last stack frame in the source file
    tbexc = traceback.TracebackException.from_exception(exc)
    last = None
    tbs = []
    currentTb = exc.__traceback__
    for depth, frame in enumerate(tbexc.stack):
        assert currentTb is not None
        tbs.append(currentTb)
        currentTb = currentTb.tb_next
        if frame.filename == sourceFile:
            last = depth
    assert last is not None

    # create new trimmed traceback
    lastTb = tbs[last]
    lastLine = lastTb.tb_lineno
    tbs = tbs[:last]
    try:
        currentTb = types.TracebackType(None, lastTb.tb_frame,
                                        lastTb.tb_lasti, lastLine)
    except TypeError:
        # Python 3.6 does not allow creation of traceback objects, so we just
        # return the original traceback
        return exc.__traceback__, lastLine

    for tb in reversed(tbs):
        currentTb = types.TracebackType(currentTb, tb.tb_frame,
                                        tb.tb_lasti, tb.tb_lineno)
    return currentTb, lastLine


class InterpreterParseError(ParseError):
    """Parse error occuring during Python execution."""

    def __init__(self, exc, line):
        self.lineno = line
        exc_name = type(exc).__name__
        super().__init__(f'Parse error in line {line}: {exc_name}: {exc}')


def executeCodeIn(code, namespace, filename):
    """Execute the final translated Python code in the given namespace."""
    executePythonFunction(lambda: exec(code, namespace), filename)


def executePythonFunction(func, filename):
    """Execute a Python function, giving correct Scenic backtraces for any exceptions."""
    try:
        return func()
    except RuntimeParseError as e:
        cause = e if showInternalBacktrace else None
        tb, line = generateTracebackFrom(e, filename)
        raise InterpreterParseError(e, line).with_traceback(tb) from cause


### TRANSLATION PHASE SEVEN: scenario construction

def storeScenarioStateIn(namespace, requirementSyntax, filename):
    """Post-process an executed Scenic module, extracting state from the veneer."""
    # Extract created Objects
    namespace['_objects'] = tuple(veneer.allObjects)

    # Extract global parameters
    namespace['_params'] = veneer.globalParameters
    for name, value in veneer.globalParameters.items():
        if needs_lazy_evaluation(value):
            raise InvalidScenarioError(f'parameter {name} uses value {value}'
                                       ' undefined outside of object definition')

    # Extract external parameters
    namespace['_externalParams'] = tuple(veneer.externalParameters)

    # Extract requirements, scan for relations used for pruning, and create closures
    requirements = veneer.pendingRequirements
    finalReqs = veneer.inheritedReqs
    requirementDeps = set()  # things needing to be sampled to evaluate the requirements
    namespace['_requirements'] = finalReqs
    namespace['_requirementDeps'] = requirementDeps

    def makeClosure(req, bindings, ego, line):
        """Create a closure testing the requirement in the correct runtime state."""

        def evaluator():
            result = req()
            assert not needs_sampling(result)
            if needs_lazy_evaluation(result):
                raise InvalidScenarioError(f'requirement on line {line} uses value'
                                           ' undefined outside of object definition')
            return result

        def closure(values):
            # rebind any names referring to sampled objects
            for name, value in bindings.items():
                if value in values:
                    namespace[name] = values[value]
            # evaluate requirement condition, reporting errors on the correct line
            try:
                veneer.evaluatingRequirement = True
                # rebind ego object, which can be referred to implicitly
                assert veneer.egoObject is None
                if ego is not None:
                    veneer.egoObject = values[ego]
                result = executePythonFunction(evaluator, filename)
            finally:
                veneer.evaluatingRequirement = False
                veneer.egoObject = None
            return result

        return closure

    for reqID, (req, bindings, ego, line, prob) in requirements.items():
        # Check whether requirement implies any relations used for pruning
        reqNode = requirementSyntax[reqID]
        relations.inferRelationsFrom(reqNode, bindings, ego, line)
        # Gather dependencies of the requirement
        for value in bindings.values():
            if needs_sampling(value):
                requirementDeps.add(value)
            if needs_lazy_evaluation(value):
                raise InvalidScenarioError(f'requirement on line {line} uses value {value}'
                                           ' undefined outside of object definition')
        if ego is not None:
            assert isinstance(ego, Samplable)
            requirementDeps.add(ego)
        # Construct closure
        finalReqs.append((makeClosure(req, bindings, ego, line), prob))


def constructScenarioFrom(namespace):
    """Build a Scenario object from an executed Scenic module."""
    # Extract workspace, if one is specified
    if 'workspace' in namespace:
        workspace = namespace['workspace']
        if not isinstance(workspace, Region):
            raise InvalidScenarioError(f'workspace {workspace} is not a Workspace')
        if needs_sampling(workspace):
            raise InvalidScenarioError('workspace must be a fixed region')
        if needs_lazy_evaluation(workspace):
            raise InvalidScenarioError('workspace uses value undefined '
                                       'outside of object definition')
    else:
        workspace = None

    # Create Scenario object
    scenario = Scenario(workspace,
                        namespace['_objects'],
                        namespace['_params'], namespace['_externalParams'],
                        namespace['_requirements'], namespace['_requirementDeps'])

    # Prune infeasible parts of the space
    if usePruning:
        pruning.prune(scenario, verbosity=verbosity)

    return scenario
