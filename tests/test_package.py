"""What the package promises simply by being importable.

``pyproject.toml`` asks for Python 3.11 or newer, and the suite runs on one
interpreter. What differs between versions is therefore unexercised, and one
of those differences hides a whole class of defect: from 3.14 an annotation is
evaluated only when something asks for it, so a name that does not resolve
raises at import on 3.11 and never here. Asking for every annotation is what
makes the difference visible on the interpreter the suite does run on.

This is not a run on the oldest interpreter the package supports and does not
stand in for one. Nothing in this repository runs on 3.11.
"""

import importlib
import inspect
import pkgutil
import typing
from pathlib import Path

import labscript_optimization


def test_every_annotation_in_the_package_resolves():
    """An annotation naming something that is not there is an import error.

    On 3.11 it is raised the moment the module is read, which in a lab is the
    lyse routine failing to import, and on 3.14 it is raised by whoever first
    asks what the annotation says -- a documentation build, a type checker, or
    nobody at all. Resolving them here is asking.
    """
    modules = [labscript_optimization]
    for found in pkgutil.walk_packages(
        labscript_optimization.__path__, 'labscript_optimization.'
    ):
        modules.append(importlib.import_module(found.name))

    # Every module the package ships, or a walk that quietly reached none of
    # them would be indistinguishable from a package whose annotations are all
    # sound.
    root = Path(labscript_optimization.__file__).parent
    assert {module.__name__ for module in modules} == {
        '.'.join(path.relative_to(root.parent).with_suffix('').parts).removesuffix(
            '.__init__'
        )
        for path in root.rglob('*.py')
    }

    unresolved = []
    for module in modules:
        annotated = [module]
        for owned in vars(module).values():
            if getattr(owned, '__module__', None) != module.__name__:
                continue
            if inspect.isclass(owned):
                annotated.append(owned)
                annotated.extend(
                    member
                    for member in vars(owned).values()
                    if inspect.isfunction(member)
                )
            elif inspect.isfunction(owned):
                annotated.append(owned)
        for target in annotated:
            try:
                typing.get_type_hints(target)
            except Exception as failure:
                unresolved.append(
                    f'{module.__name__}.'
                    f'{getattr(target, "__qualname__", "")}: {failure!r}'
                )

    assert unresolved == []
