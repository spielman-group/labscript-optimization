"""What the package promises about itself, rather than about any session.

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
import re
import subprocess
import typing
from pathlib import Path

import labscript_optimization

ROOT = Path(__file__).resolve().parent.parent

#: M-LOOP, however it is spelt.
NAME = re.compile(r'mloop|m-loop|m_loop', re.IGNORECASE)

#: Files whose subject is M-LOOP itself, which may say so as often as they
#: need to. Two are decision records for work already done, one is the
#: document a lab moving off M-LOOP reads, and the last is this file, which
#: has to name the word in order to look for it.
SPEAKS_OF_M_LOOP = frozenset(
    {
        'UPGRADING.md',
        'codex_issues_proposal.md',
        'issues/codex_issues.md',
        'tests/test_package.py',
    }
)

#: Every other file allowed to say it, and how many times. The name belongs
#: where the subject is M-LOOP -- what this package replaces, whose code it
#: carries, which table names it refuses and why a behaviour differs -- and
#: nowhere it is merely inherited.
MENTIONS_M_LOOP = {
    'LICENSE': 5,
    'README.md': 11,
    'benchmarks/README.md': 1,
    'labscript_optimization/config.py': 5,
    'tests/test_config.py': 15,
}


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


def test_the_name_m_loop_appears_only_where_the_subject_is_m_loop():
    """This package replaces M-LOOP and shares none of its code, so its own
    name is what it should be read under. The word crept back once already, in
    the configuration surface a lab types every day; nothing noticed, because
    noticing meant somebody re-running the grep.

    Two tiers, because neither shape guards on its own. A file list alone lets
    the word multiply inside a file already allowed one sentence of it -- a
    README that explains what this replaces could grow twenty mentions and
    stay on the list. A count alone, per file or in total, drifts: every edit
    that moves the number is an edit that has to change this test, and a
    number changed to make a test pass is no guard at all. So the files whose
    whole subject is M-LOOP are named and not counted, and every other file is
    counted exactly, compared as a whole mapping so that a file appearing for
    the first time fails as loudly as a count that moved.
    """
    listed = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = [path for path in listed.split('\0') if path]

    # A listing that reached nothing would agree with every expectation below.
    assert 'labscript_optimization/config.py' in tracked
    assert SPEAKS_OF_M_LOOP <= set(tracked)

    found = {}
    for path in tracked:
        if path in SPEAKS_OF_M_LOOP:
            continue
        text = (ROOT / path).read_bytes().decode('utf-8', errors='replace')
        said = len(NAME.findall(text))
        if said:
            found[path] = said

    assert found == MENTIONS_M_LOOP
