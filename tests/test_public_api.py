import importlib

import pytest

import entoscope

SUBPACKAGES = [
    "entoscope.circuit",
    "entoscope.dataset",
    "entoscope.lens",
    "entoscope.model",
    "entoscope.operation",
    "entoscope.patching",
    "entoscope.probe",
    "entoscope.script",
]


@pytest.mark.parametrize("name", ["entoscope", *SUBPACKAGES])
def test_every_exported_name_resolves(name):
    package = importlib.import_module(name)
    assert [export for export in package.__all__ if not hasattr(package, export)] == []


@pytest.mark.parametrize("name", ["entoscope", *SUBPACKAGES])
def test_the_exports_are_sorted_and_unique(name):
    exports = importlib.import_module(name).__all__
    assert exports == sorted(set(exports))


def test_the_top_level_reexports_what_a_subpackage_exports():
    # One public home per name below the top level, so a name cannot drift between the two spellings.
    homes = {}
    for name in SUBPACKAGES:
        package = importlib.import_module(name)
        homes.update({export: getattr(package, export) for export in package.__all__})
    for export in entoscope.__all__:
        if export != "__version__":
            assert getattr(entoscope, export) is homes[export], export
