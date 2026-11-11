"""The public surface of the package and the type information it carries."""

import ast
from importlib.resources import files

import parkin
from parkin import _core


def test_every_public_name_is_importable() -> None:
    assert sorted(parkin.__all__) == parkin.__all__
    for name in parkin.__all__:
        assert hasattr(parkin, name)


def test_package_is_marked_as_typed() -> None:
    assert files("parkin").joinpath("py.typed").is_file()


def test_stub_declares_exactly_the_bound_functions() -> None:
    stub_text = files("parkin").joinpath("_core.pyi").read_text(encoding="utf-8")
    declared_names = {
        node.name
        for node in ast.parse(stub_text).body
        if isinstance(node, ast.FunctionDef)
    }
    bound_names = {name for name in dir(_core) if not name.startswith("_")}
    assert declared_names == bound_names
