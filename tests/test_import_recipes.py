import pytest


def test_import_receipes():
    with pytest.raises(ModuleNotFoundError, match="No module named 'recipes'"):
        import recipes
