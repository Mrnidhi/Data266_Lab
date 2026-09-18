import pytest
from lab1.common import apply_overrides


def test_nested_json_overrides_preserve_types():
    config = {"epochs": 12, "data": {"path": None}}
    result = apply_overrides(config, ["epochs=2", 'data.path="images/monet"', "flag=false"])
    assert result == {"epochs": 2, "data": {"path": "images/monet"}, "flag": False}


def test_reject_descent_through_scalar():
    with pytest.raises(ValueError):
        apply_overrides({"epochs": 12}, ["epochs.bad=2"])
