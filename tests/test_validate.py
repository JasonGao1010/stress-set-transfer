import math

from analysis.validate import _close, _probabilities, _top


def test_validator_top_reconstructs_lexical_tie_break() -> None:
    assert _top({"b": 1.0, "a": 1.0, "c": 0.0}, 1) == {"a"}


def test_validator_numeric_comparison_handles_none() -> None:
    assert _close(None, None)
    assert not _close(None, 0.0)
    assert _close(0.1 + 0.2, 0.3)


def test_validator_near_boundary_partition_selects_exactly_k() -> None:
    probabilities = _probabilities(
        {"a": 1.0 + 5e-10, "b": 1.0, "c": 1.0, "d": 0.0},
        2,
    )
    assert probabilities == {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3, "d": 0.0}
    assert math.isclose(sum(probabilities.values()), 2.0)


def test_validator_uses_the_declared_strict_tie_tolerance() -> None:
    probabilities = _probabilities(
        {"a": 1.0 + 5e-7, "b": 1.0, "c": 1.0, "d": 0.0},
        2,
    )
    assert probabilities == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}


def test_validator_matches_the_tie_formula_boundary() -> None:
    probabilities = _probabilities(
        {"a": 1.0 + 1.0005e-9, "b": 1.0, "c": 1.0, "d": 0.0},
        2,
    )
    assert probabilities == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}
