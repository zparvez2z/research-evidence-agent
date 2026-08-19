import pytest

from research_agent.tools import calculate


def test_calculate_normal_arithmetic() -> None:
    assert calculate("2 + 3 * 4 - 5 % 2") == 13


def test_calculate_parentheses() -> None:
    assert calculate("(2 + 3) * 4") == 20


def test_calculate_unary_values_and_power() -> None:
    assert calculate("-2 ** 2 + +10") == 6


@pytest.mark.parametrize(
    "expression",
    ["name + 1", "open('file')", "thing.value", "items[0]", "'text'", "[1, 2]"],
)
def test_calculate_rejects_unsafe_syntax(expression: str) -> None:
    with pytest.raises(ValueError, match="Unsupported arithmetic syntax"):
        calculate(expression)


def test_calculate_reports_division_by_zero() -> None:
    with pytest.raises(ValueError, match="zero"):
        calculate("4 / 0")
