"""A deliberately small AST-based arithmetic calculator."""

import ast
import operator
from collections.abc import Callable


Number = int | float
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        operation = _BINARY_OPERATORS[type(node.op)]
        return operation(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operation = _UNARY_OPERATORS[type(node.op)]
        return operation(_evaluate(node.operand))
    raise ValueError(f"Unsupported arithmetic syntax: {type(node).__name__}")


def calculate(expression: str) -> Number:
    """Calculate an arithmetic expression without executing Python code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-blank string")
    try:
        tree = ast.parse(expression, mode="eval")
        return _evaluate(tree.body)
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic expression") from error
    except ZeroDivisionError as error:
        raise ValueError("Division or modulo by zero is not allowed") from error
