# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Causal DAG — build, topological sort, safe lambda compilation
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import ast
import builtins
from collections import defaultdict, deque
from typing import Any, Callable, Optional

import numpy as np

from synthflow.models.schemas import CausalDagRule


# ── Exceptions ─────────────────────────────────────────────────────────────

class CycleError(ValueError):
    """Raised when the causal DAG contains a cycle (Kahn's algorithm detects it)."""


class UnsafeLambdaError(ValueError):
    """Raised when a lambda string fails the AST whitelist safety check."""


# ── Safety configuration ───────────────────────────────────────────────────

# Functions from builtins that are safe to expose inside lambdas
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs", "bool", "dict", "enumerate", "float", "hasattr",
        "int", "isinstance", "len", "list", "max", "min",
        "range", "round", "sorted", "str", "sum", "zip",
    }
)

# Function *call* names that are always forbidden
_FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "__import__", "breakpoint", "compile", "delattr", "dir",
        "eval", "exec", "exit", "getattr", "globals", "input",
        "locals", "open", "quit", "setattr", "vars",
    }
)


def _build_safe_globals() -> dict[str, Any]:
    """Return a minimal globals dict for eval'ing untrusted lambdas."""
    safe: dict[str, Any] = {"__builtins__": {}, "np": np}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    return safe


_SAFE_GLOBALS: dict[str, Any] = _build_safe_globals()


# ── AST safety visitor ─────────────────────────────────────────────────────

class _SafetyVisitor(ast.NodeVisitor):
    """Walk the AST and raise UnsafeLambdaError on any forbidden construct."""

    def visit_Import(self, node: ast.Import) -> None:
        raise UnsafeLambdaError("import statements are forbidden in lambdas")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise UnsafeLambdaError("from-import statements are forbidden in lambdas")

    def visit_Global(self, node: ast.Global) -> None:
        raise UnsafeLambdaError("global statements are forbidden in lambdas")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise UnsafeLambdaError("nonlocal statements are forbidden in lambdas")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise UnsafeLambdaError(
                f"Call to '{node.func.id}' is forbidden in lambdas"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise UnsafeLambdaError(
                f"Access to dunder attribute '{node.attr}' is forbidden in lambdas"
            )
        self.generic_visit(node)


# ── CausalDAG ──────────────────────────────────────────────────────────────

class CausalDAG:
    """
    Directed Acyclic Graph representing causal dependencies between columns.

    Workflow::

        dag = CausalDAG()
        dag.build_from_rules(rules)          # parse + compile lambdas
        order = dag.topological_sort()       # Kahn's algorithm
        row = dag.resolve_row(partial, rng)  # apply compiled lambdas
    """

    def __init__(self) -> None:
        # parent → [children]
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        # (parent, child) → compiled callable
        self._fns: dict[tuple[str, str], Callable[..., Any]] = {}
        self._nodes: set[str] = set()

    # ── Construction ──────────────────────────────────────────────────

    def build_from_rules(self, rules: list[CausalDagRule]) -> None:
        """
        Populate the DAG from *rules*.

        Each rule's lambda is compiled eagerly so errors surface immediately.

        Args:
            rules: List of CausalDagRule objects.
        """
        self._adjacency.clear()
        self._fns.clear()
        self._nodes.clear()

        for rule in rules:
            p, c = rule.parent_column, rule.child_column
            self._adjacency[p].append(c)
            self._nodes.update((p, c))
            self._fns[(p, c)] = self._compile_lambda(rule.lambda_str)

    # ── Topological sort (Kahn's algorithm) ───────────────────────────

    def topological_sort(self) -> list[str]:
        """
        Return all nodes in topological order.

        Raises:
            CycleError: If the graph contains a cycle.

        Returns:
            Ordered list of node names (parents before children).
        """
        if not self._nodes:
            return []

        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for parent, children in self._adjacency.items():
            for child in children:
                in_degree[child] = in_degree.get(child, 0) + 1

        queue: deque[str] = deque(
            sorted(n for n, d in in_degree.items() if d == 0)
        )
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for child in sorted(self._adjacency.get(node, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self._nodes):
            remaining = self._nodes - set(result)
            raise CycleError(
                f"Cycle detected in causal DAG. Could not resolve nodes: {remaining}"
            )
        return result

    # ── Lambda compilation ────────────────────────────────────────────

    def _compile_lambda(self, lambda_str: str) -> Callable[..., Any]:
        """
        Validate and compile *lambda_str* using AST whitelisting.

        Safe builtins: abs, bool, dict, enumerate, float, hasattr, int,
        isinstance, len, list, max, min, range, round, sorted, str, sum, zip.
        numpy is available as ``np``.

        Forbidden: Import/ImportFrom/Global/Nonlocal AST nodes;
        calls to exec, eval, open, __import__, getattr, setattr, …;
        any dunder attribute access.

        Args:
            lambda_str: Python source string, e.g. ``'lambda row, rng: row["age"] * 12'``.

        Returns:
            Compiled callable.

        Raises:
            UnsafeLambdaError: If any forbidden construct is found.
        """
        try:
            tree = ast.parse(lambda_str, mode="eval")
        except SyntaxError as exc:
            raise UnsafeLambdaError(f"Syntax error in lambda: {exc}") from exc

        _SafetyVisitor().visit(tree)

        try:
            code = compile(tree, "<synthflow_lambda>", "eval")
            fn: Callable[..., Any] = eval(code, _SAFE_GLOBALS)  # noqa: S307
        except Exception as exc:
            raise UnsafeLambdaError(f"Failed to compile lambda: {exc}") from exc

        return fn

    # ── Row resolution ─────────────────────────────────────────────────

    def resolve_row(
        self,
        partial_row: dict[str, Any],
        rng: np.random.Generator,
        columns_to_resolve: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Walk the DAG in topological order, applying lambdas to derive new values.

        If a parent column is missing from *partial_row* the dependent child is
        silently skipped (it will not appear in the returned dict).

        Args:
            partial_row:        Input row; may be incomplete.
            rng:                numpy random Generator for stochastic lambdas.
            columns_to_resolve: Restrict resolution to these columns. None = all.

        Returns:
            Updated row dict (shallow copy of *partial_row* with new keys added).
        """
        row = dict(partial_row)
        resolve_set = set(columns_to_resolve) if columns_to_resolve is not None else self._nodes

        for node in self.topological_sort():
            if node not in resolve_set:
                continue
            # Find the rule(s) where this node is the child
            for (parent, child), fn in self._fns.items():
                if child != node:
                    continue
                if parent not in row:
                    # Parent not resolved yet — skip this derivation
                    continue
                try:
                    row[child] = fn(row, rng)
                except Exception:
                    # Lambda failure is non-fatal during generation
                    pass

        return row

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Shallow copy of the adjacency list."""
        return {k: list(v) for k, v in self._adjacency.items()}

    @property
    def nodes(self) -> set[str]:
        """All node names in the DAG."""
        return set(self._nodes)
