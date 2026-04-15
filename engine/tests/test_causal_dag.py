# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-004-01 through E-004-12 — CausalDAG tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

from synthflow.causal_dag import CausalDAG, CycleError, UnsafeLambdaError
from synthflow.models.schemas import CausalDagRule


# ── Fixture helpers ────────────────────────────────────────────────────────

def _rule(parent: str, child: str, expr: str = 'lambda row, rng: row["{p}"]') -> CausalDagRule:
    """Shorthand for building a CausalDagRule."""
    lam = f'lambda row, rng: row["{parent}"]'
    return CausalDagRule(parent_column=parent, child_column=child, lambda_str=lam)


def _dag_from(*pairs: tuple[str, str]) -> CausalDAG:
    """Build a CausalDAG from (parent, child) pairs with identity lambdas."""
    rules = [_rule(p, c) for p, c in pairs]
    dag = CausalDAG()
    dag.build_from_rules(rules)
    return dag


RNG = np.random.default_rng(42)


# ── E-004-01: Correct adjacency ───────────────────────────────────────────

def test_dag_builds_correct_adjacency_from_rules() -> None:
    """3 rules A→B, B→C, A→C create the correct adjacency list."""
    rules = [
        CausalDagRule(parent_column="a", child_column="b",
                      lambda_str='lambda row, rng: row["a"] + 1'),
        CausalDagRule(parent_column="b", child_column="c",
                      lambda_str='lambda row, rng: row["b"] + 1'),
        CausalDagRule(parent_column="a", child_column="c",
                      lambda_str='lambda row, rng: row["a"] * 2'),
    ]
    dag = CausalDAG()
    dag.build_from_rules(rules)

    adj = dag.adjacency
    assert "b" in adj["a"]
    assert "c" in adj["a"]
    assert "c" in adj["b"]
    assert dag.nodes == {"a", "b", "c"}


# ── E-004-02: Topological sort parents first ──────────────────────────────

def test_dag_topological_sort_parents_before_children() -> None:
    """In sort order every parent appears before its children: A→B→C."""
    dag = _dag_from(("a", "b"), ("b", "c"))
    order = dag.topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_dag_topological_sort_contains_all_nodes() -> None:
    """Sort result contains every node exactly once."""
    dag = _dag_from(("x", "y"), ("y", "z"), ("x", "z"))
    order = dag.topological_sort()
    assert set(order) == {"x", "y", "z"}
    assert len(order) == 3


# ── E-004-03: Cycle detection ─────────────────────────────────────────────

def test_dag_raises_on_cycle_detection() -> None:
    """A→B→C→A raises CycleError."""
    rules = [
        CausalDagRule(parent_column="a", child_column="b",
                      lambda_str='lambda row, rng: row["a"]'),
        CausalDagRule(parent_column="b", child_column="c",
                      lambda_str='lambda row, rng: row["b"]'),
        CausalDagRule(parent_column="c", child_column="a",
                      lambda_str='lambda row, rng: row["c"]'),
    ]
    dag = CausalDAG()
    dag.build_from_rules(rules)
    with pytest.raises(CycleError):
        dag.topological_sort()


# ── E-004-04: Age from birth year ─────────────────────────────────────────

def test_dag_resolves_age_from_birthyear() -> None:
    """Lambda '2024 - birth_year' produces correct integer age."""
    rule = CausalDagRule(
        parent_column="birth_year",
        child_column="age",
        lambda_str='lambda row, rng: 2024 - row["birth_year"]',
    )
    dag = CausalDAG()
    dag.build_from_rules([rule])

    row = dag.resolve_row({"birth_year": 1990}, RNG)
    assert row["age"] == 34


def test_dag_resolves_chain() -> None:
    """Two chained rules are applied in order: birth_year→age→is_senior."""
    rules = [
        CausalDagRule(
            parent_column="birth_year", child_column="age",
            lambda_str='lambda row, rng: 2024 - row["birth_year"]',
        ),
        CausalDagRule(
            parent_column="age", child_column="is_senior",
            lambda_str='lambda row, rng: row["age"] >= 60',
        ),
    ]
    dag = CausalDAG()
    dag.build_from_rules(rules)

    row = dag.resolve_row({"birth_year": 1950}, RNG)
    assert row["age"] == 74
    assert row["is_senior"] is True


# ── E-004-05: Safe lambda math ────────────────────────────────────────────

def test_dag_safe_lambda_allows_math_operations() -> None:
    """'lambda row, rng: row["age"] * 12' compiles and runs."""
    rule = CausalDagRule(
        parent_column="age",
        child_column="age_months",
        lambda_str='lambda row, rng: row["age"] * 12',
    )
    dag = CausalDAG()
    dag.build_from_rules([rule])

    row = dag.resolve_row({"age": 30}, RNG)
    assert row["age_months"] == 360


def test_dag_safe_lambda_allows_abs_and_round() -> None:
    """abs() and round() are allowed in safe lambdas."""
    fn = CausalDAG()._compile_lambda('lambda row, rng: round(abs(row["x"]), 2)')
    result = fn({"x": -3.14159}, RNG)
    assert abs(result - 3.14) < 0.001


# ── E-004-06: Import blocked ──────────────────────────────────────────────

def test_dag_rejects_lambda_with_import_via_dunder() -> None:
    """'lambda row, rng: __import__(\"os\")…' raises UnsafeLambdaError."""
    dag = CausalDAG()
    with pytest.raises(UnsafeLambdaError):
        dag._compile_lambda('lambda row, rng: __import__("os").system("ls")')


# ── E-004-07: exec blocked ────────────────────────────────────────────────

def test_dag_rejects_lambda_with_exec_call() -> None:
    """'lambda row, rng: exec(\"print(1)\")' raises UnsafeLambdaError."""
    dag = CausalDAG()
    with pytest.raises(UnsafeLambdaError):
        dag._compile_lambda('lambda row, rng: exec("print(1)")')


def test_dag_rejects_lambda_with_eval_call() -> None:
    """eval() in a lambda raises UnsafeLambdaError."""
    dag = CausalDAG()
    with pytest.raises(UnsafeLambdaError):
        dag._compile_lambda('lambda row, rng: eval("1+1")')


# ── E-004-08: Dunder attribute blocked ────────────────────────────────────

def test_dag_rejects_lambda_with_dunder_attribute() -> None:
    """'row.__class__.__bases__' raises UnsafeLambdaError."""
    dag = CausalDAG()
    with pytest.raises(UnsafeLambdaError):
        dag._compile_lambda('lambda row, rng: row.__class__.__bases__')


def test_dag_rejects_lambda_with_dunder_dict() -> None:
    """'row.__dict__' raises UnsafeLambdaError."""
    dag = CausalDAG()
    with pytest.raises(UnsafeLambdaError):
        dag._compile_lambda('lambda row, rng: row.__dict__')


# ── E-004-09: Missing parents skipped ────────────────────────────────────

def test_dag_resolve_row_skips_columns_with_missing_parents() -> None:
    """If parent is not in partial_row, the child column is not added."""
    rule = CausalDagRule(
        parent_column="salary",
        child_column="bonus",
        lambda_str='lambda row, rng: row["salary"] * 0.1',
    )
    dag = CausalDAG()
    dag.build_from_rules([rule])

    # salary is absent — bonus should not be computed
    row = dag.resolve_row({"name": "Alice"}, RNG, columns_to_resolve=["bonus"])
    assert "bonus" not in row


# ── E-004-10: Empty rules → empty DAG ────────────────────────────────────

def test_dag_empty_rules_returns_empty_dag() -> None:
    """No rules → empty adjacency and empty topological sort."""
    dag = CausalDAG()
    dag.build_from_rules([])
    assert dag.adjacency == {}
    assert dag.topological_sort() == []
    assert dag.nodes == set()


# ── E-004-11: Multiple roots handled ─────────────────────────────────────

def test_dag_multiple_roots_both_before_shared_child() -> None:
    """A→C and B→C: both A and B appear before C in sort order."""
    rules = [
        CausalDagRule(parent_column="a", child_column="c",
                      lambda_str='lambda row, rng: row["a"] + 1'),
        CausalDagRule(parent_column="b", child_column="c",
                      lambda_str='lambda row, rng: row["b"] + 2'),
    ]
    dag = CausalDAG()
    dag.build_from_rules(rules)
    order = dag.topological_sort()

    assert "a" in order
    assert "b" in order
    assert "c" in order
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("c")


# ── E-004-12: numpy allowed ───────────────────────────────────────────────

def test_dag_lambda_with_numpy_allowed() -> None:
    """'lambda row, rng: np.sqrt(row["x"])' compiles and runs correctly."""
    rule = CausalDagRule(
        parent_column="x",
        child_column="sqrt_x",
        lambda_str='lambda row, rng: np.sqrt(row["x"])',
    )
    dag = CausalDAG()
    dag.build_from_rules([rule])

    row = dag.resolve_row({"x": 16.0}, RNG)
    assert abs(row["sqrt_x"] - 4.0) < 1e-9


def test_dag_lambda_with_numpy_clip() -> None:
    """np.clip() is accessible in safe lambdas."""
    fn = CausalDAG()._compile_lambda('lambda row, rng: float(np.clip(row["v"], 0, 100))')
    assert fn({"v": 150}, RNG) == 100.0
    assert fn({"v": -5}, RNG) == 0.0
