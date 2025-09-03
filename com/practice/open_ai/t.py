from __future__ import annotations

import logging
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Logging setup (tweak as needed) ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

# -------------------------
# Filter AST implementation
# -------------------------
def _safe_boolean_series(df: pd.DataFrame, fill: bool) -> pd.Series:
    """Return a boolean Series of len(df) filled with `fill`."""
    return pd.Series([fill] * len(df), index=df.index, dtype=bool)

def _eval_single_predicate(df: pd.DataFrame, field: str, cmp: str, value: Any) -> pd.Series:
    """
    Evaluate a single predicate like city in ["Delhi","Mumbai"] or age >= 18.
    If field is missing, return an all-False mask and warn.
    """
    if field not in df.columns:
        logger.warning("Predicate skipped: column '%s' not found.", field)
        return _safe_boolean_series(df, False)

    s = df[field]
    cmp = (cmp or "").lower()

    try:
        if cmp == "in":
            if value is None:
                logger.warning("Predicate 'in' skipped: value is None.")
                return _safe_boolean_series(df, False)
            return s.isin(list(value))
        elif cmp in {"not_in", "nin"}:
            if value is None:
                logger.warning("Predicate 'not_in' skipped: value is None.")
                return _safe_boolean_series(df, False)
            return ~s.isin(list(value))
        elif cmp in {"==", "eq"}:
            return s == value
        elif cmp in {"!=", "ne"}:
            return s != value
        elif cmp in {">", "gt"}:
            return s > value
        elif cmp in {">=", "ge"}:
            return s >= value
        elif cmp in {"<", "lt"}:
            return s < value
        elif cmp in {"<=", "le"}:
            return s <= value
        else:
            logger.warning("Unknown comparator '%s'; predicate skipped.", cmp)
            return _safe_boolean_series(df, False)
    except Exception as e:
        logger.warning("Predicate evaluation error on column '%s' with cmp '%s': %s", field, cmp, e)
        return _safe_boolean_series(df, False)

def apply_filter_ast(df: pd.DataFrame, ast: Optional[Dict[str, Any]]) -> pd.Series:
    """
    Build a boolean mask from a filter AST.

    AST structure (keys are optional/nullable):
      - op: "AND" | "OR" | "NOT" (default = "AND")
      - field: str
      - cmp: "in" | "not_in" | "==" | "!=" | ">" | ">=" | "<" | "<="
      - value: Any (list for 'in'/'not_in')
      - children: list[AST]

    Rules:
      - If both a local predicate (field/cmp/value) and children exist, combine
        ALL masks according to 'op'.
      - If AST is None/empty, returns an all-True mask (no filtering).
    """
    if ast is None or not isinstance(ast, dict) or len(df) == 0:
        return _safe_boolean_series(df, True)

    op = (ast.get("op") or "AND").upper()
    field = ast.get("field")
    cmp = ast.get("cmp")
    value = ast.get("value")
    children = ast.get("children") or []

    masks: List[pd.Series] = []

    # Optional local predicate
    if field is not None and cmp is not None:
        masks.append(_eval_single_predicate(df, field, cmp, value))

    # Children
    for child in children:
        masks.append(apply_filter_ast(df, child))

    # Combine masks
    if op == "NOT":
        # For NOT, invert combined AND of children/predicate (or True if none)
        base = reduce(lambda a, b: a & b, masks, _safe_boolean_series(df, True)) if masks else _safe_boolean_series(df, True)
        return ~base

    if not masks:
        # No predicate and no children: default to True (no-op filter)
        return _safe_boolean_series(df, True)

    if op == "AND":
        return reduce(lambda a, b: a & b, masks)
    elif op == "OR":
        return reduce(lambda a, b: a | b, masks)
    else:
        logger.warning("Unknown logical op '%s'; defaulting to AND.", op)
        return reduce(lambda a, b: a & b, masks)

# ---------------------------
# Cleaning plan implementation
# ---------------------------
def _existing_columns(df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
    if not columns:
        return []
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("Skipping missing column(s): %s", missing)
    return [c for c in columns if c in df.columns]

def _step_fillna_constant(df: pd.DataFrame, columns: List[str], value: Any) -> pd.DataFrame:
    if value is None:
        logger.warning("fillna 'constant' skipped: value is None.")
        return df
    for col in _existing_columns(df, columns):
        df[col] = df[col].fillna(value)
    return df

def _step_clip_values(df: pd.DataFrame, columns: List[str], min_val: Optional[float], max_val: Optional[float]) -> pd.DataFrame:
    for col in _existing_columns(df, columns):
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=min_val, upper=max_val)
        except Exception as e:
            logger.warning("clip_values failed for column '%s': %s", col, e)
    return df

def run_cleaning_plan(df: pd.DataFrame, cleaning_plan: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Execute cleaning_plan['pandas']['steps'] sequentially on a COPY of df.
    Recognized steps:
      - {"fillna_categorical": {"columns": [...], "strategy": "constant", "value": ...}}
      - {"fillna_numeric":     {"columns": [...], "strategy": "constant", "value": ...}}
      - {"clip_values":        {"columns": [...], "min": <num|None>, "max": <num|None>}}
    """
    df_out = df.copy(deep=True)
    if not cleaning_plan or "pandas" not in cleaning_plan:
        return df_out

    steps = (cleaning_plan.get("pandas") or {}).get("steps") or []
    if not isinstance(steps, list):
        logger.warning("cleaning_plan.pandas.steps is not a list; skipping.")
        return df_out

    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            logger.warning("Malformed step '%s'; skipping.", step)
            continue

        name, params = list(step.items())[0]
        params = params or {}

        if name == "fillna_categorical":
            strategy = params.get("strategy", "constant")
            if strategy != "constant":
                logger.warning("Unsupported strategy '%s' for fillna_categorical; only 'constant' is supported.", strategy)
                continue
            df_out = _step_fillna_constant(df_out, params.get("columns", []), params.get("value"))

        elif name == "fillna_numeric":
            strategy = params.get("strategy", "constant")
            if strategy != "constant":
                logger.warning("Unsupported strategy '%s' for fillna_numeric; only 'constant' is supported.", strategy)
                continue
            df_out = _step_fillna_constant(df_out, params.get("columns", []), params.get("value"))

        elif name == "clip_values":
            df_out = _step_clip_values(
                df_out,
                params.get("columns", []),
                params.get("min", None),
                params.get("max", None),
            )

        else:
            logger.warning("Unknown step '%s'; skipping.", name)

    return df_out

# --------------
# Orchestration
# --------------
def process(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the pipeline:
      1) Build mask via filter_ast and produce df_filtered (copy).
      2) Apply cleaning_plan on df_filtered to produce df_cleaned (copy).
      Returns (df_filtered, df_cleaned).
    The original df is never mutated.
    """
    ast = (config or {}).get("filter_ast")
    mask = apply_filter_ast(df, ast)
    df_filtered = df.loc[mask].copy()

    cleaning_plan = (config or {}).get("cleaning_plan")
    df_cleaned = run_cleaning_plan(df_filtered, cleaning_plan)

    return df_filtered, df_cleaned

# ----------------
# Example (comment)
# ----------------
# config = {
#   "filter_ast": {"op": "OR", "field": "city", "cmp": "in", "value": ["Delhi","Mumbai"], "children": []},
#   "targets": {"numeric": ["age"], "categorical": ["department","city"], "datetime": [], "text": []},
#   "cleaning_plan": {
#     "pandas": {"steps": [
#       {"fillna_categorical": {"columns": ["department"], "strategy": "constant", "value": "IT"}},
#       {"fillna_numeric": {"columns": ["age"], "strategy": "constant", "value": 0}},
#       {"clip_values": {"columns": ["age"], "min": 0, "max": None}}
#     ]}
#   }
# }
# df_filtered, df_cleaned = process(df, config)
