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
    if not cleaning_plan or "pandas" not in cleaning_plan:
        return df

    steps = (cleaning_plan.get("pandas") or {}).get("steps") or []
    if not isinstance(steps, list):
        logger.warning("cleaning_plan.pandas.steps is not a list; skipping.")
        return df

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
            df_out = _step_fillna_constant(df, params.get("columns", []), params.get("value"))

        elif name == "fillna_numeric":
            strategy = params.get("strategy", "constant")
            if strategy != "constant":
                logger.warning("Unsupported strategy '%s' for fillna_numeric; only 'constant' is supported.", strategy)
                continue
            df_out = _step_fillna_constant(df, params.get("columns", []), params.get("value"))

        elif name == "clip_values":
            df_out = _step_clip_values(
                df_out,
                params.get("columns", []),
                params.get("min", None),
                params.get("max", None),
            )

        else:
            logger.warning("Unknown step '%s'; skipping.", name)

    return df

# --------------
# Orchestration
# --------------
def process(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Orchestrate the pipeline:
      1) Build mask via filter_ast and produce df_filtered (copy).
      2) Apply cleaning_plan on df_filtered to produce df_cleaned (copy).
      Returns (df_filtered, df_cleaned).
    The original df is never mutated.
    """

    cleaning_plan = (config or {}).get("cleaning_plan")
    df = run_cleaning_plan(df, cleaning_plan)

    return df

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
