# semantic.py
"""Semantic detection: role, aggregation, confidence per column."""
import pandas as pd
import numpy as np

RATE_KEYWORDS = ['price', 'rate', 'ratio', 'percent', 'pct', 'margin', 'unitprice', 'asp']
AMOUNT_KEYWORDS = ['amount', 'revenue', 'sales', 'total', 'grossamount', 'netamount', 'discountamount', 'cost']
QTY_KEYWORDS = ['qty', 'quantity', 'count', 'units']

def detect_role(col_name: str, dtype: str, series: pd.Series) -> tuple[str, str | None]:
    name = col_name.lower()

    if dtype.startswith('datetime'):
        return 'time', None

    if dtype in ('int64', 'float64'):
        # discount: check if ratio-like (0-1) or amount-like (raw currency)
        if 'discount' in name:
            non_null = series.dropna()
            if len(non_null) > 0 and non_null.max() <= 1.0 and 'amount' not in name:
                return 'rate', 'avg'
            if 'amount' in name:
                return 'metric', 'sum'
            return 'rate', 'avg'

        if any(kw in name for kw in RATE_KEYWORDS):
            return 'rate', 'avg'

        if any(kw in name for kw in QTY_KEYWORDS):
            return 'metric', 'sum'

        if any(kw in name for kw in AMOUNT_KEYWORDS):
            return 'metric', 'sum'

        # fallback: high cardinality relative to row count suggests ID, not metric
        cardinality_ratio = series.nunique() / max(len(series), 1)
        if cardinality_ratio > 0.95:
            return 'dimension', None

        return 'metric', 'sum'

    return 'dimension', None


def calc_confidence(col_name: str, dtype: str, role: str, series: pd.Series, n_rows: int) -> float:
    score = 0.5
    name = col_name.lower()
    cardinality = series.nunique()

    if dtype.startswith('datetime'):
        score += 0.3
        return round(min(score, 0.99), 2)

    if role == 'metric':
        if dtype in ('int64', 'float64'):
            score += 0.15
        if any(kw in name for kw in AMOUNT_KEYWORDS + QTY_KEYWORDS):
            score += 0.25
        if cardinality / n_rows > 0.9:
            score -= 0.15  # likely mis-detected ID

    elif role == 'rate':
        if any(kw in name for kw in RATE_KEYWORDS) or 'discount' in name:
            score += 0.3
        non_null = series.dropna()
        if len(non_null) > 0 and non_null.max() <= 1.0:
            score += 0.1

    elif role == 'dimension':
        if dtype == 'object' or dtype == 'str':
            score += 0.2
        if cardinality / n_rows < 0.5:
            score += 0.1

    return round(max(min(score, 0.99), 0.3), 2)


def profile_dataframe(df: pd.DataFrame) -> list[dict]:
    n_rows = len(df)
    results = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        unique = int(series.nunique())

        role, agg = detect_role(col, dtype, series)
        confidence = calc_confidence(col, dtype, role, series, n_rows)

        print(f"[AI-DEBUG] profile_dataframe: col={col} dtype={dtype} role={role} agg={agg} conf={confidence}")

        results.append({
            'column': col,
            'dtype': dtype,
            'role': role,
            'aggregation': agg,
            'confidence': confidence,
            'missing': missing,
            'missing_pct': round(missing / n_rows * 100, 1) if n_rows else 0,
            'unique': unique,
        })

    return results
