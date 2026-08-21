# detection.py
"""M3: Anomaly + threshold-based detection. Rule-based, no ML."""
import pandas as pd
import numpy as np

def detect_anomaly(series: pd.Series, std_threshold: float = 2.0) -> list[dict]:
    """Flag points beyond N std devs from rolling mean."""
    if len(series) < 3:
        return []
    mean = series.mean()
    std = series.std()
    if std == 0:
        return []

    anomalies = []
    for i, val in series.items():
        z = (val - mean) / std
        if abs(z) > std_threshold:
            anomalies.append({
                'index': i,
                'value': val,
                'z_score': round(z, 2),
                'direction': 'spike' if z > 0 else 'drop',
            })
    print(f"[AI-DEBUG] detect_anomaly: n={len(series)} threshold={std_threshold} found={len(anomalies)}")
    return anomalies


def detect_trend_break(series: pd.Series, window: int = 3, pct_threshold: float = 0.15) -> list[dict]:
    """Compare recent window avg vs prior window avg; flag if change exceeds threshold."""
    if len(series) < window * 2:
        return []
    breaks = []
    for i in range(window * 2, len(series) + 1):
        prior = series.iloc[i - window * 2:i - window].mean()
        recent = series.iloc[i - window:i].mean()
        if prior == 0:
            continue
        pct_change = (recent - prior) / prior
        if abs(pct_change) > pct_threshold:
            breaks.append({
                'index': i - 1,
                'prior_avg': round(prior, 2),
                'recent_avg': round(recent, 2),
                'pct_change': round(pct_change * 100, 2),
            })
    print(f"[AI-DEBUG] detect_trend_break: n={len(series)} window={window} found={len(breaks)}")
    return breaks


def root_cause(engine, metric: str, dimensions: list[str], top_n: int = 3) -> dict:
    """
    Drill into which dimension members drove the overall change.
    Requires engine.variance() to compare two periods per dimension.
    """
    drivers = []
    for dim in dimensions:
        try:
            df = engine.contribution(metric, dim)
            df_sorted = df.reindex(df[metric].abs().sort_values(ascending=False).index)
            top = df_sorted.head(top_n)
            for _, row in top.iterrows():
                drivers.append({
                    'dimension': dim,
                    'member': row[dim],
                    'value': row[metric],
                    'contribution_pct': row.get('contribution_pct', None),
                })
        except Exception as e:
            print(f"[AI-DEBUG] root_cause: dim={dim} error={e}")
            continue

    drivers_sorted = sorted(drivers, key=lambda d: abs(d.get('contribution_pct') or 0), reverse=True)
    print(f"[AI-DEBUG] root_cause: metric={metric} dims={dimensions} drivers_found={len(drivers_sorted)}")
    return {
        'metric': metric,
        'drivers': drivers_sorted[:top_n * len(dimensions)],
    }
