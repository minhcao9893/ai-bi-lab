# analytics.py
"""M2: Analytics Engine — DuckDB backed, query directly on parquet/csv."""
import duckdb
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/data/uploads")


class AnalyticsEngine:
    def __init__(self, dataset_path: str, catalog: list[dict]):
        self.con = duckdb.connect(database=':memory:')
        self.dataset_path = dataset_path
        self.catalog = {c['column']: c for c in catalog}
        self._load()

    def _load(self):
        ext = Path(self.dataset_path).suffix
        if ext == '.csv':
            self.con.execute(f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{self.dataset_path}')")
        elif ext in ('.xlsx', '.xls'):
            # duckdb has no native excel reader; convert via pandas once, cache as parquet
            parquet_path = self.dataset_path + '.parquet'
            if not Path(parquet_path).exists():
                df = pd.read_excel(self.dataset_path)
                df.to_parquet(parquet_path)
            self.con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{parquet_path}')")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        print(f"[AI-DEBUG] AnalyticsEngine._load: path={self.dataset_path} ext={ext}")

    def _metric_cols(self) -> list[str]:
        return [c for c, meta in self.catalog.items() if meta['role'] == 'metric']

    def _dimension_cols(self) -> list[str]:
        return [c for c, meta in self.catalog.items() if meta['role'] == 'dimension']

    def _time_col(self) -> str | None:
        for c, meta in self.catalog.items():
            if meta['role'] == 'time':
                return c
        return None

    def total(self, metric: str) -> float:
        agg = self.catalog[metric]['aggregation'] or 'sum'
        sql = f"SELECT {agg}({metric}) AS val FROM data"
        print(f"[AI-DEBUG] total: sql={sql}")
        return self.con.execute(sql).fetchone()[0]

    def by_dimension(self, metric: str, dimension: str, limit: int = 10, ascending: bool = False) -> pd.DataFrame:
        agg = self.catalog[metric]['aggregation'] or 'sum'
        order = 'ASC' if ascending else 'DESC'
        sql = f"""
            SELECT {dimension}, {agg}({metric}) AS {metric}
            FROM data
            GROUP BY {dimension}
            ORDER BY {metric} {order}
            LIMIT {limit}
        """
        print(f"[AI-DEBUG] by_dimension: metric={metric} dim={dimension} limit={limit} asc={ascending}")
        return self.con.execute(sql).fetchdf()

    def time_series(self, metric: str, granularity: str = 'month') -> pd.DataFrame:
        time_col = self._time_col()
        if not time_col:
            raise ValueError("No time column detected in catalog")
        agg = self.catalog[metric]['aggregation'] or 'sum'
        sql = f"""
            SELECT date_trunc('{granularity}', {time_col}) AS period,
                   {agg}({metric}) AS {metric}
            FROM data
            GROUP BY period
            ORDER BY period
        """
        print(f"[AI-DEBUG] time_series: metric={metric} time_col={time_col} gran={granularity}")
        return self.con.execute(sql).fetchdf()

    def mom_yoy(self, metric: str) -> pd.DataFrame:
        """Month-over-month and year-over-year % change."""
        ts = self.time_series(metric, granularity='month')
        ts = ts.sort_values('period').reset_index(drop=True)
        ts['mom_pct'] = ts[metric].pct_change(periods=1) * 100
        ts['yoy_pct'] = ts[metric].pct_change(periods=12) * 100
        print(f"[AI-DEBUG] mom_yoy: metric={metric} rows={len(ts)}")
        return ts

    def contribution(self, metric: str, dimension: str) -> pd.DataFrame:
        df = self.by_dimension(metric, dimension, limit=1000)
        total = df[metric].sum()
        df['contribution_pct'] = (df[metric] / total * 100).round(2) if total else 0
        print(f"[AI-DEBUG] contribution: metric={metric} dim={dimension} total={total}")
        return df

    def variance(self, metric: str, dimension: str, period_col: str, period_a, period_b) -> pd.DataFrame:
        """Compare metric by dimension between two period values (e.g. two months)."""
        time_col = self._time_col()
        agg = self.catalog[metric]['aggregation'] or 'sum'
        sql = f"""
            SELECT {dimension},
                   sum(CASE WHEN date_trunc('{period_col}', {time_col}) = ? THEN {metric} ELSE 0 END) AS period_a,
                   sum(CASE WHEN date_trunc('{period_col}', {time_col}) = ? THEN {metric} ELSE 0 END) AS period_b
            FROM data
            GROUP BY {dimension}
        """
        print(f"[AI-DEBUG] variance: metric={metric} dim={dimension} a={period_a} b={period_b}")
        df = self.con.execute(sql, [period_a, period_b]).fetchdf()
        df['change_pct'] = ((df['period_b'] - df['period_a']) / df['period_a'].replace(0, pd.NA) * 100).round(2)
        return df
