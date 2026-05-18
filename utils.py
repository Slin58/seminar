import pandas as pd
import numpy as np

def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the raw HF dataset into a clean analysis panel."""
    df = df.copy()

    # Parse date
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)

    # Create series_id (unique store x product combination)
    series_keys = df[["store_id", "product_id"]].drop_duplicates().reset_index(drop=True)
    series_keys["series_id"] = range(1, len(series_keys) + 1)
    df = df.merge(series_keys, on=["store_id", "product_id"], how="left")

    # Create day index (days since start)
    min_date = df["dt"].min()
    df["day_idx"] = (df["dt"] - min_date).dt.days + 1

    n_series = df["series_id"].nunique()
    n_days = df["day_idx"].nunique()
    print(f"Prepared {len(df):,} rows \u2014 {n_series:,} series x {n_days} days")
    print(f"Date range: {df['dt'].min().date()} to {df['dt'].max().date()}")
    return df


def flag_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """Add censoring flags based on stockout hours."""
    df = df.copy()
    df["is_censored"] = (df["stock_hour6_22_cnt"] > 0).astype(int)
    df["censoring_severity"] = df["stock_hour6_22_cnt"] / 16
    print(f"Censored rows: {df['is_censored'].sum():,} / {len(df):,} ({df['is_censored'].mean():.1%})")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features for EDA and forecasting."""
    df = df.sort_values(["series_id", "day_idx"]).copy()
    grp = df.groupby("series_id")["sale_amount"]
    df["sales_lag1"] = grp.shift(1)
    df["sales_lag7"] = grp.shift(7)
    df["sales_roll7"] = grp.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["sales_roll28"] = grp.transform(lambda x: x.rolling(28, min_periods=1).mean())
    df["psd"] = grp.transform("mean")  # per-series daily mean
    return df


def time_split(df: pd.DataFrame, horizon: int = 7) -> tuple:
    """Split into train and validation by time. Validation = last `horizon` days."""
    min_day = df["day_idx"].min()
    max_day = df["day_idx"].max()
    val_start = max_day - horizon + 1
    train = df[df["day_idx"] < val_start].copy()
    val = df[df["day_idx"] >= val_start].copy()
    print(f"Train: day {min_day}..{val_start - 1} ({len(train):,} rows), Val: day {val_start}..{max_day} ({len(val):,} rows)")
    return train, val


# wape evaluation function
def compute_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / denom


def evaluate_forecast(val_df: pd.DataFrame, pred_col: str = "prediction") -> dict:
    """Compute WAPE overall, low-sale, high-sale, and harmonic mean.
    Only evaluates rows where stock_hour6_22_cnt == 0 (uncensored in validation)."""
    scored = val_df[val_df["stock_hour6_22_cnt"] == 0].copy()
    if len(scored) == 0:
        return {"wape_overall": np.nan}

    y = scored["sale_amount"].values
    yhat = scored[pred_col].values

    wape_all = compute_wape(y, yhat)

    low = scored[scored["psd"] < 1]
    high = scored[scored["psd"] >= 1]

    wape_low = compute_wape(low["sale_amount"].values, low[pred_col].values) if len(low) > 0 else np.nan
    wape_high = compute_wape(high["sale_amount"].values, high[pred_col].values) if len(high) > 0 else np.nan

    if np.isnan(wape_low) or np.isnan(wape_high) or wape_all == 0 or wape_low == 0 or wape_high == 0:
        hm = np.nan
    else:
        hm = 3 / (1/wape_all + 1/wape_low + 1/wape_high)

    return {
        "wape_overall": round(wape_all, 4) if not np.isnan(wape_all) else np.nan,
        "wape_low_sale": round(wape_low, 4) if not np.isnan(wape_low) else np.nan,
        "wape_high_sale": round(wape_high, 4) if not np.isnan(wape_high) else np.nan,
        "harmonic_mean": round(hm, 4) if not np.isnan(hm) else np.nan,
        "scored_rows": len(scored),
    }