def add_seasonal_naive_forecast(df, data_col, pred_col=None, lag=7, fallback="series_mean"):

    df = df.copy()

    if pred_col is None:
        pred_col = f"pred_{data_col}_naive"

    df = df.sort_values(["series_id", "day_idx"])

    df[pred_col] = (
        df.groupby("series_id")[data_col]
        .shift(lag)
    )

    if fallback == "series_mean":
        fallback_values = (
            df.groupby("series_id")[data_col]
            .transform("mean")
        )
        df[pred_col] = df[pred_col].fillna(fallback_values)

    elif fallback == "global_mean":

        global_mean = df[data_col].mean()
        df[pred_col] = df[pred_col].fillna(global_mean)

    return df