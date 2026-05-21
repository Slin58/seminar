def global_mean(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der jeweiligen series_id im Training.
    """
    val_pred = val_df.copy()

    series_mean = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["series_id"].map(series_mean)

    fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def seasonal_naive(train_df, val_df, target_col):
    """
    Forecast = Wert von vor 7 Tagen.
    """
    val_pred = val_df.copy()

    val_start = val_pred["day_idx"].min()

    last_week = train_df[
        train_df["day_idx"].between(val_start - 7, val_start - 1)
    ][["series_id", "day_idx", target_col]].copy()

    last_week["forecast_day"] = last_week["day_idx"] + 7

    last_week = last_week.rename(columns={target_col: "prediction"})

    val_pred = val_pred.merge(
        last_week[["series_id", "forecast_day", "prediction"]],
        left_on=["series_id", "day_idx"],
        right_on=["series_id", "forecast_day"],
        how="left"
    )

    val_pred = val_pred.drop(columns=["forecast_day"], errors="ignore")

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def rolling_28d(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der letzten 28 Trainingstage pro series_id.
    """
    val_pred = val_df.copy()

    roll28 = train_df.groupby("series_id")[target_col].apply(
        lambda x: x.tail(28).mean()
    )

    val_pred["prediction"] = val_pred["series_id"].map(roll28)

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred
