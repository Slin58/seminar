
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed

# TODO: Forecast Methoden:

# - random_forest
# - xgboost
# - lightgbm
# - CNN
# - RNN, LSTM oder Transformer
# - Gaussian Process Regression

# Nils:
# - exponential smoothing (Holt-Winters) pro series_id, mit Fallbacks -> dauert wahrscheinlich zu lange
# - simple tripple exponential smoothing

# Laura: 
# - arima (lädt über eine Stunde nur für raw_sales)
# - sarima

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


#TODO single and double exponential smoothing (alpha, beta, gamma einstellen) und holt-winters nochmal optimieren bzw. Chati fragen warum er solange laden

def single_exponential_smoothing(train_df, val_df, target_col):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    val_pred = val_df.copy()

    fallback = train_df[target_col].mean()

    predictions = {}

    for series_id, train_series in train_df.groupby("series_id"):

        # Convert to numpy to avoid index warnings
        y = train_series[target_col].to_numpy()

        if len(y) == 0:
            predictions[series_id] = fallback

        elif len(y) == 1:
            predictions[series_id] = y[0]

        else:
            try:
                fit = SimpleExpSmoothing(y).fit(optimized=True)

                predictions[series_id] = float(fit.forecast(1)[0])

            except Exception:
                predictions[series_id] = y.mean()

    val_pred["prediction"] = (
        val_pred["series_id"]
        .map(predictions)
        .fillna(fallback)
        .clip(lower=0)
    )
    print(f"smoothing_level: {fit.params['smoothing_level']}")
    return val_pred

def double_exponential_smoothing(train_df, val_df, target_col):
    from statsmodels.tsa.holtwinters import Holt

    val_pred = val_df.copy()

    fallback = train_df[target_col].mean()
    predictions = {}

    for series_id, train_series in train_df.groupby("series_id"):

        y = train_series[target_col].dropna().to_numpy()

        if len(y) < 2:
            predictions[series_id] = y.mean() if len(y) else fallback
            continue

        try:
            fit = Holt(y).fit(optimized=True)

            alpha = fit.params["smoothing_level"]
            beta = fit.params["smoothing_trend"]

            # print(f"series_id={series_id}, alpha={alpha:.4f}, beta={beta:.4f}"

            predictions[series_id] = float(fit.forecast(1)[0])

        except Exception as e:
            print(f"series_id={series_id}: {e}")
            predictions[series_id] = y.mean()

    val_pred["prediction"] = (
        val_pred["series_id"]
        .map(predictions)
        .fillna(fallback)
        .clip(lower=0)
    )

    return val_pred

def triple_exponential_smoothing(
    train_df,
    val_df,
    target_col,
    seasonal_periods=7
):
    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    val_pred = val_df.copy()

    fallback = train_df[target_col].mean()
    predictions = {}

    for series_id, train_series in train_df.groupby("series_id"):

        y = train_series[target_col].dropna().to_numpy()

        # Need enough data for seasonality
        if len(y) < 2 * seasonal_periods:
            predictions[series_id] = y.mean() if len(y) else fallback
            continue

        try:
            fit = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods
            ).fit(optimized=True)

            alpha = fit.params["smoothing_level"]
            beta = fit.params["smoothing_trend"]
            gamma = fit.params["smoothing_seasonal"]

            print(
                f"series_id={series_id}, "
                f"alpha={alpha:.4f}, "
                f"beta={beta:.4f}, "
                f"gamma={gamma:.4f}"
            )

            predictions[series_id] = float(fit.forecast(1)[0])

        except Exception as e:
            print(f"series_id={series_id}: {e}")
            predictions[series_id] = y.mean()

    val_pred["prediction"] = (
        val_pred["series_id"]
        .map(predictions)
        .fillna(fallback)
        .clip(lower=0)
    )

    return val_pred


def simple_exponential_smoothing(train_df, val_df, target_col, alpha=0.3): # single exp with series_id
    """
    Simple Exponential Smoothing (SES).

    Pro series_id wird der exponentiell geglättete Level am Ende der
    Trainingsdaten berechnet:

        S_1 = y_1
        S_t = alpha * y_t + (1 - alpha) * S_{t-1}

    Der finale Level S_T wird als konstanter Forecast für alle
    Validierungstage der jeweiligen series_id verwendet.

    alpha: Glättungsfaktor zwischen 0 und 1.
           - alpha nah an 1 -> reagiert stark auf die letzten Werte
           - alpha nah an 0 -> glättet stark, reagiert langsam
    """
    val_pred = val_df.copy()

    train_sorted = train_df.sort_values(["series_id", "day_idx"])

    levels = (
        train_sorted.groupby("series_id")[target_col]
        .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
    )
    levels.name = "prediction"

    val_pred = val_pred.merge(
        levels.reset_index(), on="series_id", how="left"
    )

    # Fallback 1: series_id Durchschnitt (falls series_id nicht in train)
    fallback_series = train_df.groupby("series_id")[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback_series)
    )

    # Fallback 2: globaler Durchschnitt
    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def holt_winters_exp_forecast(train_df, val_df, target_col, seasonal_periods=7):
    """
    Optimized Holt-Winters forecasting per series_id.

    Optimisations over original:
    - Parallel execution via joblib (biggest win for many series)
    - Early complexity detection: skip HW for near-constant series
    - Cached numpy arrays to avoid repeated conversions
    - Tighter fit options: fewer optimizer iterations, no covariance matrix
    - Single-pass prediction dict build
    """

    global_fallback = float(train_df[target_col].mean())
    series_fallback = train_df.groupby("series_id")[target_col].mean().to_dict()

    train_groups = {
        sid: grp.sort_values("day_idx")[target_col].to_numpy(dtype=np.float64)
        for sid, grp in train_df.groupby("series_id")
    }
    train_max_day = {
        sid: grp["day_idx"].iloc[-1]
        for sid, grp in train_df.sort_values("day_idx").groupby("series_id")
    }

    val_groups = {
        sid: grp.sort_values("day_idx")["day_idx"].to_numpy()
        for sid, grp in val_df.groupby("series_id")
    }

    def _fit_one(series_id):
        val_days = val_groups.get(series_id)
        if val_days is None:
            return []

        y = train_groups.get(series_id)
        if y is None or len(y) == 0:
            return [(series_id, d, global_fallback) for d in val_days]

        n = len(y)
        fb = series_fallback.get(series_id, global_fallback)
        last_train_day = train_max_day[series_id]
        max_horizon = max(int(val_days.max() - last_train_day), 1)

        # Skip expensive fitting for near-constant series
        if np.std(y) < 1e-8:
            forecast = np.full(max_horizon, y[-1], dtype=np.float64)
        else:
            forecast = _try_fit(y, n, max_horizon, seasonal_periods)

        if forecast is None:
            forecast = np.full(max_horizon, fb, dtype=np.float64)

        results = []
        for day_idx in val_days:
            step = int(day_idx - last_train_day) - 1
            pred = forecast[step] if 0 <= step < len(forecast) else fb
            results.append((series_id, day_idx, pred))
        return results

    def _try_fit(y, n, max_horizon, seasonal_periods):
        """Attempt HW -> Holt -> SES with tight optimizer settings."""
        fit_kw = dict(
            optimized=True,
            use_brute=False,       # skip brute-force grid search
            remove_bias=False,
        )

        # Holt-Winters
        if n >= 2 * seasonal_periods:
            fc = _fit_model(
                y,
                dict(trend="add", damped_trend=True, seasonal="add",
                     seasonal_periods=seasonal_periods,
                     initialization_method="estimated"),
                max_horizon,
                fit_kw,
            )
            if fc is not None:
                return fc

        # Holt
        if n >= 2:
            fc = _fit_model(
                y,
                dict(trend="add", damped_trend=True, seasonal=None,
                     initialization_method="estimated"),
                max_horizon,
                fit_kw,
            )
            if fc is not None:
                return fc

        # SES
        if n >= 1:
            fc = _fit_model(
                y,
                dict(trend=None, seasonal=None,
                     initialization_method="estimated"),
                max_horizon,
                fit_kw,
            )
            if fc is not None:
                return fc

        return None

    def _fit_model(y, model_kwargs, horizon, fit_kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(y, **model_kwargs).fit(**fit_kwargs)
            return np.asarray(model.forecast(horizon))
        except Exception:
            return None

    # --- Parallel execution over series ---
    # n_jobs=-1 uses all CPU cores; tune if you want to leave headroom
    all_results = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
        delayed(_fit_one)(sid) for sid in val_groups
    )

    rows = [row for series_rows in all_results for row in series_rows]
    pred_df = pd.DataFrame(rows, columns=["series_id", "day_idx", "prediction"])

    val_pred = val_df.copy().merge(pred_df, on=["series_id", "day_idx"], how="left")
    val_pred["prediction"] = (
        val_pred["prediction"].fillna(global_fallback).clip(lower=0)
    )
    return val_pred

def arima(train_df, val_df, target_col, order=(1, 1, 1)):
    """
    ARIMA Forecast pro series_id.

    Für jede series_id wird ein ARIMA-Modell auf den Trainingsdaten
    gefittet und für die Validierungstage vorhergesagt.

    ARIMA(p,d,q):
    - p: Anzahl vergangener Werte
    - d: Anzahl Differenzierungen gegen Trend
    - q: Anzahl vergangener Fehlerterme

    Fallback-Kette:
    1. ARIMA(order)
    2. Seasonal Naive / letzter Wert der Vorwoche, falls möglich
    3. series_id Durchschnitt
    4. globaler Durchschnitt
    """

    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_fallback = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):

        train_group = train_df[
            train_df["series_id"] == series_id
        ].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            fb = global_fallback
            for day_idx in val_days:
                predictions[(series_id, day_idx)] = fb
            continue

        y = train_group[target_col].astype(float).values
        n = len(y)

        train_max_day = train_group["day_idx"].max()

        max_horizon = int(val_days.max() - train_max_day)
        max_horizon = max(max_horizon, 1)

        forecast = None

        # ARIMA braucht genug Werte
        if n >= 10:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = ARIMA(
                        y,
                        order=order
                    )

                    fitted = model.fit()

                    forecast = fitted.forecast(steps=max_horizon)

            except Exception:
                forecast = None

        # Fallback 1: letzter verfügbarer Wert
        if forecast is None:
            if n > 0:
                last_value = y[-1]
                forecast = np.full(max_horizon, last_value)
            else:
                fb = series_fallback.get(series_id, global_fallback)
                forecast = np.full(max_horizon, fb)

        for day_idx in val_days:
            step = int(day_idx - train_max_day) - 1

            if 0 <= step < len(forecast):
                pred_value = forecast[step]
            else:
                pred_value = series_fallback.get(series_id, global_fallback)

            predictions[(series_id, day_idx)] = pred_value

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get(
            (row["series_id"], row["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def arima_like_fast(train_df, val_df, target_col, order=(1, 1, 0)):
    """
    Fast ARIMA-like Forecast pro series_id.

    Idee:
    - Kein echtes ARIMA-Fitting für jede series_id.
    - Nutzt ARIMA(1,1,0)-ähnliche Logik:
        Prognose = letzter Wert + geglätteter letzter Trend
    - Sehr viel schneller als statsmodels ARIMA.
    """

    import numpy as np
    import pandas as pd

    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_mean = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):

        train_group = train_df[
            train_df["series_id"] == series_id
        ].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            forecast_values = np.full(len(val_days), global_fallback)
        else:
            y = train_group[target_col].astype(float).values
            train_max_day = train_group["day_idx"].max()

            if len(y) >= 2:
                last_value = y[-1]

                diffs = np.diff(y)

                # robuster Trend aus letzten Tagen
                recent_trend = np.nanmean(diffs[-7:]) if len(diffs) >= 7 else np.nanmean(diffs)

                if np.isnan(recent_trend):
                    recent_trend = 0.0

                forecast_values = []

                for day_idx in val_days:
                    step = int(day_idx - train_max_day)

                    pred = last_value + step * recent_trend

                    forecast_values.append(pred)

                forecast_values = np.array(forecast_values)

            elif len(y) == 1:
                forecast_values = np.full(len(val_days), y[-1])
            else:
                forecast_values = np.full(
                    len(val_days),
                    series_mean.get(series_id, global_fallback)
                )

        for day_idx, pred in zip(val_days, forecast_values):
            predictions[(series_id, day_idx)] = pred

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get(
            (row["series_id"], row["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def arima_like_fast_vectorized(train_df, val_df, target_col):
    import numpy as np
    import pandas as pd

    train = train_df.sort_values(["series_id", "day_idx"]).copy()
    val_pred = val_df.copy()

    global_fallback = train[target_col].mean()

    last = train.groupby("series_id").tail(1)[
        ["series_id", "day_idx", target_col]
    ].rename(columns={
        "day_idx": "train_max_day",
        target_col: "last_value"
    })

    train["diff"] = train.groupby("series_id")[target_col].diff()

    trend = (
        train.groupby("series_id")["diff"]
        .tail(7)
        .groupby(train["series_id"])
        .mean()
        .reset_index()
        .rename(columns={"diff": "recent_trend"})
    )

    val_pred = val_pred.merge(last, on="series_id", how="left")
    val_pred = val_pred.merge(trend, on="series_id", how="left")

    val_pred["last_value"] = val_pred["last_value"].fillna(global_fallback)
    val_pred["recent_trend"] = val_pred["recent_trend"].fillna(0)

    val_pred["step"] = val_pred["day_idx"] - val_pred["train_max_day"]

    val_pred["prediction"] = (
        val_pred["last_value"] +
        val_pred["step"] * val_pred["recent_trend"]
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def lightgbm_forecast(train_df, val_df, target_col, random_state=42):
    """
    LightGBM Forecast.

    Trainiert ein globales Modell über alle series_id hinweg.
    Zielvariable ist target_col, z. B.:
    - sale_amount
    - recovered_daily_sales_xgboost
    - recovered_daily_sales_hourly_mean

    Das Modell nutzt Zeit-, Serien- und Kontextfeatures.
    """

    import lightgbm as lgb

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # 1. Features erstellen
    # ------------------------------------------------------------

    def add_lgbm_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        # Lag-/Rolling-Features aus sale_amount bzw. vorhandenen Features
        # Falls target_col schon eigene Recovery-Spalte ist, werden Features
        # trotzdem aus dieser Zielspalte gebaut.
        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)
        df["rolling7"] = grp.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df["rolling28"] = grp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)

        return df

    combined = pd.concat([train, val_pred], axis=0).sort_values(["series_id", "day_idx"])

    combined = add_lgbm_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    # ------------------------------------------------------------
    # 2. Feature-Liste
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    # ------------------------------------------------------------
    # 3. Missing Values behandeln
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # 4. Modell trainieren
    # ------------------------------------------------------------

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 5. Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_optimized(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])
        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag2"] = grp.shift(2)
        df["lag3"] = grp.shift(3)
        df["lag7"] = grp.shift(7)
        df["lag14"] = grp.shift(14)
        df["lag28"] = grp.shift(28)

        df["rolling7"] = grp.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df["rolling14"] = grp.shift(1).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
        df["rolling28"] = grp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)

        df["rolling_std7"] = grp.shift(1).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
        df["rolling_std28"] = grp.shift(1).rolling(28, min_periods=2).std().reset_index(level=0, drop=True)

        df["diff1"] = df["lag1"] - df["lag2"]
        df["diff7"] = df["lag7"] - df["lag14"]

        df["ratio_lag1_rolling7"] = df["lag1"] / (df["rolling7"] + 1e-6)
        df["ratio_lag7_rolling28"] = df["lag7"] / (df["rolling28"] + 1e-6)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id", "management_group_id",
        "weekday", "month", "week_of_year", "is_weekend", "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature", "avg_humidity", "avg_wind_level", "precpt",
        "lag1", "lag2", "lag3", "lag7", "lag14", "lag28",
        "rolling7", "rolling14", "rolling28",
        "rolling_std7", "rolling_std28",
        "diff1", "diff7",
        "ratio_lag1_rolling7", "ratio_lag7_rolling28",
        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        # ------------------------------------------------------------
        # Lags
        # ------------------------------------------------------------
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        # ------------------------------------------------------------
        # Rolling Features
        # ------------------------------------------------------------
        shifted = grp.shift(1)

        windows = [3, 7, 14, 21, 28]

        for w in windows:
            df[f"roll_mean_{w}"] = (
                shifted
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_std_{w}"] = (
                shifted
                .rolling(w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_min_{w}"] = (
                shifted
                .rolling(w, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_max_{w}"] = (
                shifted
                .rolling(w, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_median_{w}"] = (
                shifted
                .rolling(w, min_periods=1)
                .median()
                .reset_index(level=0, drop=True)
            )

        # ------------------------------------------------------------
        # Trend Features
        # ------------------------------------------------------------
        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        # ------------------------------------------------------------
        # Interaction Features
        # ------------------------------------------------------------
        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    # ------------------------------------------------------------
    # Combine Train + Validation for lag features
    # ------------------------------------------------------------
    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    # ------------------------------------------------------------
    # Feature Columns
    # ------------------------------------------------------------
    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [f"lag_{lag}" for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    # ------------------------------------------------------------
    # Prepare Data
    # ------------------------------------------------------------
    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=700,
        learning_rate=0.03,

        num_leaves=63,
        max_depth=10,
        min_child_samples=30,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.5,
        reg_lambda=1.0,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized_v2(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        windows = [3, 7, 14, 21, 28]

        for w in windows:
            df[f"roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_std_{w}"] = (
                shifted.rolling(w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_min_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_max_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_median_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .median()
                .reset_index(level=0, drop=True)
            )

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=900,
        learning_rate=0.025,

        num_leaves=63,
        max_depth=10,
        min_child_samples=30,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.5,
        reg_lambda=1.0,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized_v3(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)
        windows = [3, 7, 14, 21, 28]

        for w in windows:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id", "management_group_id",
        "weekday", "month", "week_of_year", "is_weekend",
        "weekday_sin", "weekday_cos", "month_sin", "month_cos", "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature", "avg_humidity", "avg_wind_level", "precpt",
        "psd",
    ]

    feature_cols += [f"lag_{lag}" for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}",
            f"roll_max_{w}", f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7", "trend_14", "trend_28",
        "trend_ratio_7", "trend_ratio_14", "trend_ratio_28",
        "lag1_roll7", "lag1_roll28", "lag7_roll7",
        "ratio_roll7", "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=700,
        learning_rate=0.03,

        num_leaves=127,
        max_depth=12,
        min_child_samples=40,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.7,
        reg_lambda=1.5,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized_v4(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        windows = [3, 7, 14, 21, 28]

        for w in windows:
            df[f"roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_std_{w}"] = (
                shifted.rolling(w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_min_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .min()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_max_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_median_{w}"] = (
                shifted.rolling(w, min_periods=1)
                .median()
                .reset_index(level=0, drop=True)
            )

        # ------------------------------------------------------------
        # EWMA Features
        # ------------------------------------------------------------
        for span in [3, 7, 14, 21, 28]:
            df[f"ewm_mean_{span}"] = (
                shifted.ewm(span=span, adjust=False)
                .mean()
                .reset_index(level=0, drop=True)
            )

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        f"ewm_mean_{w}"
        for w in [3, 7, 14, 21, 28]
    ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=700,
        learning_rate=0.03,

        num_leaves=127,
        max_depth=12,
        min_child_samples=40,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.7,
        reg_lambda=1.5,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized_v5(train_df, val_df, target_col, random_state=42):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        # Trend Features
        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        # Interaction Features
        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        # ------------------------------------------------------------
        # Neue Differenz-Features
        # ------------------------------------------------------------
        df["diff_1"] = df["lag_1"] - df["lag_2"]
        df["diff_7"] = df["lag_1"] - df["lag_7"]
        df["diff_14"] = df["lag_7"] - df["lag_14"]
        df["diff_28"] = df["lag_28"] - df["lag_56"]

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",

        "diff_1",
        "diff_7",
        "diff_14",
        "diff_28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=700,
        learning_rate=0.03,

        num_leaves=127,
        max_depth=12,
        min_child_samples=40,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.7,
        reg_lambda=1.5,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def lightgbm_forecast_feature_optimized_v6_feature_selection(
    train_df,
    val_df,
    target_col,
    random_state=42,
    top_n_features=45
):
    import lightgbm as lgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id", "management_group_id",
        "weekday", "month", "week_of_year", "is_weekend",
        "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature", "avg_humidity", "avg_wind_level", "precpt",
        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7", "trend_14", "trend_28",
        "trend_ratio_7", "trend_ratio_14", "trend_ratio_28",
        "lag1_roll7", "lag1_roll28", "lag7_roll7",
        "ratio_roll7", "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # 1. Modell mit allen Features trainieren
    # ------------------------------------------------------------
    base_model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=12,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.7,
        reg_lambda=1.5,
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    base_model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": base_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("=== Feature Importance ===")
    print(importance_df.to_string(index=False))

    selected_features = importance_df.head(top_n_features)["feature"].tolist()

    print(f"\nSelected top {top_n_features} features:")
    print(selected_features)

    # ------------------------------------------------------------
    # 2. Modell nochmal nur mit Top Features trainieren
    # ------------------------------------------------------------
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    final_model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=12,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.7,
        reg_lambda=1.5,
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    final_model.fit(X_train_selected, y_train)

    val_feat["prediction"] = final_model.predict(X_val_selected)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def xgboost_forecast(train_df, val_df, target_col, random_state=42):
    """
    XGBoost Forecast.

    Globales Modell über alle series_id.
    Forecast der nächsten Tage auf Basis von
    Zeit-, Produkt-, Store- und Wetterfeatures.
    """

    import xgboost as xgb
    import numpy as np
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # 1. Features erstellen
    # ------------------------------------------------------------

    def add_xgb_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])

    combined = add_xgb_features(combined)

    train_feat = combined[
        combined["day_idx"].isin(train["day_idx"])
    ].copy()

    val_feat = combined[
        combined["day_idx"].isin(val_pred["day_idx"])
    ].copy()

    # ------------------------------------------------------------
    # 2. Features auswählen
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [
        c for c in feature_cols
        if c in train_feat.columns
    ]

    # ------------------------------------------------------------
    # 3. Daten vorbereiten
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # 4. Modell trainieren
    # ------------------------------------------------------------

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 5. Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def xgboost_forecast_feature_optimized(train_df, val_df, target_col, random_state=42):
    import xgboost as xgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])
        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
            df[f"roll_std_{w}"] = (
                shifted.rolling(w, min_periods=2).std()
                .reset_index(level=0, drop=True)
            )
            df[f"roll_min_{w}"] = (
                shifted.rolling(w, min_periods=1).min()
                .reset_index(level=0, drop=True)
            )
            df[f"roll_max_{w}"] = (
                shifted.rolling(w, min_periods=1).max()
                .reset_index(level=0, drop=True)
            )
            df[f"roll_median_{w}"] = (
                shifted.rolling(w, min_periods=1).median()
                .reset_index(level=0, drop=True)
            )

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id", "management_group_id",
        "weekday", "month", "week_of_year", "is_weekend",
        "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature", "avg_humidity", "avg_wind_level", "precpt",
        "psd",
    ]

    feature_cols += [f"lag_{lag}" for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7", "trend_14", "trend_28",
        "trend_ratio_7", "trend_ratio_14", "trend_ratio_28",
        "lag1_roll7", "lag1_roll28", "lag7_roll7",
        "ratio_roll7", "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def xgboost_forecast_feature_fast(train_df, val_df, target_col, random_state=42):
    import xgboost as xgb
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        for lag in [1, 2, 3, 7, 14, 28]:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [7, 14, 28]:
            df[f"roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )

            df[f"roll_std_{w}"] = (
                shifted.rolling(w, min_periods=2).std()
                .reset_index(level=0, drop=True)
            )

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id", "management_group_id",
        "weekday", "month", "week_of_year", "is_weekend", "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "psd",
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_std_7",
        "roll_mean_14", "roll_std_14",
        "roll_mean_28", "roll_std_28",
        "trend_7", "trend_14", "trend_28",
        "ratio_roll7", "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def random_forest_forecast(train_df, val_df, target_col, random_state=42):
    """
    Random Forest Forecast.

    Globales Modell über alle series_id.
    Nutzt Zeit-, Produkt-, Store- und Wetterfeatures.
    """

    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # Features erstellen
    # ------------------------------------------------------------

    def add_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])

    combined = add_features(combined)

    train_feat = combined[
        combined["day_idx"].isin(train["day_idx"])
    ].copy()

    val_feat = combined[
        combined["day_idx"].isin(val_pred["day_idx"])
    ].copy()

    # ------------------------------------------------------------
    # Features auswählen
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [
        c for c in feature_cols
        if c in train_feat.columns
    ]

    # ------------------------------------------------------------
    # Daten vorbereiten
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # Modell trainieren
    # ------------------------------------------------------------

    # model = RandomForestRegressor (eine runde ca 58 min)
    #     n_estimators=200,
    #     max_depth=15,
    #     min_samples_leaf=20,
    #     n_jobs=-1,
    #     random_state=random_state,
    #     verbose=1
    # )
    model = RandomForestRegressor(
        n_estimators=50,          # statt 200
        max_depth=12,             # statt 15
        min_samples_leaf=50,      # statt 20
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def random_forest_forecast_optimized(train_df, val_df, target_col, random_state=42):
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["week_of_year"] = pd.to_datetime(df["dt"]).dt.isocalendar().week.astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)
        df["lag14"] = grp.shift(14)
        df["lag28"] = grp.shift(28)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std_7"] = (
            grp.shift(1)
            .rolling(7, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std_28"] = (
            grp.shift(1)
            .rolling(28, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "week_of_year",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "lag14",
        "lag28",
        "rolling7",
        "rolling28",
        "rolling_std_7",
        "rolling_std_28",
        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = RandomForestRegressor(
        n_estimators=75,
        max_depth=15,
        min_samples_leaf=30,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def random_forest_forecast_feature_optimized(train_df, val_df, target_col, random_state=42):
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])
        df["weekday"] = df["dt"].dt.weekday
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        # Lags
        df["lag1"] = grp.shift(1)
        df["lag2"] = grp.shift(2)
        df["lag3"] = grp.shift(3)
        df["lag6"] = grp.shift(6)
        df["lag7"] = grp.shift(7)
        df["lag8"] = grp.shift(8)
        df["lag14"] = grp.shift(14)
        df["lag28"] = grp.shift(28)

        # Rolling Means
        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling14"] = (
            grp.shift(1)
            .rolling(14, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling Std
        df["rolling_std_7"] = (
            grp.shift(1)
            .rolling(7, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std_14"] = (
            grp.shift(1)
            .rolling(14, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std_28"] = (
            grp.shift(1)
            .rolling(28, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        # Change / Trend Features
        df["diff1"] = df["lag1"] - df["lag2"]
        df["diff7"] = df["lag7"] - df["lag14"]

        df["ratio_lag1_rolling7"] = df["lag1"] / (df["rolling7"] + 1e-6)
        df["ratio_lag7_rolling28"] = df["lag7"] / (df["rolling28"] + 1e-6)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "is_weekend",
        "week_of_year",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "lag1",
        "lag2",
        "lag3",
        "lag6",
        "lag7",
        "lag8",
        "lag14",
        "lag28",

        "rolling7",
        "rolling14",
        "rolling28",

        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",

        "diff1",
        "diff7",
        "ratio_lag1_rolling7",
        "ratio_lag7_rolling28",

        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = RandomForestRegressor(
        n_estimators=75,
        max_depth=15,
        min_samples_leaf=30,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def cnn_forecast(train_df, val_df, target_col, window=28, epochs=5, batch_size=4096, random_state=42):
    """
    CNN Forecast.

    Nutzt pro series_id die letzten `window` Tage als Input
    und sagt die nächsten Validierungstage voraus.
    """

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    # ------------------------------------------------------------
    # 1. Trainingssequenzen bauen
    # ------------------------------------------------------------

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # ------------------------------------------------------------
    # 3. CNN Modell
    # ------------------------------------------------------------

    class CNNForecast(nn.Module):
        def __init__(self, window):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CNNForecast(window)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("\n=== CNN Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(loader):.6f}")

    # ------------------------------------------------------------
    # 5. Forecast für Validation
    # ------------------------------------------------------------

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = train_group[target_col].fillna(global_fallback).values.astype(np.float32).tolist()

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(history_values[-window:], dtype=np.float32).reshape(1, -1)
                x_scaled = x_scaler.transform(x_input)

                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)

                with torch.no_grad():
                    pred_scaled = model(x_tensor).numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            # autoregressiv: Prognose wird für nächsten Tag weiterverwendet
            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get((r["series_id"], r["day_idx"]), global_fallback),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def cnn_forecast_fast(
    train_df,
    val_df,
    target_col,
    window=21,
    epochs=3,
    batch_size=16384,
    random_state=42
):
    """
    Faster CNN Forecast.
    Nutzt weiterhin alle Daten, aber mit kleinerem Modell,
    größerer Batch Size und optional GPU.
    """

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    # ------------------------------------------------------------
    # 1. Trainingssequenzen bauen
    # ------------------------------------------------------------

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ------------------------------------------------------------
    # 3. Kleineres CNN-Modell
    # ------------------------------------------------------------

    class CNNForecastFast(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),

                nn.Linear(32, 16),
                nn.ReLU(),

                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CNNForecastFast().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("\n=== Fast CNN Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / n_batches:.6f}")

    # ------------------------------------------------------------
    # 5. Forecast für Validation
    # ------------------------------------------------------------

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = (
            train_group[target_col]
            .fillna(global_fallback)
            .values
            .astype(np.float32)
            .tolist()
        )

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(
                    history_values[-window:],
                    dtype=np.float32
                ).reshape(1, -1)

                x_scaled = x_scaler.transform(x_input)

                x_tensor = (
                    torch.tensor(x_scaled, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )

                with torch.no_grad():
                    pred_scaled = model(x_tensor).cpu().numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            # autoregressiv weiterverwenden
            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get(
            (r["series_id"], r["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def cnn_forecast_balanced(
    train_df,
    val_df,
    target_col,
    window=28,
    epochs=3,
    batch_size=16384,
    random_state=42
):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    class CNNForecastBalanced(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(1, 24, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.Conv1d(24, 48, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),

                nn.Linear(48, 24),
                nn.ReLU(),

                nn.Linear(24, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CNNForecastBalanced().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("\n=== Balanced CNN Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / n_batches:.6f}")

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = (
            train_group[target_col]
            .fillna(global_fallback)
            .values
            .astype(np.float32)
            .tolist()
        )

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(
                    history_values[-window:],
                    dtype=np.float32
                ).reshape(1, -1)

                x_scaled = x_scaler.transform(x_input)

                x_tensor = (
                    torch.tensor(x_scaled, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )

                with torch.no_grad():
                    pred_scaled = model(x_tensor).cpu().numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get(
            (r["series_id"], r["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def cnn_forecast_3epochs(
    train_df,
    val_df,
    target_col,
    window=28,
    epochs=3,
    batch_size=16384,
    random_state=42
):
    """
    CNN Forecast.
    Originale Architektur, aber nur 3 Epochs und größere Batch Size.
    Ziel: fast gleiche Qualität wie Original-CNN, aber schneller.
    """

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    class CNNForecast(nn.Module):
        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),

                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CNNForecast().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("\n=== CNN Forecast Training: Original Architecture, 3 Epochs ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / n_batches:.6f}")

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = (
            train_group[target_col]
            .fillna(global_fallback)
            .values
            .astype(np.float32)
            .tolist()
        )

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(history_values[-window:], dtype=np.float32).reshape(1, -1)
                x_scaled = x_scaler.transform(x_input)

                x_tensor = (
                    torch.tensor(x_scaled, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )

                with torch.no_grad():
                    pred_scaled = model(x_tensor).cpu().numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]
            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get(
            (r["series_id"], r["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def cnn_forecast_original_3epochs(train_df, val_df, target_col, window=28, epochs=3, batch_size=4096, random_state=42):
    """
    CNN Forecast.

    Nutzt pro series_id die letzten `window` Tage als Input
    und sagt die nächsten Validierungstage voraus.
    """

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    # ------------------------------------------------------------
    # 1. Trainingssequenzen bauen
    # ------------------------------------------------------------

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # ------------------------------------------------------------
    # 3. CNN Modell
    # ------------------------------------------------------------

    class CNNForecast(nn.Module):
        def __init__(self, window):
            super().__init__()

            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = CNNForecast(window)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("\n=== CNN Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(loader):.6f}")

    # ------------------------------------------------------------
    # 5. Forecast für Validation
    # ------------------------------------------------------------

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = train_group[target_col].fillna(global_fallback).values.astype(np.float32).tolist()

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(history_values[-window:], dtype=np.float32).reshape(1, -1)
                x_scaled = x_scaler.transform(x_input)

                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)

                with torch.no_grad():
                    pred_scaled = model(x_tensor).numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            # autoregressiv: Prognose wird für nächsten Tag weiterverwendet
            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get((r["series_id"], r["day_idx"]), global_fallback),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def lstm_forecast(
    train_df,
    val_df,
    target_col,
    window=28,
    epochs=3,
    batch_size=4096,
    random_state=42
):
    """
    LSTM Forecast.

    Nutzt pro series_id die letzten `window` Tage als Sequenz
    und sagt die nächsten Validierungstage autoregressiv voraus.
    """

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    # ------------------------------------------------------------
    # 1. Trainingssequenzen bauen
    # ------------------------------------------------------------

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # LSTM erwartet: (batch, sequence_length, features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ------------------------------------------------------------
    # 3. LSTM Modell
    # ------------------------------------------------------------

    class LSTMForecast(nn.Module):
        def __init__(self, hidden_size=32, num_layers=1):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            # output shape: (batch, seq_len, hidden_size)
            output, _ = self.lstm(x)

            # letzter Zeitschritt
            last_output = output[:, -1, :]

            return self.fc(last_output).squeeze(-1)

    model = LSTMForecast(
        hidden_size=32,
        num_layers=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("\n=== LSTM Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / n_batches:.6f}")

    # ------------------------------------------------------------
    # 5. Forecast für Validation
    # ------------------------------------------------------------

    predictions = {}

    model.eval()

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train[train["series_id"] == series_id].sort_values("day_idx")

        history_values = (
            train_group[target_col]
            .fillna(global_fallback)
            .values
            .astype(np.float32)
            .tolist()
        )

        for _, row in val_group.sort_values("day_idx").iterrows():

            if len(history_values) >= window:
                x_input = np.array(
                    history_values[-window:],
                    dtype=np.float32
                ).reshape(1, -1)

                x_scaled = x_scaler.transform(x_input)

                x_tensor = (
                    torch.tensor(x_scaled, dtype=torch.float32)
                    .unsqueeze(-1)
                    .to(device)
                )

                with torch.no_grad():
                    pred_scaled = model(x_tensor).cpu().numpy().reshape(-1, 1)

                pred = y_scaler.inverse_transform(pred_scaled)[0, 0]

            else:
                pred = global_fallback

            pred = max(pred, 0)

            predictions[(series_id, row["day_idx"])] = pred

            # autoregressiv weiterverwenden
            history_values.append(pred)

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get(
            (r["series_id"], r["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def lstm_forecast_fast(
    train_df,
    val_df,
    target_col,
    window=21,
    epochs=3,
    batch_size=8192,
    random_state=42
):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train = train_df.copy().sort_values(["series_id", "day_idx"])
    val_pred = val_df.copy().sort_values(["series_id", "day_idx"])

    global_fallback = train[target_col].mean()

    # ------------------------------------------------------------
    # 1. Trainingssequenzen bauen
    # ------------------------------------------------------------

    X_list = []
    y_list = []

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32)

        if len(values) <= window:
            continue

        for i in range(window, len(values)):
            X_list.append(values[i-window:i])
            y_list.append(values[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    if len(X) == 0:
        val_pred["prediction"] = global_fallback
        return val_pred

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ------------------------------------------------------------
    # 3. Kleineres LSTM
    # ------------------------------------------------------------

    class LSTMForecastFast(nn.Module):
        def __init__(self, hidden_size=16):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            last_output = output[:, -1, :]
            return self.fc(last_output).squeeze(-1)

    model = LSTMForecastFast(hidden_size=16).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("\n=== Fast LSTM Forecast Training ===")
    print(f"Training samples: {len(X):,}")
    print(f"Window: {window}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / n_batches:.6f}")

    # ------------------------------------------------------------
    # 5. Schneller Forecast batchweise
    # ------------------------------------------------------------

    print("Predicting validation...")

    model.eval()

    predictions = {}

    # pro series_id letzte window Werte holen
    last_values = {}

    for series_id, g in train.groupby("series_id"):
        values = g[target_col].fillna(global_fallback).values.astype(np.float32).tolist()

        if len(values) < window:
            values = [global_fallback] * (window - len(values)) + values

        last_values[series_id] = values[-window:]

    val_days = sorted(val_pred["day_idx"].unique())

    for day_idx in val_days:
        series_ids = val_pred[val_pred["day_idx"] == day_idx]["series_id"].values

        X_pred = np.array(
            [last_values.get(sid, [global_fallback] * window) for sid in series_ids],
            dtype=np.float32
        )

        X_pred_scaled = x_scaler.transform(X_pred)

        X_pred_tensor = (
            torch.tensor(X_pred_scaled, dtype=torch.float32)
            .unsqueeze(-1)
            .to(device)
        )

        preds_scaled_all = []

        with torch.no_grad():
            for start in range(0, len(X_pred_tensor), batch_size):
                end = min(start + batch_size, len(X_pred_tensor))

                pred_scaled = model(X_pred_tensor[start:end])
                preds_scaled_all.append(pred_scaled.cpu().numpy())

        preds_scaled_all = np.concatenate(preds_scaled_all).reshape(-1, 1)
        preds = y_scaler.inverse_transform(preds_scaled_all).ravel()
        preds = np.maximum(preds, 0)

        for sid, pred in zip(series_ids, preds):
            predictions[(sid, day_idx)] = pred

            hist = last_values.get(sid, [global_fallback] * window)
            hist = hist[1:] + [float(pred)]
            last_values[sid] = hist

    val_pred["prediction"] = val_pred.apply(
        lambda r: predictions.get(
            (r["series_id"], r["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def catboost_forecast(train_df, val_df, target_col, random_state=42):
    """
    CatBoost Forecast.

    Globales Modell über alle series_id.
    """

    from catboost import CatBoostRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------

    def add_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])

    combined = add_features(combined)

    train_feat = combined[
        combined["day_idx"].isin(train["day_idx"])
    ].copy()

    val_feat = combined[
        combined["day_idx"].isin(val_pred["day_idx"])
    ].copy()

    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    categorical_features = [
        c for c in [
            "series_id",
            "product_id",
            "store_id",
            "city_id",
            "management_group_id",
            "weekday",
            "month",
        ]
        if c in feature_cols
    ]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # Modell
    # ------------------------------------------------------------

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features
    )

    # ------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def catboost_forecast_fast(train_df, val_df, target_col, random_state=42):
    from catboost import CatBoostRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1).rolling(7, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1).rolling(28, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0).sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id", "product_id", "store_id", "city_id",
        "management_group_id", "weekday", "month", "day_idx",
        "discount", "holiday_flag", "activity_flag",
        "avg_temperature", "avg_humidity", "avg_wind_level",
        "precpt", "lag1", "lag7", "rolling7", "rolling28", "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    cat_features = [
        c for c in [
            "series_id", "product_id", "store_id",
            "city_id", "management_group_id",
            "weekday", "month"
        ]
        if c in feature_cols
    ]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def catboost_forecast_optimized(train_df, val_df, target_col, random_state=42):
    from catboost import CatBoostRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()

        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag2"] = grp.shift(2)
        df["lag3"] = grp.shift(3)
        df["lag7"] = grp.shift(7)
        df["lag14"] = grp.shift(14)
        df["lag28"] = grp.shift(28)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling14"] = (
            grp.shift(1)
            .rolling(14, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std7"] = (
            grp.shift(1)
            .rolling(7, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["rolling_std28"] = (
            grp.shift(1)
            .rolling(28, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["diff1"] = df["lag1"] - df["lag2"]
        df["diff7"] = df["lag7"] - df["lag14"]

        df["ratio_lag1_rolling7"] = df["lag1"] / (df["rolling7"] + 1e-6)
        df["ratio_lag7_rolling28"] = df["lag7"] / (df["rolling28"] + 1e-6)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "lag1",
        "lag2",
        "lag3",
        "lag7",
        "lag14",
        "lag28",

        "rolling7",
        "rolling14",
        "rolling28",
        "rolling_std7",
        "rolling_std28",

        "diff1",
        "diff7",
        "ratio_lag1_rolling7",
        "ratio_lag7_rolling28",

        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    cat_features = [
        c for c in [
            "series_id",
            "product_id",
            "store_id",
            "city_id",
            "management_group_id",
            "weekday",
            "month",
            "week_of_year",
            "is_weekend",
        ]
        if c in feature_cols
    ]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        loss_function="RMSE",
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def catboost_forecast_optimized_v2(train_df, val_df, target_col, random_state=42):
    from catboost import CatBoostRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0).sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "week_of_year",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    cat_features = [
        c for c in [
            "product_id",
            "store_id",
            "city_id",
            "management_group_id",
            "weekday",
            "month",
            "week_of_year",
            "holiday_flag",
            "activity_flag",
        ]
        if c in feature_cols
    ]

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=500,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=8,
        random_strength=1.5,
        bagging_temperature=0.5,
        border_count=128,
        thread_count=-1,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def catboost_forecast_fast_numeric_v2(train_df, val_df, target_col, random_state=42):
    from catboost import CatBoostRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "week_of_year",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    # Wichtig:
    # Keine cat_features übergeben.
    # CatBoost behandelt alles numerisch -> viel schneller.
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",

        iterations=300,
        learning_rate=0.08,
        depth=6,

        l2_leaf_reg=10,
        random_strength=2.0,
        bagging_temperature=0.3,

        thread_count=-1,
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def hist_gradient_boosting_forecast(train_df, val_df, target_col, random_state=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",

        max_iter=400,
        learning_rate=0.05,

        max_leaf_nodes=63,
        max_depth=10,
        min_samples_leaf=30,

        l2_regularization=1.0,

        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def hist_gradient_boosting_forecast_optimized(train_df, val_df, target_col, random_state=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1,2,3,4,5,6,7,8,9,10,14,21,28,35,42,56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "psd",
    ]

    feature_cols += [f"lag_{lag}" for lag in [1,2,3,4,5,6,7,8,9,10,14,21,28,35,42,56]]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",

        max_iter=700,
        learning_rate=0.035,

        max_leaf_nodes=127,
        max_depth=12,
        min_samples_leaf=25,

        l2_regularization=0.5,

        max_bins=255,

        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def extra_trees_forecast(train_df, val_df, target_col, random_state=42):
    from sklearn.ensemble import ExtraTreesRegressor
    import pandas as pd
    import numpy as np

    train = train_df.copy()
    val_pred = val_df.copy()

    def add_features(df):
        df = df.copy()
        df["dt"] = pd.to_datetime(df["dt"])

        df["weekday"] = df["dt"].dt.weekday
        df["month"] = df["dt"].dt.month
        df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df = df.sort_values(["series_id", "day_idx"])
        grp = df.groupby("series_id")[target_col]

        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]

        for lag in lags:
            df[f"lag_{lag}"] = grp.shift(lag)

        shifted = grp.shift(1)

        for w in [3, 7, 14, 21, 28]:
            df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"roll_std_{w}"] = shifted.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            df[f"roll_min_{w}"] = shifted.rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            df[f"roll_max_{w}"] = shifted.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            df[f"roll_median_{w}"] = shifted.rolling(w, min_periods=1).median().reset_index(level=0, drop=True)

        df["trend_7"] = df["lag_1"] - df["lag_7"]
        df["trend_14"] = df["lag_1"] - df["lag_14"]
        df["trend_28"] = df["lag_1"] - df["lag_28"]

        df["trend_ratio_7"] = df["lag_1"] / (df["lag_7"] + 1)
        df["trend_ratio_14"] = df["lag_1"] / (df["lag_14"] + 1)
        df["trend_ratio_28"] = df["lag_1"] / (df["lag_28"] + 1)

        df["lag1_roll7"] = df["lag_1"] * df["roll_mean_7"]
        df["lag1_roll28"] = df["lag_1"] * df["roll_mean_28"]
        df["lag7_roll7"] = df["lag_7"] * df["roll_mean_7"]

        df["ratio_roll7"] = df["lag_1"] / (df["roll_mean_7"] + 1)
        df["ratio_roll28"] = df["lag_1"] / (df["roll_mean_28"] + 1)

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])
    combined = add_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",

        "weekday",
        "month",
        "week_of_year",
        "is_weekend",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "day_idx",

        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",

        "psd",
    ]

    feature_cols += [
        f"lag_{lag}"
        for lag in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 28, 35, 42, 56]
    ]

    for w in [3, 7, 14, 21, 28]:
        feature_cols += [
            f"roll_mean_{w}",
            f"roll_std_{w}",
            f"roll_min_{w}",
            f"roll_max_{w}",
            f"roll_median_{w}",
        ]

    feature_cols += [
        "trend_7",
        "trend_14",
        "trend_28",
        "trend_ratio_7",
        "trend_ratio_14",
        "trend_ratio_28",
        "lag1_roll7",
        "lag1_roll28",
        "lag7_roll7",
        "ratio_roll7",
        "ratio_roll28",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)
    X_val = val_feat[feature_cols].fillna(0)

    model = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    val_feat["prediction"] = model.predict(X_val)
    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def dlinear(train_df, val_df, target_col, input_size=28, epochs=20, batch_size=512, lr=1e-3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build training sequences
    X = []
    y = []

    for _, group in train_df.groupby("series_id"):

        group = group.sort_values("dt")

        values = group[target_col].to_numpy(np.float32)

        if len(values) <= input_size:
            continue

        for i in range(input_size, len(values)):
            X.append(values[i-input_size:i])
            y.append(values[i])

    X = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
    y = torch.tensor(np.asarray(y), dtype=torch.float32).to(device)

    print(f"Training samples: {len(X):,}")

    class DLinear(nn.Module):

        def __init__(self, seq_len):

            super().__init__()

            self.trend = nn.Linear(seq_len, 1)
            self.seasonal = nn.Linear(seq_len, 1)

        def forward(self, x):

            trend = self.trend(x)
            seasonal = self.seasonal(x)

            return (trend + seasonal).squeeze(-1)

    model = DLinear(input_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    # Train
    model.train()

    n = len(X)

    for epoch in range(epochs):

        perm = torch.randperm(n)

        epoch_loss = 0

        for i in range(0, n, batch_size):

            idx = perm[i:i+batch_size]

            xb = X[idx]
            yb = y[idx]

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(pred, yb)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        print(f"Epoch {epoch+1}/{epochs}  Loss={epoch_loss/n:.5f}")

    # Forecast validation
    model.eval()

    val_pred = val_df.copy()

    predictions = []

    history = {}

    for sid, group in train_df.groupby("series_id"):

        group = group.sort_values("dt")

        history[sid] = list(group[target_col].values)

    with torch.no_grad():

        for row in val_pred.itertuples():

            sid = row.series_id

            hist = history.get(sid, [])

            if len(hist) < input_size:

                pred = train_df[target_col].mean()

            else:

                x = torch.tensor(
                    hist[-input_size:],
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                pred = model(x).item()

            pred = max(pred, 0)

            predictions.append(pred)

            if sid not in history:
                history[sid] = []

            history[sid].append(pred)

    val_pred["prediction"] = predictions

    return val_pred

# TODO 
# Exponential Smoothing (Siehe oben) (Nils)
# DLinear (Nils)
# LSTM (Laura)
# HistGradientBoostingRegressor
# XGBoost Feature Optimized
# Extra Trees
#(( Wenn Zeit keine Rolle spielen würde. Dann wären diese beiden noch spannend: PatchTST (aktueller Forschungsliebling für Zeitreihen) und N-HiTS / N-BEATS ))