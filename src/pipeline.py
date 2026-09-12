from datetime import datetime

import numpy as np
import pandas as pd

from src.ina import observed
from src.upstream import get_upstream_history
from src.exogenous import get_exogenous_data
from src.hydrology import analizar_corrientes_san_nicolas
from src.model import train, predict


FORECAST_DAYS = 60
MIN_TRAINING_DAYS = 365 * 3
MAX_TRAINING_DAYS = 365 * 15
HYDROLOGY_HISTORY_YEARS = 20


def safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def naive_datetime(values):
    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )


def normalize_frame(df):
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame()

    result = df.copy()

    if "datetime" in result.columns:
        result["datetime"] = naive_datetime(
            result["datetime"]
        )

        result = (
            result
            .dropna(subset=["datetime"])
            .sort_values("datetime")
            .drop_duplicates(
                subset=["datetime"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    return result


def prepare_sn_observed(df):
    df = normalize_frame(df)

    if df.empty:
        return pd.DataFrame()

    level_col = None

    for candidate in [
        "nivel",
        "value",
        "nivel_san_nicolas",
    ]:
        if candidate in df.columns:
            level_col = candidate
            break

    if level_col is None:
        return pd.DataFrame()

    result = df[
        [
            "datetime",
            level_col,
        ]
    ].copy()

    result["nivel"] = numeric(
        result[level_col]
    )

    result = (
        result
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result[
        [
            "datetime",
            "nivel",
        ]
    ]


def calculate(
    base_date,
    training_years=8,
    visible_days=120,
):
    base_ts = pd.Timestamp(
        base_date
    ).normalize()

    # =================================================
    # PERÍODOS
    # =================================================

    visible_start = (
        base_ts
        - pd.Timedelta(days=visible_days)
    )

    training_days = int(
        np.clip(
            training_years * 365,
            MIN_TRAINING_DAYS,
            MAX_TRAINING_DAYS,
        )
    )

    training_start = (
        base_ts
        - pd.Timedelta(days=training_days)
    )

    hydrology_start = (
        base_ts
        - pd.DateOffset(
            years=HYDROLOGY_HISTORY_YEARS
        )
    ).normalize()

    # =================================================
    # SAN NICOLÁS
    # =================================================

    sn_raw, sn_error = observed(
        hydrology_start.strftime("%Y-%m-%d"),
        base_ts.strftime("%Y-%m-%d"),
    )

    if sn_error:
        raise RuntimeError(sn_error)

    sn_hydrology_history = prepare_sn_observed(
        sn_raw
    )

    if sn_hydrology_history.empty:
        raise RuntimeError(
            "INA no devolvió niveles válidos "
            "para San Nicolás."
        )

    sn_history = (
        sn_hydrology_history[
            sn_hydrology_history["datetime"]
            >= training_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    if sn_history.empty:
        sn_history = sn_hydrology_history.copy()

    # =================================================
    # ESTACIONES AGUAS ARRIBA
    # =================================================

    upstream_history, upstream_meta = (
        get_upstream_history(
            hydrology_start.strftime("%Y-%m-%d"),
            base_ts.strftime("%Y-%m-%d"),
        )
    )

    upstream_hydrology_history = normalize_frame(
        upstream_history
    )

    upstream_history = (
        upstream_hydrology_history[
            upstream_hydrology_history["datetime"]
            >= training_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    # =================================================
    # VARIABLES EXÓGENAS
    # =================================================

    level_history_for_flow = (
        upstream_history.copy()
    )

    sn_levels_for_flow = sn_history[
        [
            "datetime",
            "nivel",
        ]
    ].copy()

    sn_levels_for_flow = (
        sn_levels_for_flow.rename(
            columns={
                "nivel": "nivel_san_nicolas",
            }
        )
    )

    if level_history_for_flow.empty:
        level_history_for_flow = (
            sn_levels_for_flow
        )
    else:
        level_history_for_flow = (
            level_history_for_flow.merge(
                sn_levels_for_flow,
                on="datetime",
                how="outer",
            )
        )

    (
        exog_history,
        exog_future,
        exog_meta,
    ) = get_exogenous_data(
        training_start.strftime("%Y-%m-%d"),
        base_ts.strftime("%Y-%m-%d"),
        forecast_days=FORECAST_DAYS,
        level_history=level_history_for_flow,
    )

    exog_history = normalize_frame(
        exog_history
    )

    exog_future = normalize_frame(
        exog_future
    )

    # =================================================
    # HIDROLOGÍA
    # =================================================

    hydrology = analizar_corrientes_san_nicolas(
        sn_hydrology_history,
        upstream_hydrology_history,
        exog_history=exog_history,
        exog_future=exog_future,
        days=FORECAST_DAYS,
    )

    # =================================================
    # ENTRENAMIENTO DEL MODELO ACTUAL
    # =================================================

    models, metrics = train(
        sn_history,
        exog_history=exog_history,
        upstream_history=upstream_history,
        hydrology=hydrology,
    )

    # =================================================
    # PRONÓSTICO A 60 DÍAS
    # =================================================

    forecast = predict(
        sn_history,
        models,
        days=FORECAST_DAYS,
        exog_future=exog_future,
        upstream_future=None,
        hydrology=hydrology,
    )

    forecast = normalize_frame(forecast)

    if forecast.empty:
        raise RuntimeError(
            "El modelo no generó pronóstico."
        )

    # El objeto entrenado ya se utilizó para predecir.
    # Conservamos los datos y diagnósticos necesarios
    # para mostrar el tablero, sin guardar el bosque.
    models = {
        key: value
        for key, value in models.items()
        if key != "model"
    }

    return {
        "sn_history": sn_history,
        "upstream_history": upstream_history,
        "upstream_meta": upstream_meta,
        "exog_history": exog_history,
        "exog_future": exog_future,
        "exog_meta": exog_meta,
        "hydrology": hydrology,
        "models": models,
        "metrics": metrics,
        "forecast": forecast,
        "visible_start": visible_start,
        "base_date": base_ts,
        "last_update": datetime.now(),
        "data_loaded": True,
        "load_error": None,
    }
