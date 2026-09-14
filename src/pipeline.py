"""
PARANÁ · SAN NICOLÁS
Pipeline V11.21

Base del cálculo:
última lectura observada de INA por día argentino.

No reemplaza observaciones por predicciones.
No modifica los pronósticos mensuales guardados.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ina import observed
from src.upstream import get_upstream_history
from src.exogenous import get_exogenous_data
from src.hydrology import analizar_corrientes_san_nicolas
from src.model import train, predict


TZ = ZoneInfo("America/Argentina/Buenos_Aires")

FORECAST_DAYS = 60
MIN_TRAINING_DAYS = 365 * 3
MAX_TRAINING_DAYS = 365 * 15
HYDROLOGY_HISTORY_YEARS = 20


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def naive_datetime(values):
    return pd.to_datetime(
        values, errors="coerce", utc=True
    ).dt.tz_localize(None)


def normalize_frame(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    result = df.copy()

    if "datetime" in result.columns:
        result["datetime"] = naive_datetime(
            result["datetime"]
        )
        result = (
            result.dropna(subset=["datetime"])
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )

    return result


def prepare_sn_observed(df, cutoff=None):
    """
    Devuelve la última lectura válida de cada día argentino.

    datetime contiene la fecha local, sin zona horaria,
    para conservar compatibilidad con el modelo diario.

    El instante real de la última observación se conserva
    en los atributos del DataFrame.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["datetime", "nivel"])

    if "datetime" not in df.columns:
        raise ValueError("INA no devolvió la columna datetime.")

    level_col = next(
        (
            name
            for name in (
                "value",
                "nivel",
                "nivel_san_nicolas",
            )
            if name in df.columns
        ),
        None,
    )

    if level_col is None:
        raise ValueError("INA no devolvió una columna de nivel.")

    values = pd.DataFrame(
        {
            "observed_at": pd.to_datetime(
                df["datetime"],
                errors="coerce",
                utc=True,
            ).dt.tz_convert(TZ),
            "nivel": numeric(df[level_col]),
        }
    )

    values = values.dropna(
        subset=["observed_at", "nivel"]
    )
    values = values[
        np.isfinite(values["nivel"])
    ].copy()

    if cutoff is not None:
        values = values[
            values["observed_at"] < cutoff
        ].copy()

    values = values.sort_values(
        "observed_at", kind="stable"
    )

    if values.empty:
        return pd.DataFrame(columns=["datetime", "nivel"])

    # Fecha civil argentina, no fecha UTC.
    values["datetime"] = (
        values["observed_at"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Una lectura real por día: la última publicada.
    # No se promedian ni se interpolan niveles locales.
    daily = (
        values.drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    last = daily.iloc[-1]
    result = daily[["datetime", "nivel"]].copy()

    result.attrs["observation_timestamp"] = (
        last["observed_at"].isoformat()
    )
    result.attrs["observation_level"] = float(last["nivel"])

    return result


def history_until(df, last_day):
    result = normalize_frame(df)

    if result.empty or "datetime" not in result.columns:
        return pd.DataFrame(columns=["datetime"])

    return result[
        result["datetime"] < last_day + pd.Timedelta(days=1)
    ].copy().reset_index(drop=True)


def calculate(base_date, training_years=8, visible_days=120):
    requested = pd.Timestamp(base_date)

    if requested.tzinfo is not None:
        requested = requested.tz_convert(TZ)

    requested_day = requested.date()
    now = pd.Timestamp(datetime.now(TZ))

    if requested_day > now.date():
        raise ValueError(
            "La fecha de consulta no puede estar en el futuro."
        )

    requested_ts = pd.Timestamp(requested_day)

    # Se consulta hasta el día siguiente para incluir
    # todas las observaciones de la fecha solicitada.
    query_end = requested_ts + pd.Timedelta(days=1)

    cutoff = min(
        query_end.tz_localize(TZ),
        now,
    )

    hydrology_start = (
        requested_ts
        - pd.DateOffset(years=HYDROLOGY_HISTORY_YEARS)
    ).normalize()

    sn_raw, sn_error = observed(
        hydrology_start.strftime("%Y-%m-%d"),
        query_end.strftime("%Y-%m-%d"),
    )

    if sn_error:
        raise RuntimeError(
            f"No se pudo consultar el nivel oficial INA: {sn_error}"
        )

    sn_hydrology_history = prepare_sn_observed(
        sn_raw, cutoff=cutoff
    )

    if sn_hydrology_history.empty:
        raise RuntimeError(
            "INA no devolvió observaciones válidas anteriores "
            "al momento de consulta. No se generó un pronóstico."
        )

    observation_timestamp = (
        sn_hydrology_history.attrs["observation_timestamp"]
    )
    observation_level = (
        sn_hydrology_history.attrs["observation_level"]
    )

    # El calendario del pronóstico parte del dato real.
    # No se cambia su fecha para hacerlo parecer actual.
    base_ts = pd.Timestamp(
        sn_hydrology_history["datetime"].iloc[-1]
    )

    if not -2.0 <= observation_level <= 12.0:
        raise RuntimeError(
            "La última lectura INA está fuera del intervalo "
            "admitido por el modelo. Se requiere revisarla; "
            "no se sustituirá por un dato anterior."
        )

    training_days = int(
        np.clip(
            float(training_years) * 365,
            MIN_TRAINING_DAYS,
            MAX_TRAINING_DAYS,
        )
    )

    training_start = base_ts - pd.Timedelta(
        days=training_days
    )
    visible_start = requested_ts - pd.Timedelta(
        days=int(visible_days)
    )

    sn_history = sn_hydrology_history[
        sn_hydrology_history["datetime"] >= training_start
    ].copy().reset_index(drop=True)

    upstream_raw, upstream_meta = get_upstream_history(
        hydrology_start.strftime("%Y-%m-%d"),
        (base_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    upstream_hydrology_history = history_until(
        upstream_raw, base_ts
    )

    upstream_history = upstream_hydrology_history[
        upstream_hydrology_history["datetime"] >= training_start
    ].copy().reset_index(drop=True)

    sn_levels_for_flow = sn_history.rename(
        columns={"nivel": "nivel_san_nicolas"}
    )

    level_history_for_flow = upstream_history.drop(
        columns=["nivel_san_nicolas"],
        errors="ignore",
    ).copy()

    if level_history_for_flow.empty:
        level_history_for_flow = sn_levels_for_flow.copy()
    else:
        level_history_for_flow = level_history_for_flow.merge(
            sn_levels_for_flow,
            on="datetime",
            how="outer",
        ).sort_values("datetime")

    exog_history, exog_future, exog_meta = get_exogenous_data(
        training_start.strftime("%Y-%m-%d"),
        base_ts.strftime("%Y-%m-%d"),
        forecast_days=FORECAST_DAYS,
        level_history=level_history_for_flow,
    )

    exog_history = history_until(exog_history, base_ts)
    exog_future = normalize_frame(exog_future)

    hydrology = analizar_corrientes_san_nicolas(
        sn_hydrology_history,
        upstream_hydrology_history,
        exog_history=exog_history,
        exog_future=exog_future,
        days=FORECAST_DAYS,
    )

    models, metrics = train(
        sn_history,
        exog_history=exog_history,
        upstream_history=upstream_history,
        hydrology=hydrology,
    )

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
        raise RuntimeError("El modelo no generó pronóstico.")

    required = {"datetime", "prediction", "base_level"}
    if not required.issubset(forecast.columns):
        raise RuntimeError(
            "La salida del modelo no permite verificar "
            "la fecha y el nivel inicial del pronóstico."
        )

    expected_dates = pd.date_range(
        base_ts + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    if (
        len(forecast) != FORECAST_DAYS
        or not pd.DatetimeIndex(
            forecast["datetime"]
        ).equals(expected_dates)
    ):
        raise RuntimeError(
            "Las fechas del pronóstico no corresponden "
            "a los días posteriores a la observación INA."
        )

    predicted = numeric(forecast["prediction"])
    if not np.isfinite(predicted).all():
        raise RuntimeError(
            "El modelo produjo niveles no válidos."
        )

    model_base = safe_float(
        forecast.iloc[0]["base_level"]
    )

    if not np.isclose(
        model_base,
        observation_level,
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "El modelo no comenzó desde la última lectura INA. "
            "El resultado no se guardará."
        )

    forecast["origin_observed_level"] = observation_level
    forecast["origin_observed_at"] = observation_timestamp

    serializable_models = {
        key: value
        for key, value in models.items()
        if key != "model"
    }

    age_days = (requested_ts - base_ts).days

    return {
        "sn_history": sn_history,
        "upstream_history": upstream_history,
        "upstream_meta": upstream_meta,
        "exog_history": exog_history,
        "exog_future": exog_future,
        "exog_meta": exog_meta,
        "hydrology": hydrology,
        "models": serializable_models,
        "metrics": metrics,
        "forecast": forecast,
        "visible_start": visible_start,
        "base_date": base_ts,
        "requested_date": requested_ts,
        "last_update": datetime.now(TZ),
        "data_loaded": True,
        "load_error": None,
        "pipeline_version": "V11.21",
        "observation_source": "INA",
        "observation_timestamp": observation_timestamp,
        "observation_level": observation_level,
        "observation_age_days": age_days,
        "observation_is_today": age_days == 0,
        "base_description": (
            "Última lectura oficial observada de INA. "
            "No es una predicción."
        ),
    }
