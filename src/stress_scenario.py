import numpy as np
import pandas as pd

from src.flood_response import (
    fit_flood_response,
    predict_stress_growth,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_STRESS_DAYS = 60
MAX_LEVEL = 7.0


SCENARIOS = {

    "alto": {
        "label": "Alto",
        "rain_quantile": 0.90,
        "flow_quantile": 0.90,
        "upstream_quantile": 0.90,
        "uncertainty": 1.15,
    },

    "severo": {
        "label": "Severo",
        "rain_quantile": 0.95,
        "flow_quantile": 0.95,
        "upstream_quantile": 0.95,
        "uncertainty": 1.40,
    },

    "extremo": {
        "label": "Extremo histórico",
        "rain_quantile": 1.00,
        "flow_quantile": 1.00,
        "upstream_quantile": 1.00,
        "uncertainty": 1.70,
    },
}


# ============================================================
# ESTADÍSTICAS DE PRECIPITACIÓN HISTÓRICA
# ============================================================

def rainfall_statistics(
    exog_history,
):

    result = {
        "count": 0,
        "p90_day": 0.0,
        "p95_day": 0.0,
        "max_day": 0.0,
        "max_day_date": None,
        "max_3d": 0.0,
        "max_7d": 0.0,
    }

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
        or "precip_mm"
        not in exog_history.columns
    ):
        return result

    work = exog_history.copy()

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        errors="coerce",
    )

    work["precip_mm"] = pd.to_numeric(
        work["precip_mm"],
        errors="coerce",
    )

    work = (
        work
        .dropna(
            subset=[
                "datetime",
                "precip_mm",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if work.empty:
        return result

    rain = (
        work["precip_mm"]
        .clip(lower=0)
    )

    result["count"] = len(rain)

    result["p90_day"] = float(
        rain.quantile(0.90)
    )

    result["p95_day"] = float(
        rain.quantile(0.95)
    )

    idx_max = rain.idxmax()

    result["max_day"] = float(
        rain.loc[idx_max]
    )

    result["max_day_date"] = (
        work.loc[
            idx_max,
            "datetime",
        ]
    )

    rolling_3 = (
        rain
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    rolling_7 = (
        rain
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    result["max_3d"] = float(
        rolling_3.max()
    )

    result["max_7d"] = float(
        rolling_7.max()
    )

    return result


# ============================================================
# ESTADÍSTICAS DE CAUDAL HISTÓRICO
# ============================================================

def flow_statistics(
    exog_history,
):

    result = {
        "current": np.nan,
        "p90": np.nan,
        "p95": np.nan,
        "maximum": np.nan,
        "maximum_date": None,
    }

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
        or "caudal_m3s"
        not in exog_history.columns
    ):
        return result

    work = exog_history.copy()

    work["datetime"] = pd.to_datetime(
        work["datetime"],
        errors="coerce",
    )

    work["caudal_m3s"] = pd.to_numeric(
        work["caudal_m3s"],
        errors="coerce",
    )

    work = (
        work
        .dropna(
            subset=[
                "datetime",
                "caudal_m3s",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if work.empty:
        return result

    q = work["caudal_m3s"]

    result["current"] = float(
        q.iloc[-1]
    )

    result["p90"] = float(
        q.quantile(0.90)
    )

    result["p95"] = float(
        q.quantile(0.95)
    )

    idx_max = q.idxmax()

    result["maximum"] = float(
        q.loc[idx_max]
    )

    result["maximum_date"] = (
        work.loc[
            idx_max,
            "datetime",
        ]
    )

    return result


# ============================================================
# MÁXIMOS / PERCENTILES DE ESTACIONES AGUAS ARRIBA
# ============================================================

def upstream_targets(
    upstream_history,
    scenario,
):

    result = {}

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
    ):
        return result

    quantile = SCENARIOS[
        scenario
    ]["upstream_quantile"]

    columns = [
        c
        for c in upstream_history.columns
        if c.startswith("nivel_")
    ]

    for col in columns:

        values = (
            pd.to_numeric(
                upstream_history[col],
                errors="coerce",
            )
            .dropna()
        )

        if values.empty:
            continue

        if quantile >= 1.0:

            target = float(
                values.max()
            )

        else:

            target = float(
                values.quantile(
                    quantile
                )
            )

        result[col] = target

    return result


# ============================================================
# LLUVIA DIARIA DEL ESCENARIO
# ============================================================

def select_daily_rain(
    rain_stats,
    scenario,
):

    if scenario == "alto":

        value = rain_stats[
            "p90_day"
        ]

    elif scenario == "severo":

        value = rain_stats[
            "p95_day"
        ]

    else:

        value = rain_stats[
            "max_day"
        ]

    if (
        value is None
        or not np.isfinite(value)
    ):

        value = 0.0

    return max(
        float(value),
        0.0,
    )


# ============================================================
# CAUDAL DIARIO DEL ESCENARIO
# ============================================================

def select_daily_flow(
    flow_stats,
    scenario,
):

    if scenario == "alto":

        value = flow_stats[
            "p90"
        ]

    elif scenario == "severo":

        value = flow_stats[
            "p95"
        ]

    else:

        value = flow_stats[
            "maximum"
        ]

    if (
        value is None
        or not np.isfinite(value)
    ):

        value = flow_stats.get(
            "current",
            np.nan,
        )

    if (
        value is None
        or not np.isfinite(value)
    ):

        return np.nan

    return float(value)


# ============================================================
# PERFIL DE PRECIPITACIÓN
# ============================================================

def build_rain_profile(
    daily_rain,
    days,
):

    """
    Mantiene el valor de lluvia seleccionado
    durante todos los días del escenario.
    """

    return np.full(
        days,
        float(daily_rain),
        dtype=float,
    )


# ============================================================
# PERFIL DE CAUDAL
# ============================================================

def build_flow_profile(
    daily_flow,
    days,
):

    """
    Mantiene el caudal seleccionado
    durante todos los días del escenario.
    """

    if (
        daily_flow is None
        or not np.isfinite(
            daily_flow
        )
    ):

        return np.full(
            days,
            np.nan,
            dtype=float,
        )

    return np.full(
        days,
        float(daily_flow),
        dtype=float,
    )


# ============================================================
# PERFIL FUTURO DEL NIVEL
# ============================================================

def build_level_profile(
    current_level,
    peak_level,
    response_lag,
    days,
):

    """
    Construye un crecimiento progresivo
    desde el nivel actual hasta el máximo
    estimado por el modelo de crecientes.

    Como las condiciones extremas siguen
    presentes los 60 días, después del pico
    se mantiene una meseta.
    """

    response_lag = int(
        max(
            response_lag,
            1,
        )
    )

    growth = max(
        peak_level
        - current_level,
        0.0,
    )

    peak_day = int(
        np.clip(
            response_lag
            + 7,
            7,
            30,
        )
    )

    values = []

    for day in range(
        1,
        days + 1,
    ):

        if day <= peak_day:

            fraction = (
                day
                / peak_day
            )

            # Curva sigmoidal suave
            smooth = (
                3
                * fraction ** 2
                - 2
                * fraction ** 3
            )

            level = (
                current_level
                + growth
                * smooth
            )

        else:

            # Condición extrema sostenida
            level = peak_level

        values.append(
            float(
                np.clip(
                    level,
                    0.0,
                    MAX_LEVEL,
                )
            )
        )

    return (
        np.array(
            values,
            dtype=float,
        ),
        peak_day,
    )


# ============================================================
# ESCENARIO COMPLETO
# ============================================================

def build_stress_scenario(
    models,
    exog_history=None,
    upstream_history=None,
    days=DEFAULT_STRESS_DAYS,
    scenario="extremo",
):

    # ========================================================
    # VALIDAR ESCENARIO
    # ========================================================

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Escenario desconocido: {scenario}"
        )

    # ========================================================
    # VALIDAR MODELO
    # ========================================================

    if (
        models is None
        or not isinstance(
            models,
            dict,
        )
        or "dataset"
        not in models
    ):

        raise ValueError(
            "No existe dataset histórico del modelo."
        )

    # ========================================================
    # VALIDAR HORIZONTE
    # ========================================================

    days = int(
        np.clip(
            int(days),
            1,
            60,
        )
    )

    # ========================================================
    # DATASET HISTÓRICO
    # ========================================================

    dataset = models[
        "dataset"
    ].copy()

    dataset["datetime"] = pd.to_datetime(
        dataset["datetime"],
        errors="coerce",
    )

    dataset["nivel"] = pd.to_numeric(
        dataset["nivel"],
        errors="coerce",
    )

    dataset = (
        dataset
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if dataset.empty:

        raise ValueError(
            "El histórico de niveles está vacío."
        )

    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    current_level = float(
        dataset[
            "nivel"
        ].iloc[-1]
    )

    last_date = (
        dataset[
            "datetime"
        ]
        .max()
        .normalize()
    )

    # ========================================================
    # ESTADÍSTICAS HISTÓRICAS
    # ========================================================

    rain_stats = rainfall_statistics(
        exog_history
    )

    flow_stats = flow_statistics(
        exog_history
    )

    # ========================================================
    # ELEGIR CONDICIÓN SEGÚN ESCENARIO
    # ========================================================

    daily_rain = select_daily_rain(
        rain_stats,
        scenario,
    )

    daily_flow = select_daily_flow(
        flow_stats,
        scenario,
    )

    upstream = upstream_targets(
        upstream_history,
        scenario,
    )

    # ========================================================
    # ACUMULADOS DE LLUVIA
    #
    # Si usamos máximo diario todos los días:
    # 3 días = max diario × 3
    # 7 días = max diario × 7
    # ========================================================

    rain_3d = (
        daily_rain
        * 3.0
    )

    rain_7d = (
        daily_rain
        * 7.0
    )

    # ========================================================
    # ENTRENAR MODELO DE RESPUESTA A CRECIENTES
    # ========================================================

    response_model = (
        fit_flood_response(
            dataset
        )
    )

    # ========================================================
    # CALCULAR CRECIMIENTO POTENCIAL
    # ========================================================

    response = (
        predict_stress_growth(
            response_model=(
                response_model
            ),
            dataset=dataset,
            rain_day=(
                daily_rain
            ),
            rain_3d=(
                rain_3d
            ),
            rain_7d=(
                rain_7d
            ),
            flow_peak=(
                daily_flow
            ),
            upstream_targets=(
                upstream
            ),
        )
    )

    predicted_growth = float(
        response[
            "growth"
        ]
    )

    peak_level = float(
        response[
            "peak_level"
        ]
    )

    response_lag = int(
        response[
            "response_lag"
        ]
    )

    # ========================================================
    # PERFILES PARA LOS 60 DÍAS
    # ========================================================

    rain_profile = (
        build_rain_profile(
            daily_rain,
            days,
        )
    )

    flow_profile = (
        build_flow_profile(
            daily_flow,
            days,
        )
    )

    (
        level_profile,
        level_peak_day,
    ) = build_level_profile(
        current_level=(
            current_level
        ),
        peak_level=(
            peak_level
        ),
        response_lag=(
            response_lag
        ),
        days=days,
    )

    # ========================================================
    # FECHAS FUTURAS
    # ========================================================

    dates = pd.date_range(
        start=(
            last_date
            + pd.Timedelta(
                days=1
            )
        ),
        periods=days,
        freq="D",
    )

    # ========================================================
    # INCERTIDUMBRE
    # ========================================================

    config = SCENARIOS[
        scenario
    ]

    rmse = max(
        float(
            response[
                "rmse"
            ]
        ),
        0.05,
    )

    lower = []
    upper = []

    for i, level in enumerate(
        level_profile,
        start=1,
    ):

        horizon_factor = (
            1.0
            + 0.025
            * i
        )

        margin = (
            1.96
            * rmse
            * config[
                "uncertainty"
            ]
            * horizon_factor
        )

        lower.append(
            max(
                0.0,
                level
                - margin,
            )
        )

        upper.append(
            min(
                MAX_LEVEL,
                level
                + margin,
            )
        )

    # ========================================================
    # DATAFRAME FINAL
    # ========================================================

    scenario_df = pd.DataFrame(
        {

            "datetime":
                dates,

            "prediction":
                level_profile,

            "lower":
                lower,

            "upper":
                upper,

            "precip_mm":
                rain_profile,

            "caudal_m3s":
                flow_profile,

            "horizon_day":
                np.arange(
                    1,
                    days + 1,
                ),

            "scenario":
                config[
                    "label"
                ],
        }
    )

    # ========================================================
    # FECHA DEL MÁXIMO DE NIVEL
    # ========================================================

    peak_level_date = (
        last_date
        + pd.Timedelta(
            days=level_peak_day
        )
    )

    # ========================================================
    # CRECIMIENTO %
    # ========================================================

    growth_pct = (
        predicted_growth
        / current_level
        * 100
        if current_level != 0
        else np.nan
    )

    # ========================================================
    # ACUMULADO DEL ESCENARIO COMPLETO
    # ========================================================

    total_rain_60d = float(
        daily_rain
        * days
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "scenario":
            config[
                "label"
            ],

        "current_level":
            current_level,

        "max_level":
            peak_level,

        "growth_m":
            predicted_growth,

        "growth_pct":
            growth_pct,

        # Primer día del escenario.
        # Las condiciones máximas se aplican
        # desde aquí y durante los 60 días.
        "peak_future_date":
            dates[0],

        "max_level_date":
            peak_level_date,

        "response_lag_days":
            response_lag,

        "rain_peak_scenario":
            daily_rain,

        "rain_3d_scenario":
            rain_3d,

        "rain_7d_scenario":
            rain_7d,

        "rain_event_total":
            total_rain_60d,

        "flow_scenario_max":
            daily_flow,

        "rain_stats":
            rain_stats,

        "flow_stats":
            flow_stats,

        "level_day_30":
            float(
                scenario_df[
                    "prediction"
                ].iloc[
                    min(
                        29,
                        len(
                            scenario_df
                        )
                        - 1,
                    )
                ]
            ),

        "level_day_60":
            float(
                scenario_df[
                    "prediction"
                ].iloc[-1]
            ),

        "flood_model_rmse":
            response[
                "rmse"
            ],

        "flood_model_mae":
            response[
                "mae"
            ],

        "flood_training_rows":
            response[
                "training_rows"
            ],

        "historical_max_growth":
            response[
                "historical_max_growth"
            ],

        "historical_p95_growth":
            response[
                "historical_p95_growth"
            ],

        "constant_max_conditions":
            True,

        "scenario_days":
            days,
    }

    return (
        scenario_df,
        metadata,
    )
