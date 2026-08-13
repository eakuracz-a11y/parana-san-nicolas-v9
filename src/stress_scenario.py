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

# Día futuro donde hacemos coincidir:
# lluvia máxima + caudal máximo + niveles altos aguas arriba
FORCING_PEAK_DAY = 10

MAX_LEVEL = 7.0


SCENARIOS = {

    "alto": {
        "label": "Alto",
        "quantile": 0.90,
        "rain_factor": 0.90,
        "flow_factor": 0.90,
        "upstream_quantile": 0.90,
        "uncertainty": 1.15,
    },

    "severo": {
        "label": "Severo",
        "quantile": 0.95,
        "rain_factor": 1.00,
        "flow_factor": 0.95,
        "upstream_quantile": 0.95,
        "uncertainty": 1.40,
    },

    "extremo": {
        "label": "Extremo histórico",
        "quantile": 1.00,
        "rain_factor": 1.00,
        "flow_factor": 1.00,
        "upstream_quantile": 1.00,
        "uncertainty": 1.70,
    },
}


# ============================================================
# LLUVIA
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
        "max_3d_date": None,
        "max_7d": 0.0,
        "max_7d_date": None,
        "worst_7d_pattern": (
            np.zeros(
                7,
                dtype=float,
            )
        ),
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

    work[
        "datetime"
    ] = pd.to_datetime(
        work[
            "datetime"
        ],
        errors="coerce",
    )

    work[
        "precip_mm"
    ] = pd.to_numeric(
        work[
            "precip_mm"
        ],
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
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if work.empty:

        return result

    rain = work[
        "precip_mm"
    ].clip(
        lower=0
    )

    result[
        "count"
    ] = len(
        rain
    )

    result[
        "p90_day"
    ] = float(
        rain.quantile(
            0.90
        )
    )

    result[
        "p95_day"
    ] = float(
        rain.quantile(
            0.95
        )
    )

    idx_day = rain.idxmax()

    result[
        "max_day"
    ] = float(
        rain.loc[
            idx_day
        ]
    )

    result[
        "max_day_date"
    ] = work.loc[
        idx_day,
        "datetime",
    ]

    # ========================================================
    # 3 DÍAS
    # ========================================================

    rolling3 = (
        rain
        .rolling(
            3,
            min_periods=3,
        )
        .sum()
    )

    if rolling3.notna().any():

        idx3 = rolling3.idxmax()

        result[
            "max_3d"
        ] = float(
            rolling3.loc[
                idx3
            ]
        )

        result[
            "max_3d_date"
        ] = work.loc[
            idx3,
            "datetime",
        ]

    # ========================================================
    # 7 DÍAS
    # ========================================================

    rolling7 = (
        rain
        .rolling(
            7,
            min_periods=7,
        )
        .sum()
    )

    if rolling7.notna().any():

        idx7 = rolling7.idxmax()

        result[
            "max_7d"
        ] = float(
            rolling7.loc[
                idx7
            ]
        )

        result[
            "max_7d_date"
        ] = work.loc[
            idx7,
            "datetime",
        ]

        start = (
            idx7
            - 6
        )

        if start >= 0:

            pattern = (
                rain.iloc[
                    start:
                    idx7 + 1
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if len(
                pattern
            ) == 7:

                result[
                    "worst_7d_pattern"
                ] = pattern

    # ========================================================
    # FALLBACK
    # ========================================================

    if (
        np.nansum(
            result[
                "worst_7d_pattern"
            ]
        )
        <= 0
    ):

        pattern = np.zeros(
            7,
            dtype=float,
        )

        pattern[
            3
        ] = result[
            "max_day"
        ]

        result[
            "worst_7d_pattern"
        ] = pattern

    return result


# ============================================================
# CAUDAL
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

    work[
        "datetime"
    ] = pd.to_datetime(
        work[
            "datetime"
        ],
        errors="coerce",
    )

    work[
        "caudal_m3s"
    ] = pd.to_numeric(
        work[
            "caudal_m3s"
        ],
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
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    if work.empty:

        return result

    q = work[
        "caudal_m3s"
    ]

    result[
        "current"
    ] = float(
        q.iloc[-1]
    )

    result[
        "p90"
    ] = float(
        q.quantile(
            0.90
        )
    )

    result[
        "p95"
    ] = float(
        q.quantile(
            0.95
        )
    )

    idx = q.idxmax()

    result[
        "maximum"
    ] = float(
        q.loc[
            idx
        ]
    )

    result[
        "maximum_date"
    ] = work.loc[
        idx,
        "datetime",
    ]

    return result


# ============================================================
# AGUAS ARRIBA
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
    ][
        "upstream_quantile"
    ]

    columns = [
        c
        for c in upstream_history.columns
        if c.startswith(
            "nivel_"
        )
    ]

    for col in columns:

        values = (
            pd.to_numeric(
                upstream_history[
                    col
                ],
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

        result[
            col
        ] = target

    return result


# ============================================================
# SELECCIONAR LLUVIA DEL ESCENARIO
# ============================================================

def select_rain_values(
    stats,
    scenario,
):

    config = SCENARIOS[
        scenario
    ]

    if scenario == "alto":

        rain_day = (
            stats[
                "p90_day"
            ]
        )

        rain_3d = min(
            stats[
                "max_3d"
            ],
            stats[
                "max_7d"
            ],
        )

        rain_7d = (
            stats[
                "max_7d"
            ]
            * 0.90
        )

    elif scenario == "severo":

        rain_day = (
            stats[
                "p95_day"
            ]
        )

        rain_3d = (
            stats[
                "max_3d"
            ]
        )

        rain_7d = (
            stats[
                "max_7d"
            ]
        )

    else:

        rain_day = (
            stats[
                "max_day"
            ]
        )

        rain_3d = (
            stats[
                "max_3d"
            ]
        )

        rain_7d = (
            stats[
                "max_7d"
            ]
        )

    return {
        "day": max(
            rain_day
            * config[
                "rain_factor"
            ],
            0.0,
        ),
        "three": max(
            rain_3d
            * config[
                "rain_factor"
            ],
            0.0,
        ),
        "seven": max(
            rain_7d
            * config[
                "rain_factor"
            ],
            0.0,
        ),
    }


# ============================================================
# CAUDAL DEL ESCENARIO
# ============================================================

def select_flow_peak(
    stats,
    scenario,
):

    config = SCENARIOS[
        scenario
    ]

    current = stats[
        "current"
    ]

    if scenario == "alto":

        target = stats[
            "p90"
        ]

    elif scenario == "severo":

        target = stats[
            "p95"
        ]

    else:

        target = stats[
            "maximum"
        ]

    if not np.isfinite(
        current
    ):

        current = target

    if not np.isfinite(
        target
    ):

        target = current

    return float(
        current
        + (
            target
            - current
        )
        * config[
            "flow_factor"
        ]
    )


# ============================================================
# PERFIL DE LLUVIA 60 DÍAS
# ============================================================

def build_rain_profile(
    rain_stats,
    scenario,
    days,
):

    config = SCENARIOS[
        scenario
    ]

    profile = np.zeros(
        days,
        dtype=float,
    )

    pattern = np.array(
        rain_stats[
            "worst_7d_pattern"
        ],
        dtype=float,
    )

    pattern = (
        pattern
        * config[
            "rain_factor"
        ]
    )

    peak_index = int(
        np.argmax(
            pattern
        )
    )

    start = (
        FORCING_PEAK_DAY
        - peak_index
        - 1
    )

    for i, value in enumerate(
        pattern
    ):

        pos = (
            start
            + i
        )

        if (
            0
            <= pos
            < days
        ):

            profile[
                pos
            ] = max(
                float(
                    value
                ),
                0.0,
            )

    return profile


# ============================================================
# PERFIL DEL CAUDAL
# ============================================================

def build_flow_profile(
    current,
    peak,
    days,
):

    if (
        not np.isfinite(
            current
        )
        or not np.isfinite(
            peak
        )
    ):

        return np.full(
            days,
            np.nan,
        )

    values = []

    for day in range(
        1,
        days + 1,
    ):

        if day <= FORCING_PEAK_DAY:

            fraction = (
                day
                / FORCING_PEAK_DAY
            )

            smooth = (
                3
                * fraction ** 2
                - 2
                * fraction ** 3
            )

            value = (
                current
                + (
                    peak
                    - current
                )
                * smooth
            )

        elif day <= (
            FORCING_PEAK_DAY
            + 5
        ):

            value = peak

        else:

            elapsed = (
                day
                - (
                    FORCING_PEAK_DAY
                    + 5
                )
            )

            value = (
                current
                + (
                    peak
                    - current
                )
                * np.exp(
                    -elapsed
                    / 20.0
                )
            )

        values.append(
            max(
                float(
                    value
                ),
                0.0,
            )
        )

    return np.array(
        values,
        dtype=float,
    )


# ============================================================
# HIDROGRAMA DEL NIVEL
# ============================================================

def build_level_profile(
    current_level,
    peak_level,
    forcing_peak_day,
    response_lag,
    days,
):

    level_peak_day = min(
        forcing_peak_day
        + response_lag,
        days,
    )

    rise_start = max(
        1,
        forcing_peak_day
        - 4,
    )

    growth = max(
        peak_level
        - current_level,
        0.0,
    )

    values = []

    for day in range(
        1,
        days + 1,
    ):

        # ====================================================
        # ANTES DEL INICIO DE RESPUESTA
        # ====================================================

        if day < rise_start:

            level = (
                current_level
            )

        # ====================================================
        # CRECIMIENTO
        # ====================================================

        elif day <= level_peak_day:

            denominator = max(
                level_peak_day
                - rise_start,
                1,
            )

            fraction = (
                day
                - rise_start
            ) / denominator

            fraction = np.clip(
                fraction,
                0.0,
                1.0,
            )

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

        # ====================================================
        # RECESIÓN
        # ====================================================

        else:

            elapsed = (
                day
                - level_peak_day
            )

            remaining_growth = (
                growth
                * np.exp(
                    -elapsed
                    / 24.0
                )
            )

            level = (
                current_level
                + remaining_growth
            )

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
        level_peak_day,
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

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Escenario desconocido: {scenario}"
        )

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
            "No existe dataset del modelo principal."
        )

    days = max(
        30,
        min(
            int(
                days
            ),
            60,
        ),
    )

    dataset = models[
        "dataset"
    ].copy()

    dataset[
        "datetime"
    ] = pd.to_datetime(
        dataset[
            "datetime"
        ],
        errors="coerce",
    )

    dataset = dataset.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if dataset.empty:

        raise ValueError(
            "El dataset histórico está vacío."
        )

    current_level = float(
        pd.to_numeric(
            dataset[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
        .iloc[-1]
    )

    last_date = (
        dataset[
            "datetime"
        ]
        .max()
        .normalize()
    )

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    rain_stats = rainfall_statistics(
        exog_history
    )

    flow_stats = flow_statistics(
        exog_history
    )

    rain_values = select_rain_values(
        rain_stats,
        scenario,
    )

    flow_peak = select_flow_peak(
        flow_stats,
        scenario,
    )

    upstream = upstream_targets(
        upstream_history,
        scenario,
    )

    # ========================================================
    # ENTRENAR RESPUESTA A CRECIENTES
    # ========================================================

    response_model = (
        fit_flood_response(
            dataset
        )
    )

    response = (
        predict_stress_growth(
            response_model=(
                response_model
            ),
            dataset=dataset,
            rain_day=(
                rain_values[
                    "day"
                ]
            ),
            rain_3d=(
                rain_values[
                    "three"
                ]
            ),
            rain_7d=(
                rain_values[
                    "seven"
                ]
            ),
            flow_peak=flow_peak,
            upstream_targets=upstream,
        )
    )

    predicted_growth = response[
        "growth"
    ]

    peak_level = response[
        "peak_level"
    ]

    response_lag = response[
        "response_lag"
    ]

    # ========================================================
    # PERFILES TEMPORALES
    # ========================================================

    rain_profile = (
        build_rain_profile(
            rain_stats,
            scenario,
            days,
        )
    )

    flow_profile = (
        build_flow_profile(
            current=(
                flow_stats[
                    "current"
                ]
            ),
            peak=flow_peak,
            days=days,
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
        forcing_peak_day=(
            FORCING_PEAK_DAY
        ),
        response_lag=(
            response_lag
        ),
        days=days,
    )

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

    uncertainty_factor = SCENARIOS[
        scenario
    ][
        "uncertainty"
    ]

    rmse = max(
        response[
            "rmse"
        ],
        0.05,
    )

    lower = []

    upper = []

    for i, level in enumerate(
        level_profile,
        start=1,
    ):

        # Más incertidumbre después del máximo forzante
        horizon_factor = (
            1.0
            + 0.035
            * i
        )

        margin = (
            1.96
            * rmse
            * uncertainty_factor
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

    scenario_df = pd.DataFrame(
        {
            "datetime": dates,
            "prediction": (
                level_profile
            ),
            "lower": lower,
            "upper": upper,
            "precip_mm": (
                rain_profile
            ),
            "caudal_m3s": (
                flow_profile
            ),
            "horizon_day": (
                np.arange(
                    1,
                    days + 1,
                )
            ),
            "scenario": (
                SCENARIOS[
                    scenario
                ][
                    "label"
                ]
            ),
        }
    )

    # ========================================================
    # FECHAS IMPORTANTES
    # ========================================================

    forcing_peak_date = (
        last_date
        + pd.Timedelta(
            days=FORCING_PEAK_DAY
        )
    )

    peak_level_date = (
        last_date
        + pd.Timedelta(
            days=level_peak_day
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    growth_pct = (
        predicted_growth
        / current_level
        * 100
        if current_level
        else np.nan
    )

    metadata = {
        "scenario": (
            SCENARIOS[
                scenario
            ][
                "label"
            ]
        ),
        "current_level": (
            current_level
        ),
        "max_level": (
            peak_level
        ),
        "growth_m": (
            predicted_growth
        ),
        "growth_pct": (
            growth_pct
        ),
        "peak_future_date": (
            forcing_peak_date
        ),
        "max_level_date": (
            peak_level_date
        ),
        "response_lag_days": (
            response_lag
        ),
        "rain_peak_scenario": (
            rain_values[
                "day"
            ]
        ),
        "rain_3d_scenario": (
            rain_values[
                "three"
            ]
        ),
        "rain_7d_scenario": (
            rain_values[
                "seven"
            ]
        ),
        "rain_event_total": float(
            np.sum(
                rain_profile
            )
        ),
        "flow_scenario_max": (
            flow_peak
        ),
        "rain_stats": (
            rain_stats
        ),
        "flow_stats": (
            flow_stats
        ),
        "level_day_30": float(
            scenario_df[
                "prediction"
            ].iloc[
                29
            ]
        ),
        "level_day_60": float(
            scenario_df[
                "prediction"
            ].iloc[
                59
            ]
        ),
        "flood_model_rmse": (
            response[
                "rmse"
            ]
        ),
        "flood_model_mae": (
            response[
                "mae"
            ]
        ),
        "flood_training_rows": (
            response[
                "training_rows"
            ]
        ),
        "historical_max_growth": (
            response[
                "historical_max_growth"
            ]
        ),
        "historical_p95_growth": (
            response[
                "historical_p95_growth"
            ]
        ),
    }

    return (
        scenario_df,
        metadata,
    )
