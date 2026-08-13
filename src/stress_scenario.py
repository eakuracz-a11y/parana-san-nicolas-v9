import numpy as np
import pandas as pd

from src.model import crear_features


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_STRESS_DAYS = 60

# Día futuro en el que hacemos coincidir
# caudal máximo + máximo de lluvia
PEAK_DAY = 10


SCENARIOS = {

    "alto": {
        "label": "Alto",
        "quantile": 0.90,
        "rain_factor": 0.90,
        "flow_factor": 0.90,
        "uncertainty_factor": 1.20,
    },

    "severo": {
        "label": "Severo",
        "quantile": 0.95,
        "rain_factor": 1.00,
        "flow_factor": 0.95,
        "uncertainty_factor": 1.50,
    },

    "extremo": {
        "label": "Extremo histórico",
        "quantile": 1.00,
        "rain_factor": 1.00,
        "flow_factor": 1.00,
        "uncertainty_factor": 1.80,
    },
}


# ============================================================
# UTILIDADES
# ============================================================

def numeric_series(
    df,
    column,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or column not in df.columns
    ):

        return pd.Series(
            dtype=float
        )

    return (
        pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )
        .dropna()
        .astype(float)
    )


# ============================================================
# LLUVIA HISTÓRICA
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

        "worst_7d_pattern": np.zeros(
            7,
            dtype=float,
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

    # ========================================================
    # MÁXIMO DIARIO
    # ========================================================

    idx_max = rain.idxmax()

    result[
        "max_day"
    ] = float(
        rain.loc[
            idx_max
        ]
    )

    result[
        "max_day_date"
    ] = work.loc[
        idx_max,
        "datetime",
    ]

    # ========================================================
    # MÁXIMO 3 DÍAS
    # ========================================================

    rolling_3 = (
        rain
        .rolling(
            3,
            min_periods=3,
        )
        .sum()
    )

    if rolling_3.notna().any():

        idx3 = rolling_3.idxmax()

        result[
            "max_3d"
        ] = float(
            rolling_3.loc[
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
    # MÁXIMO 7 DÍAS
    # ========================================================

    rolling_7 = (
        rain
        .rolling(
            7,
            min_periods=7,
        )
        .sum()
    )

    if rolling_7.notna().any():

        idx7 = rolling_7.idxmax()

        result[
            "max_7d"
        ] = float(
            rolling_7.loc[
                idx7
            ]
        )

        result[
            "max_7d_date"
        ] = work.loc[
            idx7,
            "datetime",
        ]

        start_idx = (
            idx7
            - 6
        )

        if start_idx >= 0:

            pattern = (
                rain.loc[
                    start_idx:
                    idx7
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
    # FALLBACK SI NO HAY 7 DÍAS COMPLETOS
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

        # El máximo diario queda en el centro
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
# CAUDAL HISTÓRICO
# ============================================================

def flow_statistics(
    exog_history,
):

    result = {

        "count": 0,

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

    flow = work[
        "caudal_m3s"
    ]

    result[
        "count"
    ] = len(
        flow
    )

    result[
        "current"
    ] = float(
        flow.iloc[
            -1
        ]
    )

    result[
        "p90"
    ] = float(
        flow.quantile(
            0.90
        )
    )

    result[
        "p95"
    ] = float(
        flow.quantile(
            0.95
        )
    )

    idx = flow.idxmax()

    result[
        "maximum"
    ] = float(
        flow.loc[
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

def upstream_statistics(
    upstream_history,
    quantile,
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

    level_cols = [
        c
        for c in upstream_history.columns
        if c.startswith(
            "nivel_"
        )
    ]

    for col in level_cols:

        values = numeric_series(
            upstream_history,
            col,
        )

        if values.empty:

            continue

        current = float(
            values.iloc[
                -1
            ]
        )

        if quantile >= 1:

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
        ] = {
            "current": current,
            "target": target,
        }

    return result


# ============================================================
# PERFIL DE LLUVIA FUTURA
# ============================================================

def build_rain_scenario(
    exog_history,
    days,
    scenario,
):

    stats = rainfall_statistics(
        exog_history
    )

    config = SCENARIOS[
        scenario
    ]

    rain = np.zeros(
        days,
        dtype=float,
    )

    pattern = np.array(
        stats[
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

    # ========================================================
    # ALINEAR EL MÁXIMO DE LLUVIA CON PEAK_DAY
    # ========================================================

    if len(
        pattern
    ):

        peak_pattern_index = int(
            np.argmax(
                pattern
            )
        )

    else:

        peak_pattern_index = 3

    start_day = (
        PEAK_DAY
        - peak_pattern_index
        - 1
    )

    for i, value in enumerate(
        pattern
    ):

        pos = (
            start_day
            + i
        )

        if (
            0
            <= pos
            < days
        ):

            rain[
                pos
            ] = max(
                float(
                    value
                ),
                0.0,
            )

    return (
        rain,
        stats,
    )


# ============================================================
# PERFIL DE CAUDAL FUTURO
# ============================================================

def build_flow_scenario(
    exog_history,
    days,
    scenario,
):

    stats = flow_statistics(
        exog_history
    )

    current = stats[
        "current"
    ]

    if not np.isfinite(
        current
    ):

        return (
            np.full(
                days,
                np.nan,
            ),
            stats,
        )

    config = SCENARIOS[
        scenario
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
        target
    ):

        target = current

    # Factor según escenario
    target = (
        current
        + (
            target
            - current
        )
        * config[
            "flow_factor"
        ]
    )

    values = []

    for h in range(
        1,
        days + 1,
    ):

        # ====================================================
        # CRECIMIENTO HASTA PEAK_DAY
        # ====================================================

        if h <= PEAK_DAY:

            fraction = (
                h
                / PEAK_DAY
            )

            # Curva suavizada
            fraction = (
                3
                * fraction ** 2
                - 2
                * fraction ** 3
            )

            value = (
                current
                + (
                    target
                    - current
                )
                * fraction
            )

        # ====================================================
        # MESETA ALTA
        # ====================================================

        elif h <= (
            PEAK_DAY
            + 7
        ):

            value = target

        # ====================================================
        # DESCENSO GRADUAL
        # ====================================================

        elif h <= 50:

            span = (
                50
                - (
                    PEAK_DAY
                    + 7
                )
            )

            fraction = (
                h
                - (
                    PEAK_DAY
                    + 7
                )
            ) / span

            value = (
                target
                + (
                    current
                    - target
                )
                * fraction
            )

        else:

            value = current

        values.append(
            max(
                float(
                    value
                ),
                0.0,
            )
        )

    return (
        np.array(
            values,
            dtype=float,
        ),
        stats,
    )


# ============================================================
# PERFIL AGUAS ARRIBA
# ============================================================

def upstream_value_for_day(
    current,
    target,
    h,
):

    if h <= PEAK_DAY:

        fraction = (
            h
            / PEAK_DAY
        )

        fraction = (
            3
            * fraction ** 2
            - 2
            * fraction ** 3
        )

        return float(
            current
            + (
                target
                - current
            )
            * fraction
        )

    if h <= (
        PEAK_DAY
        + 7
    ):

        return float(
            target
        )

    if h <= 50:

        fraction = (
            h
            - (
                PEAK_DAY
                + 7
            )
        ) / (
            50
            - (
                PEAK_DAY
                + 7
            )
        )

        return float(
            target
            + (
                current
                - target
            )
            * fraction
        )

    return float(
        current
    )


# ============================================================
# CREAR FEATURES
# ============================================================

def build_feature_row(
    history,
    feature_cols,
):

    featured = crear_features(
        history
    )

    latest = featured.iloc[
        -1
    ]

    row = {}

    for col in feature_cols:

        value = latest.get(
            col,
            np.nan,
        )

        if pd.isna(
            value
        ):

            value = 0.0

        row[
            col
        ] = float(
            value
        )

    return pd.DataFrame(
        [
            row
        ]
    )


# ============================================================
# ESCENARIO DE ESTRÉS
# ============================================================

def build_stress_scenario(
    models,
    exog_history=None,
    upstream_history=None,
    days=60,
    scenario="extremo",
):

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Escenario desconocido: {scenario}"
        )

    if (
        not isinstance(
            models,
            dict,
        )
        or "model"
        not in models
        or "dataset"
        not in models
    ):

        raise ValueError(
            "No existe un modelo entrenado válido."
        )

    days = max(
        1,
        min(
            int(
                days
            ),
            60,
        ),
    )

    model = models[
        "model"
    ]

    feature_cols = models[
        "feature_cols"
    ]

    rmse = float(
        models.get(
            "rmse",
            0.15,
        )
    )

    history = models[
        "dataset"
    ].copy()

    history[
        "datetime"
    ] = pd.to_datetime(
        history[
            "datetime"
        ],
        errors="coerce",
    ).dt.normalize()

    last_date = history[
        "datetime"
    ].max()

    nivel_actual = float(
        history[
            "nivel"
        ].iloc[-1]
    )

    # ========================================================
    # ESCENARIOS
    # ========================================================

    rain_values, rain_stats = (
        build_rain_scenario(
            exog_history,
            days,
            scenario,
        )
    )

    flow_values, flow_stats = (
        build_flow_scenario(
            exog_history,
            days,
            scenario,
        )
    )

    config = SCENARIOS[
        scenario
    ]

    upstream_stats = (
        upstream_statistics(
            upstream_history,
            config[
                "quantile"
            ],
        )
    )

    upstream_cols = [
        c
        for c in history.columns
        if c.startswith(
            "nivel_"
        )
    ]

    peak_future_date = (
        last_date
        + pd.Timedelta(
            days=PEAK_DAY
        )
    )

    output = []

    # ========================================================
    # SIMULACIÓN RECURSIVA 60 DÍAS
    # ========================================================

    for h in range(
        1,
        days + 1,
    ):

        target_date = (
            last_date
            + pd.Timedelta(
                days=h
            )
        )

        row = {

            "datetime": target_date,

            "nivel": float(
                history[
                    "nivel"
                ].iloc[-1]
            ),

            "precip_mm": float(
                rain_values[
                    h - 1
                ]
            ),

            "caudal_m3s": (
                float(
                    flow_values[
                        h - 1
                    ]
                )
                if np.isfinite(
                    flow_values[
                        h - 1
                    ]
                )
                else np.nan
            ),
        }

        # ====================================================
        # AGUAS ARRIBA
        # ====================================================

        for col in upstream_cols:

            info = upstream_stats.get(
                col
            )

            if info:

                row[
                    col
                ] = upstream_value_for_day(
                    info[
                        "current"
                    ],
                    info[
                        "target"
                    ],
                    h,
                )

            else:

                valid = (
                    history[
                        col
                    ]
                    .dropna()
                )

                row[
                    col
                ] = (
                    float(
                        valid.iloc[-1]
                    )
                    if len(
                        valid
                    )
                    else np.nan
                )

        # ====================================================
        # PREDICCIÓN
        # ====================================================

        trial = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        row
                    ]
                ),
            ],
            ignore_index=True,
        )

        X = build_feature_row(
            trial,
            feature_cols,
        )

        prediction = float(
            model.predict(
                X
            )[0]
        )

        prediction = float(
            np.clip(
                prediction,
                0.0,
                7.0,
            )
        )

        # ====================================================
        # INCERTIDUMBRE
        # ====================================================

        sigma = (
            rmse
            * np.sqrt(
                h
            )
            * config[
                "uncertainty_factor"
            ]
        )

        lower = max(
            0.0,
            prediction
            - 1.96
            * sigma,
        )

        upper = min(
            7.0,
            prediction
            + 1.96
            * sigma,
        )

        output.append(
            {
                "datetime": target_date,
                "prediction": prediction,
                "lower": lower,
                "upper": upper,
                "precip_mm": row[
                    "precip_mm"
                ],
                "caudal_m3s": row[
                    "caudal_m3s"
                ],
                "horizon_day": h,
                "scenario": config[
                    "label"
                ],
            }
        )

        row[
            "nivel"
        ] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        row
                    ]
                ),
            ],
            ignore_index=True,
        )

    scenario_df = pd.DataFrame(
        output
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    idx_max = scenario_df[
        "prediction"
    ].idxmax()

    max_level = float(
        scenario_df.loc[
            idx_max,
            "prediction",
        ]
    )

    max_level_date = (
        scenario_df.loc[
            idx_max,
            "datetime",
        ]
    )

    growth_m = (
        max_level
        - nivel_actual
    )

    growth_pct = (
        growth_m
        / nivel_actual
        * 100
        if nivel_actual
        != 0
        else np.nan
    )

    rain_peak = float(
        np.nanmax(
            rain_values
        )
    )

    rain_total = float(
        np.nansum(
            rain_values
        )
    )

    flow_peak = (
        float(
            np.nanmax(
                flow_values
            )
        )
        if np.isfinite(
            flow_values
        ).any()
        else np.nan
    )

    metadata = {

        "scenario": config[
            "label"
        ],

        "current_level": nivel_actual,

        "peak_future_date": (
            peak_future_date
        ),

        "max_level": max_level,

        "max_level_date": (
            max_level_date
        ),

        "growth_m": growth_m,

        "growth_pct": growth_pct,

        "rain_peak_scenario": (
            rain_peak
        ),

        "rain_event_total": (
            rain_total
        ),

        "flow_scenario_max": (
            flow_peak
        ),

        "rain_stats": rain_stats,

        "flow_stats": flow_stats,

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

    }

    return (
        scenario_df,
        metadata,
    )
