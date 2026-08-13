import numpy as np
import pandas as pd

from src.model import crear_features


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_STRESS_DAYS = 60


SCENARIOS = {
    "alto": {
        "label": "Alto",
        "quantile": 0.90,
        "rain_factor": 0.85,
        "uncertainty_factor": 1.20,
    },
    "severo": {
        "label": "Severo",
        "quantile": 0.95,
        "rain_factor": 1.00,
        "uncertainty_factor": 1.50,
    },
    "extremo": {
        "label": "Extremo histórico",
        "quantile": 1.00,
        "rain_factor": 1.15,
        "uncertainty_factor": 1.80,
    },
}


# ============================================================
# UTILIDADES
# ============================================================

def _numeric_series(df, column):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or column not in df.columns
    ):
        return pd.Series(dtype=float)

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .dropna()
        .astype(float)
    )


def _quantile(values, q):

    if values is None or len(values) == 0:
        return np.nan

    if q >= 1.0:
        return float(np.nanmax(values))

    return float(
        np.nanquantile(
            values,
            q,
        )
    )


# ============================================================
# RESUMEN HISTÓRICO DE LLUVIA
# ============================================================

def rainfall_statistics(exog_history):

    result = {
        "p90_day": 0.0,
        "p95_day": 0.0,
        "max_day": 0.0,
        "max_3d": 0.0,
        "max_7d": 0.0,
        "worst_7d_pattern": np.zeros(7),
    }

    rain = _numeric_series(
        exog_history,
        "precip_mm",
    )

    if rain.empty:
        return result

    values = rain.to_numpy(
        dtype=float
    )

    result["p90_day"] = float(
        np.nanquantile(
            values,
            0.90,
        )
    )

    result["p95_day"] = float(
        np.nanquantile(
            values,
            0.95,
        )
    )

    result["max_day"] = float(
        np.nanmax(values)
    )

    s = pd.Series(
        values
    )

    result["max_3d"] = float(
        s.rolling(
            3
        ).sum().max()
    )

    result["max_7d"] = float(
        s.rolling(
            7
        ).sum().max()
    )

    # --------------------------------------------------------
    # ENCONTRAR EL PEOR EPISODIO DE 7 DÍAS
    # --------------------------------------------------------

    if len(s) >= 7:

        rolling_7 = (
            s.rolling(
                7
            ).sum()
        )

        end_index = int(
            rolling_7.idxmax()
        )

        start_index = (
            end_index
            - 6
        )

        pattern = s.iloc[
            start_index:
            end_index + 1
        ].to_numpy(
            dtype=float
        )

        if len(pattern) == 7:

            result[
                "worst_7d_pattern"
            ] = pattern

    return result


# ============================================================
# RESUMEN HISTÓRICO DE CAUDAL
# ============================================================

def flow_statistics(exog_history):

    result = {
        "current": np.nan,
        "p90": np.nan,
        "p95": np.nan,
        "maximum": np.nan,
    }

    q = _numeric_series(
        exog_history,
        "caudal_m3s",
    )

    if q.empty:
        return result

    values = q.to_numpy(
        dtype=float
    )

    result["current"] = float(
        values[-1]
    )

    result["p90"] = float(
        np.nanquantile(
            values,
            0.90,
        )
    )

    result["p95"] = float(
        np.nanquantile(
            values,
            0.95,
        )
    )

    result["maximum"] = float(
        np.nanmax(
            values
        )
    )

    return result


# ============================================================
# ESTADÍSTICAS DE ESTACIONES AGUAS ARRIBA
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

    columns = [
        c
        for c in upstream_history.columns
        if c.startswith(
            "nivel_"
        )
    ]

    for col in columns:

        values = _numeric_series(
            upstream_history,
            col,
        )

        if values.empty:
            continue

        current = float(
            values.iloc[-1]
        )

        target = _quantile(
            values.to_numpy(
                dtype=float
            ),
            quantile,
        )

        result[col] = {
            "current": current,
            "target": target,
        }

    return result


# ============================================================
# CREAR ESCENARIO DE LLUVIA
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

    rain = np.zeros(
        days,
        dtype=float,
    )

    # --------------------------------------------------------
    # EL EVENTO SE COLOCA ENTRE LOS DÍAS 4 Y 10
    # PARA QUE COINCIDA CON EL ASCENSO DEL CAUDAL
    # --------------------------------------------------------

    start = 3

    for i, value in enumerate(
        pattern
    ):

        position = (
            start
            + i
        )

        if position < days:

            rain[
                position
            ] = max(
                float(value),
                0.0,
            )

    return (
        rain,
        stats,
    )


# ============================================================
# CREAR ESCENARIO DE CAUDAL
# ============================================================

def build_flow_scenario(
    exog_history,
    days,
    scenario,
):

    stats = flow_statistics(
        exog_history
    )

    if np.isnan(
        stats[
            "current"
        ]
    ):

        return (
            np.full(
                days,
                np.nan,
            ),
            stats,
        )

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

    if np.isnan(
        target
    ):

        target = current

    values = []

    for h in range(
        1,
        days + 1,
    ):

        # ====================================================
        # DÍAS 1–7:
        # CAUDAL SUBE HACIA EL VALOR HISTÓRICO ALTO
        # ====================================================

        if h <= 7:

            fraction = (
                h
                / 7.0
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
        # DÍAS 8–14:
        # PERMANECE EN ZONA ALTA
        # ====================================================

        elif h <= 14:

            value = target

        # ====================================================
        # DÍAS 15–40:
        # DESCENSO AMORTIGUADO
        # ====================================================

        elif h <= 40:

            fraction = (
                (
                    h
                    - 14
                )
                / 26.0
            )

            value = (
                target
                + (
                    current
                    - target
                )
                * fraction
            )

        # ====================================================
        # DÍAS 41–60:
        # REGRESO CERCA DEL CAUDAL DE REFERENCIA
        # ====================================================

        else:

            value = current

        values.append(
            max(
                float(value),
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
# PERFIL DE NIVEL AGUAS ARRIBA
# ============================================================

def upstream_value_for_day(
    current,
    target,
    h,
):

    # Subida hasta día 10
    if h <= 10:

        fraction = (
            h
            / 10.0
        )

        value = (
            current
            + (
                target
                - current
            )
            * fraction
        )

    # Zona alta días 11-18
    elif h <= 18:

        value = target

    # Retorno gradual días 19-45
    elif h <= 45:

        fraction = (
            (
                h
                - 18
            )
            / 27.0
        )

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

    return float(
        value
    )


# ============================================================
# CONSTRUIR X DEL RANDOM FOREST
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
# SIMULAR ESCENARIO
# ============================================================

def build_stress_scenario(
    models,
    exog_history=None,
    upstream_history=None,
    days=DEFAULT_STRESS_DAYS,
    scenario="severo",
):

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Escenario no válido: {scenario}"
        )

    if (
        models is None
        or not isinstance(
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

    days = int(
        max(
            1,
            min(
                days,
                60,
            ),
        )
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

    # ========================================================
    # CREAR FORZANTES
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

    upstream_stats = upstream_statistics(
        upstream_history,
        config[
            "quantile"
        ],
    )

    upstream_columns = [
        c
        for c in history.columns
        if c.startswith(
            "nivel_"
        )
    ]

    output = []

    # ========================================================
    # SIMULACIÓN RECURSIVA
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
                if not np.isnan(
                    flow_values[
                        h - 1
                    ]
                )
                else np.nan
            ),
        }

        # ====================================================
        # NIVELES AGUAS ARRIBA
        # ====================================================

        for col in upstream_columns:

            info = upstream_stats.get(
                col
            )

            if info is not None:

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
                    if len(valid)
                    else np.nan
                )

        # ====================================================
        # AGREGAR FILA TEMPORAL
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

        # Mantener dentro de la escala pública
        prediction = float(
            np.clip(
                prediction,
                0.0,
                7.0,
            )
        )

        # ====================================================
        # INCERTIDUMBRE DEL ESCENARIO
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
                    if not np.isnan(
                        flow_values[
                            h - 1
                        ]
                    )
                    else np.nan
                ),
                "horizon_day": h,
                "scenario": config[
                    "label"
                ],
            }
        )

        # ====================================================
        # ACTUALIZAR HISTÓRICO RECURSIVO
        # ====================================================

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
    # METADATOS
    # ========================================================

    metadata = {
        "scenario": config[
            "label"
        ],
        "rain_stats": rain_stats,
        "flow_stats": flow_stats,
        "rain_event_total": float(
            np.nansum(
                rain_values
            )
        ),
        "flow_scenario_max": (
            float(
                np.nanmax(
                    flow_values
                )
            )
            if np.isfinite(
                flow_values
            ).any()
            else np.nan
        ),
    }

    if not scenario_df.empty:

        idx_max = scenario_df[
            "prediction"
        ].idxmax()

        metadata[
            "max_level"
        ] = float(
            scenario_df.loc[
                idx_max,
                "prediction",
            ]
        )

        metadata[
            "max_level_date"
        ] = scenario_df.loc[
            idx_max,
            "datetime",
        ]

        if len(
            scenario_df
        ) >= 30:

            metadata[
                "level_day_30"
            ] = float(
                scenario_df[
                    "prediction"
                ].iloc[
                    29
                ]
            )

        if len(
            scenario_df
        ) >= 60:

            metadata[
                "level_day_60"
            ] = float(
                scenario_df[
                    "prediction"
                ].iloc[
                    59
                ]
            )

    return (
        scenario_df,
        metadata,
    )
