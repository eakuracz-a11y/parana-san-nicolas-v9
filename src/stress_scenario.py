import numpy as np
import pandas as pd

from src.historical_scenario import (
    build_daily_historical_envelope,
)

from src.flood_response import (
    fit_flood_response,
    predict_growth,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_STRESS_DAYS = 60

MAX_LEVEL = 7.0


SCENARIOS = {

    "alto": {
        "label":
            "Alto · P90 histórico",

        "rain_column":
            "rain_p90_mm",

        "flow_column":
            "flow_p90_m3s",

        "uncertainty":
            1.15,
    },

    "severo": {
        "label":
            "Severo · P95 histórico",

        "rain_column":
            "rain_p95_mm",

        "flow_column":
            "flow_p95_m3s",

        "uncertainty":
            1.40,
    },

    "extremo": {
        "label":
            "Extremo histórico por fecha",

        "rain_column":
            "rain_max_mm",

        "flow_column":
            "flow_max_m3s",

        "uncertainty":
            1.70,
    },
}


# ============================================================
# ESCENARIO
# ============================================================

def build_stress_scenario(
    models=None,
    exog_history=None,
    upstream_history=None,
    days=DEFAULT_STRESS_DAYS,
    scenario="extremo",
    current_level=None,
    current_date=None,
):

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Escenario desconocido: {scenario}"
        )

    days = int(
        np.clip(
            int(
                days
            ),
            1,
            60,
        )
    )

    config = SCENARIOS[
        scenario
    ]

    # ========================================================
    # FECHA BASE
    # ========================================================

    if current_date is None:

        if (
            models is not None
            and isinstance(
                models,
                dict,
            )
            and "dataset"
            in models
        ):

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

            dataset[
                "nivel"
            ] = pd.to_numeric(
                dataset[
                    "nivel"
                ],
                errors="coerce",
            )

            dataset = dataset.dropna(
                subset=[
                    "datetime",
                    "nivel",
                ]
            )

            if not dataset.empty:

                current_date = (
                    dataset[
                        "datetime"
                    ]
                    .max()
                    .normalize()
                )

                if current_level is None:

                    current_level = float(
                        dataset[
                            "nivel"
                        ].iloc[-1]
                    )

    if current_date is None:

        current_date = (
            pd.Timestamp.today()
            .normalize()
        )

    current_date = pd.Timestamp(
        current_date
    ).normalize()

    # ========================================================
    # TODO EL HISTÓRICO
    # ========================================================

    envelope, historical = (
        build_daily_historical_envelope(
            start_date=current_date,
            days=days,
        )
    )

    full_level = historical[
        "level"
    ]

    full_rain = historical[
        "rain"
    ]

    full_flow = historical[
        "flow"
    ]

    if full_level.empty:

        raise ValueError(
            "INA no devolvió histórico de nivel "
            "para San Nicolás."
        )

    # ========================================================
    # NIVEL BASE REAL
    # ========================================================

    full_level = full_level.copy()

    full_level[
        "datetime"
    ] = pd.to_datetime(
        full_level[
            "datetime"
        ],
        errors="coerce",
    )

    full_level[
        "nivel"
    ] = pd.to_numeric(
        full_level[
            "nivel"
        ],
        errors="coerce",
    )

    full_level = full_level.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if current_level is None:

        current_level = float(
            full_level[
                "nivel"
            ].iloc[-1]
        )

    current_level = float(
        current_level
    )

    # ========================================================
    # ENTRENAR CON TODO EL HISTÓRICO COINCIDENTE
    # ========================================================

    response_model = (
        fit_flood_response(
            level_history=full_level,
            rain_history=full_rain,
            flow_history=full_flow,
        )
    )

    # ========================================================
    # SELECCIONAR LA SERIE DEL ESCENARIO
    # ========================================================

    rain_column = config[
        "rain_column"
    ]

    flow_column = config[
        "flow_column"
    ]

    scenario_df = envelope.copy()

    scenario_df[
        "precip_mm"
    ] = pd.to_numeric(
        scenario_df[
            rain_column
        ],
        errors="coerce",
    )

    scenario_df[
        "caudal_m3s"
    ] = pd.to_numeric(
        scenario_df[
            flow_column
        ],
        errors="coerce",
    )

    # ========================================================
    # NUNCA LLUVIA CERO EN EL ESCENARIO SI EXISTE
    # HISTÓRICO POSITIVO
    # ========================================================

    historical_positive_rain = (
        pd.to_numeric(
            full_rain[
                "precip_mm"
            ],
            errors="coerce",
        )
        .dropna()
    )

    historical_positive_rain = (
        historical_positive_rain[
            historical_positive_rain
            > 0
        ]
    )

    if not historical_positive_rain.empty:

        global_rain_fallback = float(
            historical_positive_rain.max()
        )

    else:

        global_rain_fallback = 0.0

    scenario_df[
        "precip_mm"
    ] = (
        scenario_df[
            "precip_mm"
        ]
        .fillna(
            global_rain_fallback
        )
    )

    if (
        global_rain_fallback
        > 0
    ):

        scenario_df.loc[
            scenario_df[
                "precip_mm"
            ] <= 0,
            "precip_mm",
        ] = global_rain_fallback

    # ========================================================
    # CAUDAL SIN VACÍOS
    # ========================================================

    historical_flow_values = (
        pd.to_numeric(
            full_flow[
                "caudal_m3s"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if historical_flow_values.empty:

        raise ValueError(
            "No existe histórico de caudal "
            "para construir el escenario."
        )

    global_flow_fallback = float(
        historical_flow_values.max()
    )

    scenario_df[
        "caudal_m3s"
    ] = (
        scenario_df[
            "caudal_m3s"
        ]
        .fillna(
            global_flow_fallback
        )
    )

    scenario_df.loc[
        scenario_df[
            "caudal_m3s"
        ] <= 0,
        "caudal_m3s",
    ] = global_flow_fallback

    # ========================================================
    # REFERENCIA HISTÓRICA DE CAUDAL
    # ========================================================

    historical_flow_reference = float(
        historical_flow_values.median()
    )

    # ========================================================
    # ACUMULADOS DE LLUVIA DEL ESCENARIO
    # ========================================================

    scenario_df[
        "rain_3d"
    ] = (
        scenario_df[
            "precip_mm"
        ]
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    scenario_df[
        "rain_7d"
    ] = (
        scenario_df[
            "precip_mm"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    # ========================================================
    # CAMBIO DE CAUDAL 3 DÍAS
    # ========================================================

    scenario_df[
        "flow_change_3d"
    ] = (
        scenario_df[
            "caudal_m3s"
        ]
        - scenario_df[
            "caudal_m3s"
        ].shift(
            3
        )
    ).clip(
        lower=0
    )

    scenario_df[
        "flow_change_3d"
    ] = (
        scenario_df[
            "flow_change_3d"
        ]
        .fillna(
            0.0
        )
    )

    # ========================================================
    # CRECIMIENTO POTENCIAL PARA CADA FECHA
    # ========================================================

    potential_growth = []

    for _, row in (
        scenario_df.iterrows()
    ):

        growth = predict_growth(
            response_model=response_model,

            current_level=current_level,

            rain_1d=row[
                "precip_mm"
            ],

            rain_3d=row[
                "rain_3d"
            ],

            rain_7d=row[
                "rain_7d"
            ],

            flow_current=row[
                "caudal_m3s"
            ],

            flow_change_3d=row[
                "flow_change_3d"
            ],

            historical_flow_reference=(
                historical_flow_reference
            ),
        )

        potential_growth.append(
            growth
        )

    scenario_df[
        "potential_growth"
    ] = potential_growth

    # ========================================================
    # RETARDO HISTÓRICO
    # ========================================================

    response_lag = int(
        response_model[
            "response_lag"
        ]
    )

    # ========================================================
    # NIVEL FUTURO
    #
    # El nivel inicial es SIEMPRE el nivel actual.
    # Nunca puede comenzar debajo de 2,31 m,
    # por ejemplo.
    # ========================================================

    level_values = []

    previous_level = (
        current_level
    )

    for i in range(
        len(
            scenario_df
        )
    ):

        source_index = (
            i
            - response_lag
        )

        if source_index < 0:

            candidate_level = (
                current_level
            )

        else:

            candidate_growth = float(
                scenario_df[
                    "potential_growth"
                ].iloc[
                    source_index
                ]
            )

            candidate_level = (
                current_level
                + candidate_growth
            )

        # En un escenario denominado CRECIDA:
        # nunca retrocedemos debajo del nivel base
        # ni debajo del máximo alcanzado previamente.
        level = max(
            current_level,
            previous_level,
            candidate_level,
        )

        level = min(
            level,
            MAX_LEVEL,
        )

        level_values.append(
            float(
                level
            )
        )

        previous_level = (
            level
        )

    scenario_df[
        "prediction"
    ] = level_values

    # ========================================================
    # INCERTIDUMBRE
    # ========================================================

    rmse = max(
        float(
            response_model[
                "rmse"
            ]
        ),
        0.05,
    )

    lower = []

    upper = []

    for i, level in enumerate(
        scenario_df[
            "prediction"
        ],
        start=1,
    ):

        horizon_factor = (
            1.0
            + 0.02
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

        # Para representar el escenario,
        # la banda inferior tampoco se muestra
        # por debajo del nivel base real.
        lower.append(
            max(
                current_level,
                float(
                    level
                )
                - margin,
            )
        )

        upper.append(
            min(
                MAX_LEVEL,
                float(
                    level
                )
                + margin,
            )
        )

    scenario_df[
        "lower"
    ] = lower

    scenario_df[
        "upper"
    ] = upper

    scenario_df[
        "scenario"
    ] = config[
        "label"
    ]

    scenario_df[
        "horizon_day"
    ] = np.arange(
        1,
        len(
            scenario_df
        )
        + 1,
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    max_index = (
        scenario_df[
            "prediction"
        ].idxmax()
    )

    max_level = float(
        scenario_df.loc[
            max_index,
            "prediction",
        ]
    )

    max_level_date = (
        scenario_df.loc[
            max_index,
            "datetime",
        ]
    )

    growth = max(
        max_level
        - current_level,
        0.0,
    )

    growth_pct = (
        growth
        / current_level
        * 100
        if current_level
        != 0
        else np.nan
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
            max_level,

        "growth_m":
            growth,

        "growth_pct":
            growth_pct,

        "max_level_date":
            max_level_date,

        "response_lag_days":
            response_lag,

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

        "rain_max_60d":
            float(
                scenario_df[
                    "precip_mm"
                ].max()
            ),

        "rain_total_60d":
            float(
                scenario_df[
                    "precip_mm"
                ].sum()
            ),

        "flow_max_60d":
            float(
                scenario_df[
                    "caudal_m3s"
                ].max()
            ),

        "flood_model_rmse":
            response_model[
                "rmse"
            ],

        "flood_model_mae":
            response_model[
                "mae"
            ],

        "flood_training_rows":
            response_model[
                "training_rows"
            ],

        "historical_max_growth":
            response_model[
                "historical_max_growth"
            ],

        "historical_p95_growth":
            response_model[
                "historical_p95_growth"
            ],

        "rain_history_start":
            (
                full_rain[
                    "datetime"
                ].min()
                if not full_rain.empty
                else None
            ),

        "rain_history_end":
            (
                full_rain[
                    "datetime"
                ].max()
                if not full_rain.empty
                else None
            ),

        "level_history_start":
            (
                full_level[
                    "datetime"
                ].min()
                if not full_level.empty
                else None
            ),

        "level_history_end":
            (
                full_level[
                    "datetime"
                ].max()
                if not full_level.empty
                else None
            ),

        "flow_history_start":
            (
                full_flow[
                    "datetime"
                ].min()
                if not full_flow.empty
                else None
            ),

        "flow_history_end":
            (
                full_flow[
                    "datetime"
                ].max()
                if not full_flow.empty
                else None
            ),
    }

    return (
        scenario_df,
        metadata,
    )
