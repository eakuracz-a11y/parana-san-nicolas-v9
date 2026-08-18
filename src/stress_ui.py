import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRESS_DAYS = 60

LEVEL_MIN = 0.0
LEVEL_MAX_INTERNAL = 12.0

MIN_HISTORY_DAYS = 90

EPS = 1e-9


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_datetime(
    serie,
):

    return (
        pd.to_datetime(
            serie,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
        .dt
        .normalize()
    )


def _numeric(
    serie,
):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


def _format_number(
    value,
    decimals=1,
):

    if value is None:

        return "--"

    try:

        if pd.isna(value):

            return "--"

    except Exception:

        pass

    text = (
        f"{float(value):,.{decimals}f}"
    )

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def _safe_last(
    serie,
    default=np.nan,
):

    s = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
    )

    if s.empty:

        return default

    return float(
        s.iloc[-1]
    )


def _safe_mean(
    serie,
    window=7,
    default=np.nan,
):

    s = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
        .tail(
            window
        )
    )

    if s.empty:

        return default

    return float(
        s.mean()
    )


def _upstream_cols(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return []

    return [
        c
        for c
        in df.columns
        if (
            c.startswith("nivel_")
            and c != "nivel"
            and "_lag" not in c
            and "_diff" not in c
            and "_trend" not in c
            and "_mean" not in c
            and "_actual" not in c
            and "_next" not in c
        )
    ]


# ============================================================
# PREPARAR NIVEL SAN NICOLÁS
# ============================================================

def _preparar_nivel(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x = df.copy()

    if "nivel" in x.columns:

        x[
            "nivel"
        ] = _numeric(
            x[
                "nivel"
            ]
        )

    elif "value" in x.columns:

        x[
            "nivel"
        ] = _numeric(
            x[
                "value"
            ]
        )

    else:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    x = (
        x
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )[
            "nivel"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# PREPARAR LLUVIA Y CAUDAL
# ============================================================

def _preparar_exog(
    exog_history,
):

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
        or "datetime"
        not in exog_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
        )

    x = exog_history.copy()

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    if "precip_mm" not in x.columns:

        x[
            "precip_mm"
        ] = np.nan

    if "caudal_m3s" not in x.columns:

        x[
            "caudal_m3s"
        ] = np.nan

    x[
        "precip_mm"
    ] = (
        _numeric(
            x[
                "precip_mm"
            ]
        )
        .clip(
            lower=0.0
        )
    )

    x[
        "caudal_m3s"
    ] = _numeric(
        x[
            "caudal_m3s"
        ]
    )

    return (
        x[
            [
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            {
                "precip_mm":
                    "mean",

                "caudal_m3s":
                    "mean",
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PREPARAR AGUAS ARRIBA
# ============================================================

def _preparar_upstream(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
        or "datetime"
        not in upstream_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        )

    x = upstream_history.copy()

    x[
        "datetime"
    ] = _normalizar_datetime(
        x[
            "datetime"
        ]
    )

    cols = _upstream_cols(
        x
    )

    if not cols:

        return pd.DataFrame(
            columns=[
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        )

    for col in cols:

        x[
            col
        ] = _numeric(
            x[
                col
            ]
        )

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 2
        ):

            x[
                col
            ] = (
                x[
                    col
                ]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    x[
        "upstream_mean"
    ] = (
        x[
            cols
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    x[
        "upstream_max"
    ] = (
        x[
            cols
        ]
        .max(
            axis=1,
            skipna=True,
        )
    )

    return (
        x[
            [
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            {
                "upstream_mean":
                    "mean",

                "upstream_max":
                    "max",
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HISTÓRICO INTEGRADO
# ============================================================

def _armar_historico(
    df,
    exog_history,
    upstream_history,
):

    nivel = _preparar_nivel(
        df
    )

    if nivel.empty:

        return pd.DataFrame()

    exog = _preparar_exog(
        exog_history
    )

    upstream = _preparar_upstream(
        upstream_history
    )

    hist = nivel.copy()

    if not exog.empty:

        hist = hist.merge(
            exog,
            on="datetime",
            how="left",
        )

    if not upstream.empty:

        hist = hist.merge(
            upstream,
            on="datetime",
            how="left",
        )

    for col in [
        "precip_mm",
        "caudal_m3s",
        "upstream_mean",
        "upstream_max",
    ]:

        if col not in hist.columns:

            hist[
                col
            ] = np.nan

        hist[
            col
        ] = _numeric(
            hist[
                col
            ]
        )

    # ========================================================
    # INTERPOLAR CAUDAL
    # ========================================================

    if (
        hist[
            "caudal_m3s"
        ]
        .notna()
        .sum()
        >= 7
    ):

        hist[
            "caudal_m3s"
        ] = (
            hist[
                "caudal_m3s"
            ]
            .interpolate(
                limit=5,
                limit_direction="both",
            )
        )

    # ========================================================
    # INTERPOLAR AGUAS ARRIBA
    # ========================================================

    for col in [
        "upstream_mean",
        "upstream_max",
    ]:

        if (
            hist[
                col
            ]
            .notna()
            .sum()
            >= 7
        ):

            hist[
                col
            ] = (
                hist[
                    col
                ]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    return (
        hist
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BUSCAR VENTANA DE MÁXIMA CRECIDA REAL
# ============================================================

def _buscar_maxima_crecida(
    hist,
    days=STRESS_DAYS,
):

    if (
        hist is None
        or hist.empty
        or len(hist) < days
    ):

        return pd.DataFrame()

    best = None

    for start_idx in range(
        0,
        len(hist) - days + 1,
    ):

        block = (
            hist
            .iloc[
                start_idx:
                start_idx + days
            ]
            .copy()
        )

        niveles = (
            block[
                "nivel"
            ]
            .dropna()
        )

        if len(
            niveles
        ) < max(
            30,
            int(
                days
                * 0.7
            ),
        ):

            continue

        nivel_inicio = float(
            niveles.iloc[
                0
            ]
        )

        nivel_max = float(
            niveles.max()
        )

        idx_local_max = (
            block[
                "nivel"
            ]
            .idxmax()
        )

        pos_max = (
            block.index
            .get_loc(
                idx_local_max
            )
        )

        crecimiento = (
            nivel_max
            - nivel_inicio
        )

        # ====================================================
        # VARIABLES DEL EVENTO
        # ====================================================

        lluvia_60 = float(
            block[
                "precip_mm"
            ]
            .fillna(0.0)
            .sum()
        )

        lluvia_15_max = float(
            block[
                "precip_mm"
            ]
            .fillna(0.0)
            .rolling(
                15,
                min_periods=1,
            )
            .sum()
            .max()
        )

        if (
            block[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            caudal_max = float(
                block[
                    "caudal_m3s"
                ]
                .max()
            )

            caudal_mean = float(
                block[
                    "caudal_m3s"
                ]
                .mean()
            )

        else:

            caudal_max = np.nan
            caudal_mean = np.nan

        if (
            block[
                "upstream_mean"
            ]
            .notna()
            .any()
        ):

            upstream_max = float(
                block[
                    "upstream_mean"
                ]
                .max()
            )

        else:

            upstream_max = np.nan

        # ====================================================
        # SCORE
        # ====================================================

        # La variable principal es la CRECIDA REAL.
        # Los drivers sólo sirven como desempate.

        score = (
            crecimiento
            * 100.0
        )

        score += (
            min(
                lluvia_60,
                1000.0,
            )
            * 0.002
        )

        score += (
            min(
                lluvia_15_max,
                500.0,
            )
            * 0.004
        )

        if pd.notna(
            caudal_max
        ):

            score += (
                caudal_max
                / 100000.0
            )

        if pd.notna(
            upstream_max
        ):

            score += (
                upstream_max
                * 0.02
            )

        candidate = {

            "score":
                score,

            "growth":
                crecimiento,

            "nivel_inicio":
                nivel_inicio,

            "nivel_max":
                nivel_max,

            "peak_position":
                pos_max,

            "rain_60":
                lluvia_60,

            "rain_15_max":
                lluvia_15_max,

            "q_max":
                caudal_max,

            "q_mean":
                caudal_mean,

            "up_max":
                upstream_max,

            "block":
                block,
        }

        if (
            best is None
            or candidate[
                "score"
            ]
            > best[
                "score"
            ]
        ):

            best = candidate

    if best is None:

        return pd.DataFrame()

    block = (
        best[
            "block"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    block.attrs[
        "growth"
    ] = best[
        "growth"
    ]

    block.attrs[
        "nivel_inicio"
    ] = best[
        "nivel_inicio"
    ]

    block.attrs[
        "nivel_max"
    ] = best[
        "nivel_max"
    ]

    block.attrs[
        "peak_position"
    ] = best[
        "peak_position"
    ]

    block.attrs[
        "rain_60"
    ] = best[
        "rain_60"
    ]

    block.attrs[
        "rain_15_max"
    ] = best[
        "rain_15_max"
    ]

    block.attrs[
        "q_max"
    ] = best[
        "q_max"
    ]

    block.attrs[
        "q_mean"
    ] = best[
        "q_mean"
    ]

    block.attrs[
        "up_max"
    ] = best[
        "up_max"
    ]

    return block


# ============================================================
# PERCENTILES HISTÓRICOS ALTOS
# ============================================================

def _historical_thresholds(
    hist,
):

    thresholds = {}

    # ========================================================
    # LLUVIA
    # ========================================================

    rain = (
        hist[
            "precip_mm"
        ]
        .dropna()
    )

    if not rain.empty:

        thresholds[
            "rain_p90"
        ] = float(
            rain.quantile(
                0.90
            )
        )

        thresholds[
            "rain_p95"
        ] = float(
            rain.quantile(
                0.95
            )
        )

        thresholds[
            "rain_p99"
        ] = float(
            rain.quantile(
                0.99
            )
        )

        thresholds[
            "rain_max"
        ] = float(
            rain.max()
        )

    else:

        thresholds[
            "rain_p90"
        ] = 0.0

        thresholds[
            "rain_p95"
        ] = 0.0

        thresholds[
            "rain_p99"
        ] = 0.0

        thresholds[
            "rain_max"
        ] = 0.0

    # ========================================================
    # CAUDAL
    # ========================================================

    q = (
        hist[
            "caudal_m3s"
        ]
        .dropna()
    )

    if not q.empty:

        thresholds[
            "q_p90"
        ] = float(
            q.quantile(
                0.90
            )
        )

        thresholds[
            "q_p95"
        ] = float(
            q.quantile(
                0.95
            )
        )

        thresholds[
            "q_p99"
        ] = float(
            q.quantile(
                0.99
            )
        )

        thresholds[
            "q_max"
        ] = float(
            q.max()
        )

    else:

        thresholds[
            "q_p90"
        ] = np.nan

        thresholds[
            "q_p95"
        ] = np.nan

        thresholds[
            "q_p99"
        ] = np.nan

        thresholds[
            "q_max"
        ] = np.nan

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    up = (
        hist[
            "upstream_mean"
        ]
        .dropna()
    )

    if not up.empty:

        thresholds[
            "up_p90"
        ] = float(
            up.quantile(
                0.90
            )
        )

        thresholds[
            "up_p95"
        ] = float(
            up.quantile(
                0.95
            )
        )

        thresholds[
            "up_p99"
        ] = float(
            up.quantile(
                0.99
            )
        )

        thresholds[
            "up_max"
        ] = float(
            up.max()
        )

    else:

        thresholds[
            "up_p90"
        ] = np.nan

        thresholds[
            "up_p95"
        ] = np.nan

        thresholds[
            "up_p99"
        ] = np.nan

        thresholds[
            "up_max"
        ] = np.nan

    return thresholds


# ============================================================
# LÍMITE HISTÓRICO DE VARIACIÓN DIARIA
# ============================================================

def _daily_change_limit(
    hist,
):

    delta = (
        hist[
            "nivel"
        ]
        .diff()
        .abs()
        .dropna()
    )

    if delta.empty:

        return 0.25

    p995 = float(
        delta.quantile(
            0.995
        )
    )

    return float(
        np.clip(
            p995,
            0.08,
            0.60,
        )
    )


# ============================================================
# CURVA HISTÓRICA DE CRECIDA NORMALIZADA
# ============================================================

def _historical_growth_shape(
    block,
):

    if block.empty:

        return np.zeros(
            STRESS_DAYS
        )

    niveles = (
        block[
            "nivel"
        ]
        .to_numpy(
            dtype=float
        )
    )

    initial = niveles[
        0
    ]

    growth = (
        niveles
        - initial
    )

    # ========================================================
    # IMPORTANTE
    #
    # Para un escenario de PEOR CRECIDA no usamos las
    # bajantes históricas posteriores como crecimiento negativo.
    #
    # Conservamos el máximo crecimiento alcanzado hasta cada día.
    # ========================================================

    growth = np.maximum.accumulate(
        growth
    )

    growth = np.maximum(
        growth,
        0.0,
    )

    if len(
        growth
    ) < STRESS_DAYS:

        growth = np.pad(
            growth,
            (
                0,
                STRESS_DAYS
                - len(
                    growth
                ),
            ),
            mode="edge",
        )

    return growth[
        :STRESS_DAYS
    ]


# ============================================================
# CREAR DRIVERS SEVEROS
# ============================================================

def _crear_drivers_severos(
    hist,
    block,
):

    thresholds = (
        _historical_thresholds(
            hist
        )
    )

    source = (
        block
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        source
    ) < STRESS_DAYS:

        repetitions = int(
            np.ceil(
                STRESS_DAYS
                / len(source)
            )
        )

        source = pd.concat(
            [
                source
            ]
            * repetitions,
            ignore_index=True,
        )

    source = (
        source
        .head(
            STRESS_DAYS
        )
        .copy()
    )

    output = []

    for i, row in source.iterrows():

        # ====================================================
        # LLUVIA
        # ====================================================

        rain_hist = _safe_float(
            row.get(
                "precip_mm",
                np.nan,
            )
        )

        if pd.isna(
            rain_hist
        ):

            rain = thresholds[
                "rain_p95"
            ]

        elif rain_hist > 0:

            # Llevar los días lluviosos del evento hacia
            # condiciones severas sin convertir todos en el máximo.

            rain = max(
                rain_hist,
                thresholds[
                    "rain_p90"
                ],
            )

            rain = min(
                rain,
                thresholds[
                    "rain_max"
                ],
            )

        else:

            rain = 0.0

        # ====================================================
        # CAUDAL
        # ====================================================

        q_hist = _safe_float(
            row.get(
                "caudal_m3s",
                np.nan,
            )
        )

        if pd.isna(
            q_hist
        ):

            q = thresholds[
                "q_p95"
            ]

        else:

            q = q_hist

            if pd.notna(
                thresholds[
                    "q_p90"
                ]
            ):

                q = max(
                    q,
                    thresholds[
                        "q_p90"
                    ],
                )

            if pd.notna(
                thresholds[
                    "q_max"
                ]
            ):

                q = min(
                    q,
                    thresholds[
                        "q_max"
                    ],
                )

        # ====================================================
        # AGUAS ARRIBA
        # ====================================================

        up_hist = _safe_float(
            row.get(
                "upstream_mean",
                np.nan,
            )
        )

        if pd.isna(
            up_hist
        ):

            up = thresholds[
                "up_p95"
            ]

        else:

            up = up_hist

            if pd.notna(
                thresholds[
                    "up_p90"
                ]
            ):

                up = max(
                    up,
                    thresholds[
                        "up_p90"
                    ],
                )

            if pd.notna(
                thresholds[
                    "up_max"
                ]
            ):

                up = min(
                    up,
                    thresholds[
                        "up_max"
                    ],
                )

        output.append(
            {
                "source_date":
                    row[
                        "datetime"
                    ],

                "precip_mm":
                    float(
                        rain
                        if pd.notna(rain)
                        else 0.0
                    ),

                "caudal_m3s":
                    (
                        float(q)
                        if pd.notna(q)
                        else np.nan
                    ),

                "upstream_mean":
                    (
                        float(up)
                        if pd.notna(up)
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# CALCULAR FUERZA DE LOS DRIVERS
# ============================================================

def _driver_strength(
    drivers,
    hist,
):

    thresholds = (
        _historical_thresholds(
            hist
        )
    )

    score = []

    rain_buffer = []

    for _, row in drivers.iterrows():

        rain = _safe_float(
            row[
                "precip_mm"
            ],
            0.0,
        )

        q = _safe_float(
            row[
                "caudal_m3s"
            ],
            np.nan,
        )

        up = _safe_float(
            row[
                "upstream_mean"
            ],
            np.nan,
        )

        rain_buffer.append(
            max(
                rain,
                0.0,
            )
        )

        rain7 = sum(
            rain_buffer[
                -7:
            ]
        )

        components = []

        # ====================================================
        # LLUVIA
        # ====================================================

        rain_reference = max(
            thresholds[
                "rain_p95"
            ]
            * 3.0,
            1.0,
        )

        components.append(
            min(
                rain7
                / rain_reference,
                2.0,
            )
        )

        # ====================================================
        # CAUDAL
        # ====================================================

        if (
            pd.notna(q)
            and pd.notna(
                thresholds[
                    "q_p95"
                ]
            )
            and thresholds[
                "q_p95"
            ] > 0
        ):

            components.append(
                min(
                    q
                    / thresholds[
                        "q_p95"
                    ],
                    1.5,
                )
            )

        # ====================================================
        # UPSTREAM
        # ====================================================

        if (
            pd.notna(up)
            and pd.notna(
                thresholds[
                    "up_p95"
                ]
            )
            and thresholds[
                "up_p95"
            ] > 0
        ):

            components.append(
                min(
                    up
                    / thresholds[
                        "up_p95"
                    ],
                    1.5,
                )
            )

        if components:

            value = float(
                np.mean(
                    components
                )
            )

        else:

            value = 1.0

        score.append(
            value
        )

    return np.array(
        score,
        dtype=float,
    )


# ============================================================
# CREAR ESCENARIO 60 DÍAS
# ============================================================

def _crear_escenario_60_dias(
    hist,
    block,
):

    if (
        hist.empty
        or block.empty
    ):

        return pd.DataFrame()

    nivel_actual = _safe_last(
        hist[
            "nivel"
        ]
    )

    if pd.isna(
        nivel_actual
    ):

        return pd.DataFrame()

    ultima_fecha = (
        hist[
            "datetime"
        ]
        .max()
    )

    future_dates = pd.date_range(
        ultima_fecha
        + pd.Timedelta(
            days=1
        ),
        periods=
            STRESS_DAYS,
        freq="D",
    )

    # ========================================================
    # FORMA REAL DE LA PEOR CRECIDA
    # ========================================================

    growth_shape = (
        _historical_growth_shape(
            block
        )
    )

    historical_growth_max = float(
        np.nanmax(
            growth_shape
        )
    )

    # ========================================================
    # DRIVERS EXTREMOS
    # ========================================================

    drivers = (
        _crear_drivers_severos(
            hist,
            block,
        )
    )

    strengths = (
        _driver_strength(
            drivers,
            hist,
        )
    )

    # ========================================================
    # FACTOR DE SEVERIDAD
    # ========================================================

    strength_smoothed = (
        pd.Series(
            strengths
        )
        .rolling(
            5,
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    # Condición severa:
    # factor mínimo 0.85 y máximo 1.30.

    severity_factor = np.clip(
        strength_smoothed,
        0.85,
        1.30,
    )

    # ========================================================
    # CRECIMIENTO OBJETIVO
    # ========================================================

    target_growth = (
        growth_shape
        * severity_factor
    )

    # ========================================================
    # GARANTIZAR QUE LA PEOR CONDICIÓN NO SEA UNA BAJANTE
    # ========================================================

    target_growth = np.maximum.accumulate(
        target_growth
    )

    target_growth = np.maximum(
        target_growth,
        0.0,
    )

    # ========================================================
    # EVITAR SOBREPASAR UNA CRECIDA IMPOSIBLE
    # ========================================================

    max_hist_growth = (
        hist[
            "nivel"
        ]
        .rolling(
            STRESS_DAYS,
            min_periods=
                STRESS_DAYS,
        )
        .max()
        - hist[
            "nivel"
        ]
        .rolling(
            STRESS_DAYS,
            min_periods=
                STRESS_DAYS,
        )
        .min()
    )

    max_hist_growth = (
        max_hist_growth
        .dropna()
    )

    if not max_hist_growth.empty:

        historical_growth_cap = float(
            max_hist_growth.quantile(
                0.995
            )
        )

    else:

        historical_growth_cap = (
            historical_growth_max
        )

    historical_growth_cap = max(
        historical_growth_cap,
        historical_growth_max,
    )

    historical_growth_cap *= 1.10

    target_growth = np.minimum(
        target_growth,
        historical_growth_cap,
    )

    # ========================================================
    # CONSTRUIR NIVEL DÍA A DÍA
    # ========================================================

    daily_limit = (
        _daily_change_limit(
            hist
        )
    )

    results = []

    nivel = float(
        nivel_actual
    )

    growth_prev = 0.0

    for i in range(
        STRESS_DAYS
    ):

        desired_growth = float(
            target_growth[
                i
            ]
        )

        desired_delta = (
            desired_growth
            - growth_prev
        )

        desired_delta = float(
            np.clip(
                desired_delta,
                0.0,
                daily_limit,
            )
        )

        # ====================================================
        # IMPORTANTE
        #
        # Este escenario representa deliberadamente una
        # PEOR CONDICIÓN DE CRECIDA.
        #
        # Por eso no permitimos variación negativa mientras
        # los drivers son severos.
        # ====================================================

        delta = desired_delta

        nivel_base = nivel

        nivel = float(
            np.clip(
                nivel
                + delta,
                LEVEL_MIN,
                LEVEL_MAX_INTERNAL,
            )
        )

        growth_prev = (
            nivel
            - nivel_actual
        )

        results.append(
            {
                "datetime":
                    future_dates[
                        i
                    ],

                "prediction":
                    nivel,

                "nivel_base":
                    nivel_base,

                "variacion_dia":
                    delta,

                "crecimiento_acumulado":
                    growth_prev,

                "precip_mm":
                    float(
                        drivers.loc[
                            i,
                            "precip_mm"
                        ]
                    ),

                "caudal_m3s":
                    drivers.loc[
                        i,
                        "caudal_m3s"
                    ],

                "upstream_mean":
                    drivers.loc[
                        i,
                        "upstream_mean"
                    ],

                "driver_strength":
                    float(
                        strengths[
                            i
                        ]
                    ),

                "source_date":
                    drivers.loc[
                        i,
                        "source_date"
                    ],
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# ENVOLVENTE HISTÓRICA POR FECHA
# ============================================================

def _envolvente_historica_diaria(
    df,
    future_dates,
):

    nivel = _preparar_nivel(
        df
    )

    if nivel.empty:

        return pd.DataFrame()

    nivel[
        "month"
    ] = (
        nivel[
            "datetime"
        ]
        .dt
        .month
    )

    nivel[
        "day"
    ] = (
        nivel[
            "datetime"
        ]
        .dt
        .day
    )

    env = (
        nivel
        .groupby(
            [
                "month",
                "day",
            ],
            as_index=False,
        )
        .agg(

            nivel_min_historico=(
                "nivel",
                "min",
            ),

            nivel_max_historico=(
                "nivel",
                "max",
            ),

            nivel_promedio_historico=(
                "nivel",
                "mean",
            ),

            registros=(
                "nivel",
                "count",
            ),
        )
    )

    future = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates,
                    errors="coerce",
                )
        }
    )

    future[
        "month"
    ] = (
        future[
            "datetime"
        ]
        .dt
        .month
    )

    future[
        "day"
    ] = (
        future[
            "datetime"
        ]
        .dt
        .day
    )

    return future.merge(
        env,
        on=[
            "month",
            "day",
        ],
        how="left",
    )


# ============================================================
# RENDER
# ============================================================

def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
):

    st.subheader(
        "⚠️ Peor escenario histórico de creciente · 60 días"
    )

    st.caption(
        "Simulación experimental que parte del nivel real actual "
        "y aplica la forma de la mayor creciente histórica de 60 días, "
        "reforzada con lluvia, caudal y niveles aguas arriba elevados."
    )

    # ========================================================
    # HISTÓRICO
    # ========================================================

    hist = _armar_historico(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
    )

    if (
        hist.empty
        or len(hist)
        < MIN_HISTORY_DAYS
    ):

        st.info(
            "No hay suficiente historial para calcular "
            "el escenario severo de 60 días."
        )

        return

    # ========================================================
    # BUSCAR MAYOR CRECIDA
    # ========================================================

    block = _buscar_maxima_crecida(
        hist,
        days=STRESS_DAYS,
    )

    if block.empty:

        st.info(
            "No fue posible identificar una creciente histórica "
            "continua de 60 días."
        )

        return

    # ========================================================
    # ESCENARIO
    # ========================================================

    scenario = (
        _crear_escenario_60_dias(
            hist,
            block,
        )
    )

    if scenario.empty:

        st.info(
            "No fue posible calcular el escenario de 60 días."
        )

        return

    envelope = (
        _envolvente_historica_diaria(
            df,
            scenario[
                "datetime"
            ],
        )
    )

    if not envelope.empty:

        scenario = scenario.merge(
            envelope[
                [
                    "datetime",
                    "nivel_min_historico",
                    "nivel_max_historico",
                    "nivel_promedio_historico",
                    "registros",
                ]
            ],
            on="datetime",
            how="left",
        )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    nivel_actual = float(
        hist[
            "nivel"
        ]
        .dropna()
        .iloc[
            -1
        ]
    )

    idx_max = (
        scenario[
            "prediction"
        ]
        .idxmax()
    )

    nivel_max_escenario = float(
        scenario.loc[
            idx_max,
            "prediction"
        ]
    )

    fecha_max_escenario = pd.to_datetime(
        scenario.loc[
            idx_max,
            "datetime"
        ]
    )

    crecimiento_maximo = (
        nivel_max_escenario
        - nivel_actual
    )

    nivel_final = float(
        scenario[
            "prediction"
        ]
        .iloc[
            -1
        ]
    )

    max_hist_periodo = np.nan

    if (
        "nivel_max_historico"
        in scenario.columns
        and scenario[
            "nivel_max_historico"
        ]
        .notna()
        .any()
    ):

        max_hist_periodo = float(
            scenario[
                "nivel_max_historico"
            ]
            .max()
        )

    distancia_max = np.nan

    if pd.notna(
        max_hist_periodo
    ):

        distancia_max = (
            max_hist_periodo
            - nivel_max_escenario
        )

    # ========================================================
    # EVENTO HISTÓRICO ORIGEN
    # ========================================================

    block_start = pd.to_datetime(
        block[
            "datetime"
        ]
        .iloc[
            0
        ]
    )

    block_end = pd.to_datetime(
        block[
            "datetime"
        ]
        .iloc[
            -1
        ]
    )

    historical_growth = float(
        block.attrs.get(
            "growth",
            np.nan,
        )
    )

    historical_start = float(
        block.attrs.get(
            "nivel_inicio",
            np.nan,
        )
    )

    historical_max = float(
        block.attrs.get(
            "nivel_max",
            np.nan,
        )
    )

    # ========================================================
    # FILA PRINCIPAL
    # ========================================================

    st.markdown(
        "**Resultado del escenario**"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel real inicial",
        f"{nivel_actual:.2f} m",
    )

    c2.metric(
        "Máximo escenario",
        f"{nivel_max_escenario:.2f} m",
        f"{crecimiento_maximo:+.2f} m",
    )

    c3.metric(
        "Máximo histórico período",
        (
            f"{max_hist_periodo:.2f} m"
            if pd.notna(
                max_hist_periodo
            )
            else "--"
        ),
    )

    c4.metric(
        "Distancia al máximo histórico",
        (
            f"{distancia_max:.2f} m"
            if pd.notna(
                distancia_max
            )
            else "--"
        ),
    )

    # ========================================================
    # SEGUNDA FILA
    # ========================================================

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Nivel día 60",
        f"{nivel_final:.2f} m",
        f"{nivel_final - nivel_actual:+.2f} m",
    )

    d2.metric(
        "Crecimiento máximo",
        f"{crecimiento_maximo:+.2f} m",
    )

    d3.metric(
        "Fecha del máximo",
        fecha_max_escenario.strftime(
            "%d/%m/%Y"
        ),
    )

    d4.metric(
        "Crecida histórica usada",
        (
            f"+{historical_growth:.2f} m"
            if pd.notna(
                historical_growth
            )
            else "--"
        ),
    )

    # ========================================================
    # DRIVERS
    # ========================================================

    lluvia_acum = float(
        scenario[
            "precip_mm"
        ]
        .fillna(0.0)
        .sum()
    )

    lluvia_max = float(
        scenario[
            "precip_mm"
        ]
        .fillna(0.0)
        .max()
    )

    caudal_max = (
        float(
            scenario[
                "caudal_m3s"
            ]
            .max()
        )
        if scenario[
            "caudal_m3s"
        ]
        .notna()
        .any()
        else np.nan
    )

    upstream_max = (
        float(
            scenario[
                "upstream_mean"
            ]
            .max()
        )
        if scenario[
            "upstream_mean"
        ]
        .notna()
        .any()
        else np.nan
    )

    st.markdown(
        "**Variables severas utilizadas**"
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Lluvia acumulada 60 d",
        f"{_format_number(lluvia_acum, 1)} mm",
    )

    v2.metric(
        "Lluvia máxima diaria",
        f"{_format_number(lluvia_max, 1)} mm",
    )

    v3.metric(
        "Caudal máximo",
        (
            f"{_format_number(caudal_max, 0)} m³/s"
            if pd.notna(
                caudal_max
            )
            else "--"
        ),
    )

    v4.metric(
        "Aguas arriba máximo medio",
        (
            f"{upstream_max:.2f} m"
            if pd.notna(
                upstream_max
            )
            else "--"
        ),
    )

    # ========================================================
    # INFORMACIÓN DEL EVENTO HISTÓRICO
    # ========================================================

    st.caption(
        "Evento histórico utilizado como patrón: "
        f"**{block_start.strftime('%d/%m/%Y')} → "
        f"{block_end.strftime('%d/%m/%Y')}** · "
        f"nivel inicial histórico "
        f"**{historical_start:.2f} m** · "
        f"máximo histórico del evento "
        f"**{historical_max:.2f} m**."
    )

    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    fig = go.Figure()

    if (
        "nivel_max_historico"
        in scenario.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "nivel_max_historico"
                ],
                mode="lines",
                name="Máximo histórico del día",
                line=dict(
                    color="#d62728",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Máximo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    if (
        "nivel_min_historico"
        in scenario.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "nivel_min_historico"
                ],
                mode="lines",
                name="Mínimo histórico del día",
                line=dict(
                    color="#2ca02c",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Mínimo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "prediction"
            ],
            mode="lines+markers",
            name="Peor escenario de creciente",
            line=dict(
                color="#1f77b4",
                width=4,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel escenario: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        annotation_text=(
            f"Nivel real de partida: "
            f"{nivel_actual:.2f} m"
        ),
    )

    fig.add_vline(
        x=fecha_max_escenario,
        line_dash="dot",
        line_width=1,
    )

    fig.add_annotation(
        x=fecha_max_escenario,
        y=nivel_max_escenario,
        text=(
            f"Máximo escenario<br>"
            f"{nivel_max_escenario:.2f} m"
        ),
        showarrow=True,
        arrowhead=2,
        yshift=25,
    )

    valores_max = [
        nivel_actual,
        nivel_max_escenario,
    ]

    if pd.notna(
        max_hist_periodo
    ):

        valores_max.append(
            max_hist_periodo
        )

    y_top = min(
        LEVEL_MAX_INTERNAL,
        max(
            7.0,
            max(
                valores_max
            )
            + 0.5,
        ),
    )

    fig.update_layout(
        height=530,
        margin=dict(
            l=10,
            r=10,
            t=35,
            b=10,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m/%Y",
    )

    fig.update_yaxes(
        title_text="Nivel hidrométrico (m)",
        range=[
            LEVEL_MIN,
            y_top,
        ],
        dtick=0.5,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # GRÁFICO DE CRECIMIENTO
    # ========================================================

    growth_fig = go.Figure()

    growth_fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "crecimiento_acumulado"
            ],
            mode="lines+markers",
            name="Crecimiento desde nivel actual",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=4,
            ),
        )
    )

    growth_fig.update_layout(
        height=280,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=5,
        ),
        yaxis_title="Crecimiento acumulado (m)",
        showlegend=False,
    )

    growth_fig.update_xaxes(
        tickformat="%d/%m",
    )

    st.plotly_chart(
        growth_fig,
        use_container_width=True,
    )

    # ========================================================
    # LLUVIA + CAUDAL
    # ========================================================

    rain_col, q_col = st.columns(
        2
    )

    with rain_col:

        st.markdown(
            "**🌧️ Lluvia severa utilizada**"
        )

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "precip_mm"
                ],
                name="Lluvia",
            )
        )

        rain_fig.update_layout(
            height=270,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),
            yaxis_title="mm/día",
            showlegend=False,
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    with q_col:

        st.markdown(
            "**💧 Caudal severo utilizado**"
        )

        if (
            scenario[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            q_fig = go.Figure()

            q_fig.add_trace(
                go.Scatter(
                    x=scenario[
                        "datetime"
                    ],
                    y=scenario[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    name="Caudal",
                )
            )

            q_fig.update_layout(
                height=270,
                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),
                yaxis_title="m³/s",
                showlegend=False,
            )

            q_fig.update_xaxes(
                tickformat="%d/%m",
            )

            st.plotly_chart(
                q_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "Sin caudal histórico suficiente."
            )

    # ========================================================
    # TABLA AUDITORÍA
    # ========================================================

    with st.expander(
        "🔎 Auditoría diaria del peor escenario"
    ):

        tabla = scenario.copy()

        tabla[
            "Fecha"
        ] = (
            pd.to_datetime(
                tabla[
                    "datetime"
                ]
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        tabla[
            "Nivel base"
        ] = (
            tabla[
                "nivel_base"
            ]
            .round(2)
        )

        tabla[
            "Δ día"
        ] = (
            tabla[
                "variacion_dia"
            ]
            .round(3)
        )

        tabla[
            "Crecimiento"
        ] = (
            tabla[
                "crecimiento_acumulado"
            ]
            .round(2)
        )

        tabla[
            "Nivel escenario"
        ] = (
            tabla[
                "prediction"
            ]
            .round(2)
        )

        tabla[
            "Lluvia"
        ] = (
            tabla[
                "precip_mm"
            ]
            .round(1)
        )

        tabla[
            "Caudal"
        ] = (
            tabla[
                "caudal_m3s"
            ]
            .round(0)
        )

        tabla[
            "Aguas arriba"
        ] = (
            tabla[
                "upstream_mean"
            ]
            .round(2)
        )

        tabla[
            "Fecha origen histórica"
        ] = (
            pd.to_datetime(
                tabla[
                    "source_date"
                ],
                errors="coerce",
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        columnas = [
            "Fecha",
            "Nivel base",
            "Lluvia",
            "Caudal",
            "Aguas arriba",
            "Δ día",
            "Crecimiento",
            "Nivel escenario",
            "Fecha origen histórica",
        ]

        if (
            "nivel_max_historico"
            in tabla.columns
        ):

            tabla[
                "Máximo histórico"
            ] = (
                tabla[
                    "nivel_max_historico"
                ]
                .round(2)
            )

            columnas.append(
                "Máximo histórico"
            )

        st.dataframe(
            tabla[
                columnas
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # EXPLICACIÓN
    # ========================================================

    with st.expander(
        "ℹ️ Cómo se construye este escenario"
    ):

        st.markdown(
            f"""
            Este escenario **no intenta predecir el tiempo de los
            próximos 60 días**.

            Su objetivo es responder:

            **¿Qué crecimiento podría producirse si, partiendo del
            nivel real actual de {nivel_actual:.2f} m, se repitiera
            una de las peores crecientes históricas observadas?**

            El sistema busca todas las ventanas históricas continuas
            de **60 días** y selecciona principalmente la que produjo
            la **mayor suba real de nivel en San Nicolás**.

            El evento seleccionado fue:

            **{block_start.strftime('%d/%m/%Y')} →
            {block_end.strftime('%d/%m/%Y')}**

            En esa ventana la altura pasó aproximadamente de
            **{historical_start:.2f} m** hasta
            **{historical_max:.2f} m**, una creciente de
            **{historical_growth:+.2f} m**.

            La forma de esa creciente se aplica desde el nivel actual,
            reforzándola cuando lluvia, caudal y niveles aguas arriba
            están dentro de sus rangos históricos altos.

            En este módulo de **peor escenario de creciente** no se
            permiten bajantes importantes durante el evento severo,
            porque el objetivo no es representar el escenario más
            probable sino una condición histórica desfavorable.
            """
        )

    st.warning(
        "Escenario experimental de estrés histórico. "
        "No constituye una alerta ni reemplaza pronósticos "
        "o comunicaciones oficiales."
    )
