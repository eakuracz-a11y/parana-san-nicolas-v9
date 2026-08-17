import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRESS_DAYS = 60

LEVEL_MIN = 0.0
LEVEL_MAX = 7.0

MIN_HISTORY_DAYS = 60

EPS = 1e-9


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_fechas(serie):

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


def _numeric(serie):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _percentile_rank(serie):

    s = _numeric(
        serie
    )

    if s.notna().sum() < 2:

        return pd.Series(
            0.0,
            index=s.index,
            dtype=float,
        )

    return (
        s
        .rank(
            pct=True,
            method="average",
        )
        .fillna(
            0.0
        )
    )


def _upstream_cols(df):

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        return []

    return [
        c
        for c
        in df.columns
        if (
            c.startswith(
                "nivel_"
            )
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

def _preparar_nivel(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime" not in df.columns
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
    ] = _normalizar_fechas(
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
# PREPARAR VARIABLES EXÓGENAS
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
        or "datetime" not in exog_history.columns
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
    ] = _normalizar_fechas(
        x[
            "datetime"
        ]
    )

    if "precip_mm" not in x.columns:

        x[
            "precip_mm"
        ] = 0.0

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
        .fillna(
            0.0
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
        or "datetime" not in upstream_history.columns
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
    ] = _normalizar_fechas(
        x[
            "datetime"
        ]
    )

    cols = _upstream_cols(
        x
    )

    if not cols:

        return pd.DataFrame(
            {
                "datetime":
                    x[
                        "datetime"
                    ],

                "upstream_mean":
                    np.nan,

                "upstream_max":
                    np.nan,
            }
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
    ] = x[
        cols
    ].mean(
        axis=1,
        skipna=True,
    )

    x[
        "upstream_max"
    ] = x[
        cols
    ].max(
        axis=1,
        skipna=True,
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
# ARMAR HISTÓRICO INTEGRADO
# ============================================================

def _armar_historico(
    df,
    exog_history,
    upstream_history,
):

    nivel = _preparar_nivel(
        df
    )

    exog = _preparar_exog(
        exog_history
    )

    upstream = _preparar_upstream(
        upstream_history
    )

    if nivel.empty:

        return pd.DataFrame()

    hist = nivel.copy()

    if not exog.empty:

        hist = hist.merge(
            exog,
            on="datetime",
            how="outer",
        )

    if not upstream.empty:

        hist = hist.merge(
            upstream,
            on="datetime",
            how="outer",
        )

    if "precip_mm" not in hist.columns:

        hist[
            "precip_mm"
        ] = 0.0

    if "caudal_m3s" not in hist.columns:

        hist[
            "caudal_m3s"
        ] = np.nan

    if "upstream_mean" not in hist.columns:

        hist[
            "upstream_mean"
        ] = np.nan

    if "upstream_max" not in hist.columns:

        hist[
            "upstream_max"
        ] = np.nan

    hist[
        "precip_mm"
    ] = (
        _numeric(
            hist[
                "precip_mm"
            ]
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    for col in [
        "nivel",
        "caudal_m3s",
        "upstream_mean",
        "upstream_max",
    ]:

        hist[
            col
        ] = _numeric(
            hist[
                col
            ]
        )

    hist = (
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

    return hist


# ============================================================
# PUNTAJE DE SEVERIDAD HISTÓRICA
# ============================================================

def _calcular_severidad(
    hist,
):

    x = hist.copy()

    x[
        "rain_7d"
    ] = (
        x[
            "precip_mm"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .sum()
    )

    x[
        "rain_30d"
    ] = (
        x[
            "precip_mm"
        ]
        .rolling(
            30,
            min_periods=1,
        )
        .sum()
    )

    x[
        "q_7d"
    ] = (
        x[
            "caudal_m3s"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    x[
        "up_7d"
    ] = (
        x[
            "upstream_mean"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    x[
        "nivel_7d"
    ] = (
        x[
            "nivel"
        ]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    components = []
    weights = []

    for col, weight in [

        (
            "rain_7d",
            0.20,
        ),

        (
            "rain_30d",
            0.15,
        ),

        (
            "q_7d",
            0.25,
        ),

        (
            "up_7d",
            0.20,
        ),

        (
            "nivel_7d",
            0.20,
        ),
    ]:

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 5
        ):

            components.append(
                _percentile_rank(
                    x[
                        col
                    ]
                )
                * weight
            )

            weights.append(
                weight
            )

    if not components:

        x[
            "severity"
        ] = 0.0

        return x

    total_weight = sum(
        weights
    )

    severity = components[
        0
    ].copy()

    for component in components[
        1:
    ]:

        severity = (
            severity
            + component
        )

    x[
        "severity"
    ] = (
        severity
        / max(
            total_weight,
            EPS,
        )
    )

    return x


# ============================================================
# SELECCIONAR PEOR BLOQUE HISTÓRICO
# ============================================================

def _seleccionar_peor_bloque(
    hist,
    days=STRESS_DAYS,
):

    x = _calcular_severidad(
        hist
    )

    if x.empty:

        return pd.DataFrame()

    if len(
        x
    ) <= days:

        return x.copy()

    x[
        "window_score"
    ] = (
        x[
            "severity"
        ]
        .rolling(
            days,
            min_periods=max(
                30,
                int(
                    days
                    * 0.65
                ),
            ),
        )
        .mean()
    )

    if (
        x[
            "window_score"
        ]
        .notna()
        .sum()
        == 0
    ):

        return (
            x
            .tail(
                days
            )
            .copy()
        )

    end_idx = x[
        "window_score"
    ].idxmax()

    start_idx = max(
        0,
        int(
            end_idx
        )
        - days
        + 1,
    )

    block = (
        x
        .loc[
            start_idx:
            end_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        block
    ) > days:

        block = (
            block
            .tail(
                days
            )
            .reset_index(
                drop=True
            )
        )

    return block


# ============================================================
# MODELO EMPÍRICO DE RESPUESTA DIARIA
# ============================================================

def _ajustar_respuesta_empirica(
    hist,
):

    x = hist.copy()

    x[
        "delta_nivel"
    ] = x[
        "nivel"
    ].diff()

    x[
        "rain_3d"
    ] = (
        x[
            "precip_mm"
        ]
        .rolling(
            3,
            min_periods=1,
        )
        .sum()
    )

    x[
        "q_diff1"
    ] = x[
        "caudal_m3s"
    ].diff()

    x[
        "up_diff1"
    ] = x[
        "upstream_mean"
    ].diff()

    feature_names = []

    for col in [

        "rain_3d",
        "caudal_m3s",
        "q_diff1",
        "upstream_mean",
        "up_diff1",
    ]:

        if (
            x[
                col
            ]
            .notna()
            .sum()
            >= 20
        ):

            feature_names.append(
                col
            )

    if not feature_names:

        return None

    work = (
        x[
            [
                "delta_nivel",
            ]
            + feature_names
        ]
        .dropna()
    )

    if len(
        work
    ) < 30:

        return None

    means = work[
        feature_names
    ].mean()

    stds = work[
        feature_names
    ].std()

    stds = (
        stds
        .replace(
            0.0,
            1.0,
        )
        .fillna(
            1.0
        )
    )

    X = (
        (
            work[
                feature_names
            ]
            - means
        )
        / stds
    ).to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            np.ones(
                len(
                    X
                )
            ),
            X,
        ]
    )

    y = work[
        "delta_nivel"
    ].to_numpy(
        dtype=float
    )

    try:

        coef, _, _, _ = (
            np.linalg.lstsq(
                X,
                y,
                rcond=None,
            )
        )

    except Exception:

        return None

    hist_abs_delta = (
        x[
            "delta_nivel"
        ]
        .abs()
        .dropna()
    )

    if hist_abs_delta.empty:

        daily_limit = 0.20

    else:

        daily_limit = float(
            hist_abs_delta.quantile(
                0.99
            )
        )

    daily_limit = float(
        np.clip(
            daily_limit,
            0.08,
            0.50,
        )
    )

    return {

        "feature_names":
            feature_names,

        "means":
            means,

        "stds":
            stds,

        "coef":
            coef,

        "daily_limit":
            daily_limit,
    }


# ============================================================
# CREAR ESCENARIO DE 60 DÍAS
# ============================================================

def _crear_escenario_60_dias(
    hist,
    block,
):

    if (
        hist.empty
        or block.empty
        or "nivel"
        not in hist.columns
    ):

        return pd.DataFrame()

    niveles_validos = hist[
        "nivel"
    ].dropna()

    if niveles_validos.empty:

        return pd.DataFrame()

    nivel_actual = float(
        niveles_validos.iloc[
            -1
        ]
    )

    ultima_fecha = hist[
        "datetime"
    ].max()

    future_dates = pd.date_range(
        ultima_fecha
        + pd.Timedelta(
            days=1
        ),
        periods=STRESS_DAYS,
        freq="D",
    )

    response = (
        _ajustar_respuesta_empirica(
            hist
        )
    )

    block = block.copy()

    if len(
        block
    ) < STRESS_DAYS:

        repeats = int(
            np.ceil(
                STRESS_DAYS
                / max(
                    len(
                        block
                    ),
                    1,
                )
            )
        )

        block = pd.concat(
            [
                block
            ]
            * repeats,
            ignore_index=True,
        )

    block = (
        block
        .head(
            STRESS_DAYS
        )
        .copy()
    )

    block[
        "historical_delta"
    ] = block[
        "nivel"
    ].diff()

    q95_rain = (
        float(
            hist[
                "precip_mm"
            ].quantile(
                0.95
            )
        )
        if hist[
            "precip_mm"
        ].notna().any()
        else 0.0
    )

    q95_flow = (
        float(
            hist[
                "caudal_m3s"
            ].quantile(
                0.95
            )
        )
        if hist[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    q95_up = (
        float(
            hist[
                "upstream_mean"
            ].quantile(
                0.95
            )
        )
        if hist[
            "upstream_mean"
        ].notna().any()
        else np.nan
    )

    results = []

    nivel = nivel_actual

    prev_q = (
        float(
            block[
                "caudal_m3s"
            ]
            .dropna()
            .iloc[
                0
            ]
        )
        if block[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    prev_up = (
        float(
            block[
                "upstream_mean"
            ]
            .dropna()
            .iloc[
                0
            ]
        )
        if block[
            "upstream_mean"
        ].notna().any()
        else np.nan
    )

    rain_buffer = []

    for i in range(
        STRESS_DAYS
    ):

        row = block.iloc[
            i
        ]

        rain = pd.to_numeric(
            row.get(
                "precip_mm",
                np.nan,
            ),
            errors="coerce",
        )

        q = pd.to_numeric(
            row.get(
                "caudal_m3s",
                np.nan,
            ),
            errors="coerce",
        )

        up = pd.to_numeric(
            row.get(
                "upstream_mean",
                np.nan,
            ),
            errors="coerce",
        )

        # ====================================================
        # CONDICIONES SEVERAS
        #
        # Se conserva la secuencia histórica del peor bloque.
        # Los valores se acercan suavemente al percentil 95,
        # sin convertir todos los días en máximos absolutos.
        # ====================================================

        if pd.isna(
            rain
        ):

            rain = q95_rain

        if pd.notna(
            rain
        ):

            rain = float(
                max(
                    rain,
                    min(
                        q95_rain,
                        (
                            rain
                            * 1.15
                            if rain > 0
                            else 0.0
                        ),
                    ),
                )
            )

        if pd.isna(
            q
        ):

            q = q95_flow

        if (
            pd.notna(
                q95_flow
            )
            and pd.notna(
                q
            )
        ):

            q = float(
                max(
                    q,
                    min(
                        q95_flow,
                        q
                        * 1.03,
                    ),
                )
            )

        if pd.isna(
            up
        ):

            up = q95_up

        if (
            pd.notna(
                q95_up
            )
            and pd.notna(
                up
            )
        ):

            up = float(
                max(
                    up,
                    min(
                        q95_up,
                        up
                        * 1.02,
                    ),
                )
            )

        rain_buffer.append(
            float(
                rain
                if pd.notna(
                    rain
                )
                else 0.0
            )
        )

        rain_3d = float(
            sum(
                rain_buffer[
                    -3:
                ]
            )
        )

        empirical_delta = np.nan

        if response is not None:

            feature_values = {

                "rain_3d":
                    rain_3d,

                "caudal_m3s":
                    q,

                "q_diff1":
                    (
                        q
                        - prev_q
                        if (
                            pd.notna(
                                q
                            )
                            and pd.notna(
                                prev_q
                            )
                        )
                        else 0.0
                    ),

                "upstream_mean":
                    up,

                "up_diff1":
                    (
                        up
                        - prev_up
                        if (
                            pd.notna(
                                up
                            )
                            and pd.notna(
                                prev_up
                            )
                        )
                        else 0.0
                    ),
            }

            z = []

            for name in response[
                "feature_names"
            ]:

                value = (
                    feature_values
                    .get(
                        name,
                        np.nan,
                    )
                )

                mean = float(
                    response[
                        "means"
                    ][
                        name
                    ]
                )

                std = float(
                    response[
                        "stds"
                    ][
                        name
                    ]
                )

                if pd.isna(
                    value
                ):

                    value = mean

                z.append(
                    (
                        float(
                            value
                        )
                        - mean
                    )
                    / max(
                        std,
                        EPS,
                    )
                )

            vector = np.array(
                [
                    1.0
                ]
                + z,
                dtype=float,
            )

            empirical_delta = float(
                vector
                @ response[
                    "coef"
                ]
            )

        historical_delta = pd.to_numeric(
            row.get(
                "historical_delta",
                np.nan,
            ),
            errors="coerce",
        )

        if (
            pd.notna(
                empirical_delta
            )
            and pd.notna(
                historical_delta
            )
        ):

            delta = (
                0.65
                * empirical_delta
                + 0.35
                * float(
                    historical_delta
                )
            )

        elif pd.notna(
            empirical_delta
        ):

            delta = empirical_delta

        elif pd.notna(
            historical_delta
        ):

            delta = float(
                historical_delta
            )

        else:

            delta = 0.0

        if response is not None:

            daily_limit = response[
                "daily_limit"
            ]

        else:

            daily_limit = 0.25

        delta = float(
            np.clip(
                delta,
                -daily_limit,
                daily_limit,
            )
        )

        nivel_base = nivel

        nivel = float(
            np.clip(
                nivel
                + delta,
                LEVEL_MIN,
                LEVEL_MAX,
            )
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

                "precip_mm":
                    float(
                        rain
                        if pd.notna(
                            rain
                        )
                        else 0.0
                    ),

                "caudal_m3s":
                    (
                        float(
                            q
                        )
                        if pd.notna(
                            q
                        )
                        else np.nan
                    ),

                "upstream_mean":
                    (
                        float(
                            up
                        )
                        if pd.notna(
                            up
                        )
                        else np.nan
                    ),

                "historical_source_date":
                    row[
                        "datetime"
                    ],
            }
        )

        prev_q = q
        prev_up = up

    return pd.DataFrame(
        results
    )


# ============================================================
# MÁXIMO Y MÍNIMO HISTÓRICO POR DÍA DEL AÑO
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
    ] = nivel[
        "datetime"
    ].dt.month

    nivel[
        "day"
    ] = nivel[
        "datetime"
    ].dt.day

    env = (
        nivel
        .groupby(
            [
                "month",
                "day",
            ],
            as_index=False,
        )[
            "nivel"
        ]
        .agg(
            nivel_min_historico="min",
            nivel_max_historico="max",
            nivel_promedio_historico="mean",
            registros="count",
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
    ] = future[
        "datetime"
    ].dt.month

    future[
        "day"
    ] = future[
        "datetime"
    ].dt.day

    future = future.merge(
        env,
        on=[
            "month",
            "day",
        ],
        how="left",
    )

    return future


# ============================================================
# RENDER PRINCIPAL
# ============================================================

def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
):

    st.subheader(
        "⚠️ Escenario histórico severo · 60 días"
    )

    hist = _armar_historico(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
    )

    if hist.empty:

        st.info(
            "No hay información histórica suficiente "
            "para construir el escenario de 60 días."
        )

        return

    nivel_valid = hist[
        "nivel"
    ].dropna()

    if nivel_valid.empty:

        st.info(
            "No hay niveles válidos de San Nicolás "
            "para construir el escenario."
        )

        return

    block = _seleccionar_peor_bloque(
        hist,
        days=STRESS_DAYS,
    )

    if block.empty:

        st.info(
            "No fue posible identificar una secuencia "
            "histórica severa."
        )

        return

    scenario = _crear_escenario_60_dias(
        hist,
        block,
    )

    if scenario.empty:

        st.info(
            "No fue posible calcular el escenario "
            "histórico severo."
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

    nivel_actual = float(
        nivel_valid.iloc[
            -1
        ]
    )

    nivel_max_60 = float(
        scenario[
            "prediction"
        ].max()
    )

    nivel_final = float(
        scenario[
            "prediction"
        ].iloc[
            -1
        ]
    )

    cambio_max = (
        nivel_max_60
        - nivel_actual
    )

    lluvia_max_bloque = (
        float(
            block[
                "precip_mm"
            ].max()
        )
        if block[
            "precip_mm"
        ].notna().any()
        else 0.0
    )

    caudal_max_bloque = (
        float(
            block[
                "caudal_m3s"
            ].max()
        )
        if block[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    lluvia_max_hist = (
        float(
            hist[
                "precip_mm"
            ].max()
        )
        if hist[
            "precip_mm"
        ].notna().any()
        else 0.0
    )

    caudal_max_hist = (
        float(
            hist[
                "caudal_m3s"
            ].max()
        )
        if hist[
            "caudal_m3s"
        ].notna().any()
        else np.nan
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
    )

    c2.metric(
        "Máximo escenario 60 días",
        f"{nivel_max_60:.2f} m",
        f"{cambio_max:+.2f} m",
    )

    c3.metric(
        "Nivel al día 60",
        f"{nivel_final:.2f} m",
    )

    c4.metric(
        "Máx. lluvia bloque severo",
        f"{lluvia_max_bloque:.1f} mm/día",
    )

    # ========================================================
    # PERÍODO HISTÓRICO UTILIZADO
    # ========================================================

    block_start = block[
        "datetime"
    ].min()

    block_end = block[
        "datetime"
    ].max()

    st.caption(
        "La trayectoria se construye a partir de la secuencia "
        "histórica más severa disponible, conservando el orden "
        "temporal de lluvia, caudal y niveles aguas arriba. "
        f"Período histórico de referencia: "
        f"**{block_start.strftime('%d/%m/%Y')} – "
        f"{block_end.strftime('%d/%m/%Y')}**."
    )

    # ========================================================
    # GRÁFICO:
    # NIVEL ESTIMADO + MÁXIMO + MÍNIMO HISTÓRICO
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

    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "prediction"
            ],
            mode="lines+markers",
            name="Escenario severo 60 días",
            line=dict(
                color="#1f77b4",
                width=3,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel estimado: %{y:.2f} m"
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

    fig.add_hline(
        y=nivel_actual,
        line_dash="dash",
        annotation_text=(
            f"Nivel real de partida: "
            f"{nivel_actual:.2f} m"
        ),
    )

    fig.update_layout(
        height=520,
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
            LEVEL_MAX,
        ],
        dtick=0.5,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # CONDICIONES UTILIZADAS
    # ========================================================

    st.markdown(
        "**Condiciones utilizadas en el escenario severo**"
    )

    s1, s2, s3, s4 = st.columns(
        4
    )

    s1.metric(
        "Lluvia máx. histórica disponible",
        f"{lluvia_max_hist:.1f} mm/día",
    )

    s2.metric(
        "Lluvia máx. bloque seleccionado",
        f"{lluvia_max_bloque:.1f} mm/día",
    )

    s3.metric(
        "Caudal máx. histórico disponible",
        (
            f"{caudal_max_hist:,.0f} m³/s"
            if pd.notna(
                caudal_max_hist
            )
            else "Sin datos"
        ),
    )

    s4.metric(
        "Caudal máx. bloque seleccionado",
        (
            f"{caudal_max_bloque:,.0f} m³/s"
            if pd.notna(
                caudal_max_bloque
            )
            else "Sin datos"
        ),
    )

    # ========================================================
    # PRECIPITACIÓN DEL ESCENARIO
    # ========================================================

    rain_fig = go.Figure()

    rain_fig.add_trace(
        go.Bar(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "precip_mm"
            ],
            name="Lluvia escenario severo",
            marker_color="#17becf",
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Precipitación: %{y:.1f} mm"
                "<extra></extra>"
            ),
        )
    )

    rain_fig.update_layout(
        height=300,
        yaxis_title="Precipitación (mm/día)",
    )

    rain_fig.update_xaxes(
        tickformat="%d/%m",
    )

    st.plotly_chart(
        rain_fig,
        use_container_width=True,
    )

    # ========================================================
    # CAUDAL DEL ESCENARIO
    # ========================================================

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
                name="Caudal escenario severo",
                line=dict(
                    color="#9467bd",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Caudal: %{y:,.0f} m³/s"
                    "<extra></extra>"
                ),
            )
        )

        q_fig.update_layout(
            height=300,
            yaxis_title="Caudal (m³/s)",
        )

        q_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )

    # ========================================================
    # TABLA DE AUDITORÍA
    # ========================================================

    with st.expander(
        "🔎 Detalle diario del escenario de 60 días"
    ):

        tabla = scenario.copy()

        tabla[
            "Fecha"
        ] = tabla[
            "datetime"
        ].dt.strftime(
            "%d/%m/%Y"
        )

        tabla[
            "Nivel base (m)"
        ] = tabla[
            "nivel_base"
        ].round(
            2
        )

        tabla[
            "Variación diaria (m)"
        ] = tabla[
            "variacion_dia"
        ].round(
            3
        )

        tabla[
            "Nivel estimado (m)"
        ] = tabla[
            "prediction"
        ].round(
            2
        )

        tabla[
            "Lluvia (mm)"
        ] = tabla[
            "precip_mm"
        ].round(
            1
        )

        tabla[
            "Caudal (m³/s)"
        ] = tabla[
            "caudal_m3s"
        ].round(
            0
        )

        tabla[
            "Fecha histórica base"
        ] = pd.to_datetime(
            tabla[
                "historical_source_date"
            ],
            errors="coerce",
        ).dt.strftime(
            "%d/%m/%Y"
        )

        columnas = [

            "Fecha",
            "Nivel base (m)",
            "Lluvia (mm)",
            "Caudal (m³/s)",
            "Variación diaria (m)",
            "Nivel estimado (m)",
            "Fecha histórica base",
        ]

        if (
            "nivel_max_historico"
            in tabla.columns
        ):

            tabla[
                "Máx. histórico (m)"
            ] = tabla[
                "nivel_max_historico"
            ].round(
                2
            )

            columnas.append(
                "Máx. histórico (m)"
            )

        if (
            "nivel_min_historico"
            in tabla.columns
        ):

            tabla[
                "Mín. histórico (m)"
            ] = tabla[
                "nivel_min_historico"
            ].round(
                2
            )

            columnas.append(
                "Mín. histórico (m)"
            )

        st.dataframe(
            tabla[
                columnas
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # ADVERTENCIA
    # ========================================================

    if len(
        hist
    ) < MIN_HISTORY_DAYS:

        st.warning(
            "El historial disponible para este cálculo es corto. "
            "El escenario mejora cuando la aplicación consulta "
            "un historial más extenso."
        )

    st.caption(
        "Este bloque representa un escenario de estrés histórico, "
        "no un pronóstico meteorológico de 60 días. La secuencia "
        "se selecciona buscando la combinación histórica más severa "
        "disponible de lluvia, caudal, niveles aguas arriba y nivel "
        "de San Nicolás, preservando el orden temporal de los eventos."
    )
