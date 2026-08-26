import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/stress_ui.py
# BASE V11.0
# ESCENARIO HIPOTÉTICO 60 DÍAS
# ============================================================

STRESS_DAYS = 60

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# UTILIDADES
# ============================================================

def _to_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _get_level_column(df):

    if (
        isinstance(df, pd.DataFrame)
        and "nivel" in df.columns
    ):
        return "nivel"

    if (
        isinstance(df, pd.DataFrame)
        and "value" in df.columns
    ):
        return "value"

    return None


def _safe_quantile(
    series,
    q,
):

    values = _to_numeric(
        series
    ).dropna()

    if values.empty:
        return None

    return float(
        values.quantile(q)
    )


# ============================================================
# SEÑAL DE ESTACIONES AGUAS ARRIBA
# ============================================================

def _upstream_signal(
    upstream_history,
):

    result = {
        "mean_delta_3":
            0.0,

        "mean_delta_7":
            0.0,

        "available":
            0,
    }

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
        col
        for col
        in upstream_history.columns
        if col.startswith(
            "nivel_"
        )
    ]

    deltas_3 = []
    deltas_7 = []

    for col in level_cols:

        values = (
            _to_numeric(
                upstream_history[
                    col
                ]
            )
            .dropna()
        )

        if values.empty:
            continue

        result[
            "available"
        ] += 1

        if len(values) >= 4:

            deltas_3.append(
                float(
                    values.iloc[-1]
                    - values.iloc[-4]
                )
            )

        if len(values) >= 8:

            deltas_7.append(
                float(
                    values.iloc[-1]
                    - values.iloc[-8]
                )
            )

    if deltas_3:

        result[
            "mean_delta_3"
        ] = float(
            np.mean(
                deltas_3
            )
        )

    if deltas_7:

        result[
            "mean_delta_7"
        ] = float(
            np.mean(
                deltas_7
            )
        )

    return result


# ============================================================
# NIVEL HISTÓRICO PARA MISMA FECHA
# ============================================================

def _historical_level_envelope(
    df,
    future_dates,
):

    level_col = _get_level_column(
        df
    )

    if (
        level_col is None
        or "datetime" not in df.columns
    ):

        return pd.DataFrame()

    hist = df.copy()

    hist[
        "datetime"
    ] = pd.to_datetime(
        hist[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    hist[
        level_col
    ] = _to_numeric(
        hist[
            level_col
        ]
    )

    hist = hist.dropna(
        subset=[
            "datetime",
            level_col,
        ]
    )

    if hist.empty:
        return pd.DataFrame()

    hist[
        "month_day"
    ] = (
        hist[
            "datetime"
        ]
        .dt.strftime(
            "%m-%d"
        )
    )

    daily = (
        hist
        .groupby(
            "month_day"
        )[
            level_col
        ]
        .agg(
            level_min_historical=
                "min",

            level_max_historical=
                "max",

            level_mean_historical=
                "mean",
        )
        .reset_index()
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates,
                    utc=True,
                )
        }
    )

    result[
        "month_day"
    ] = (
        result[
            "datetime"
        ]
        .dt.strftime(
            "%m-%d"
        )
    )

    result = result.merge(
        daily,
        on="month_day",
        how="left",
    )

    return result


# ============================================================
# LLUVIA HISTÓRICA ALTA
# ============================================================

def _rain_stress(
    exog_history,
):

    result = {
        "daily_high":
            0.0,

        "max":
            0.0,

        "p95":
            0.0,
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

    rain = (
        _to_numeric(
            exog_history[
                "precip_mm"
            ]
        )
        .dropna()
    )

    if rain.empty:
        return result

    rain = rain.clip(
        lower=0.0
    )

    result[
        "max"
    ] = float(
        rain.max()
    )

    result[
        "p95"
    ] = float(
        rain.quantile(
            0.95
        )
    )

    result[
        "daily_high"
    ] = max(
        result[
            "p95"
        ],
        0.0,
    )

    return result


# ============================================================
# CAUDAL HISTÓRICO ALTO
# ============================================================

def _flow_stress(
    exog_history,
):

    result = {
        "current":
            None,

        "high":
            None,

        "max":
            None,

        "ratio":
            1.0,
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

    q = (
        _to_numeric(
            exog_history[
                "caudal_m3s"
            ]
        )
        .dropna()
    )

    if q.empty:
        return result

    current = float(
        q.iloc[-1]
    )

    high = float(
        q.quantile(
            0.98
        )
    )

    max_value = float(
        q.max()
    )

    result[
        "current"
    ] = current

    result[
        "high"
    ] = high

    result[
        "max"
    ] = max_value

    if current > 0:

        result[
            "ratio"
        ] = max(
            high / current,
            1.0,
        )

    return result


# ============================================================
# CONSTRUIR ESCENARIO 60 DÍAS
# ============================================================

def _build_stress_curve(
    df,
    exog_history,
    upstream_history,
):

    level_col = _get_level_column(
        df
    )

    if (
        level_col is None
        or "datetime" not in df.columns
    ):

        return pd.DataFrame(), {}

    data = df.copy()

    data[
        "datetime"
    ] = pd.to_datetime(
        data[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    data[
        level_col
    ] = _to_numeric(
        data[
            level_col
        ]
    )

    data = (
        data
        .dropna(
            subset=[
                "datetime",
                level_col,
            ]
        )
        .sort_values(
            "datetime"
        )
    )

    if data.empty:

        return pd.DataFrame(), {}

    level_current = float(
        data[
            level_col
        ].iloc[-1]
    )

    last_date = (
        data[
            "datetime"
        ].iloc[-1]
    )

    future_dates = pd.date_range(
        start=
            last_date
            + pd.Timedelta(
                days=1
            ),
        periods=
            STRESS_DAYS,
        freq="D",
    )

    envelope = (
        _historical_level_envelope(
            data,
            future_dates,
        )
    )

    rain_info = (
        _rain_stress(
            exog_history
        )
    )

    flow_info = (
        _flow_stress(
            exog_history
        )
    )

    upstream_info = (
        _upstream_signal(
            upstream_history
        )
    )

    # ========================================================
    # OBJETIVO DE NIVEL
    # ========================================================

    historical_target = (
        _safe_quantile(
            data[
                level_col
            ],
            0.98,
        )
    )

    if historical_target is None:

        historical_target = (
            level_current
        )

    historical_target = max(
        historical_target,
        level_current,
    )

    # --------------------------------------------------------
    # IMPACTO POR CAUDAL
    # --------------------------------------------------------

    flow_effect = 0.0

    if (
        flow_info[
            "current"
        ] is not None
        and flow_info[
            "high"
        ] is not None
        and flow_info[
            "current"
        ] > 0
    ):

        flow_effect = (
            min(
                max(
                    flow_info[
                        "ratio"
                    ]
                    - 1.0,
                    0.0,
                ),
                1.0,
            )
            * 0.65
        )

    # --------------------------------------------------------
    # IMPACTO POR LLUVIA
    # --------------------------------------------------------

    rain_effect = min(
        rain_info[
            "daily_high"
        ]
        / 100.0,
        0.45,
    )

    # --------------------------------------------------------
    # IMPACTO AGUAS ARRIBA
    # --------------------------------------------------------

    upstream_effect = float(
        np.clip(
            (
                upstream_info[
                    "mean_delta_3"
                ]
                * 0.30
                +
                upstream_info[
                    "mean_delta_7"
                ]
                * 0.15
            ),
            -0.40,
            0.80,
        )
    )

    target = (
        level_current
        + 0.60
        * max(
            historical_target
            - level_current,
            0.0,
        )
        + flow_effect
        + rain_effect
        + max(
            upstream_effect,
            0.0,
        )
    )

    target = float(
        np.clip(
            target,
            Y_MIN,
            Y_MAX,
        )
    )

    # ========================================================
    # TRAYECTORIA AMORTIGUADA
    # ========================================================

    levels = []

    current = (
        level_current
    )

    for i in range(
        STRESS_DAYS
    ):

        # crecimiento rápido al inicio,
        # luego se estabiliza
        progress = (
            1.0
            - np.exp(
                -(i + 1)
                / 18.0
            )
        )

        theoretical = (
            level_current
            + (
                target
                - level_current
            )
            * progress
        )

        # limitar cambio diario
        max_step = 0.12

        next_level = float(
            np.clip(
                theoretical,
                current
                - max_step,
                current
                + max_step,
            )
        )

        next_level = float(
            np.clip(
                next_level,
                Y_MIN,
                Y_MAX,
            )
        )

        levels.append(
            next_level
        )

        current = (
            next_level
        )

    scenario = pd.DataFrame(
        {
            "datetime":
                future_dates,

            "stress_level":
                levels,
        }
    )

    if not envelope.empty:

        scenario = scenario.merge(
            envelope[
                [
                    "datetime",
                    "level_min_historical",
                    "level_max_historical",
                    "level_mean_historical",
                ]
            ],
            on="datetime",
            how="left",
        )

    metadata = {
        "level_current":
            level_current,

        "level_target":
            target,

        "level_historical_p98":
            historical_target,

        "rain":
            rain_info,

        "flow":
            flow_info,

        "upstream":
            upstream_info,
    }

    return (
        scenario,
        metadata,
    )


# ============================================================
# UI PRINCIPAL
# ============================================================

def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
):

    st.subheader(
        "⚠️ Escenario hipotético · 60 días"
    )

    st.caption(
        "Simulación de tipo “qué pasa si” basada en "
        "niveles históricamente elevados, lluvia intensa, "
        "caudal alto y señales de estaciones aguas arriba. "
        "No constituye un pronóstico oficial."
    )

    scenario, meta = (
        _build_stress_curve(
            df=df,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
        )
    )

    if (
        scenario is None
        or scenario.empty
        or not meta
    ):

        st.info(
            "No hay datos suficientes para construir "
            "el escenario de 60 días."
        )

        return

    level_current = meta[
        "level_current"
    ]

    level_target = meta[
        "level_target"
    ]

    rain_info = meta[
        "rain"
    ]

    flow_info = meta[
        "flow"
    ]

    upstream_info = meta[
        "upstream"
    ]

    # ========================================================
    # MÉTRICAS
    # ========================================================

    m1, m2, m3, m4 = (
        st.columns(
            4
        )
    )

    m1.metric(
        "Nivel de partida",
        f"{level_current:.2f} m",
    )

    m2.metric(
        "Nivel estimado día 60",
        f"{scenario['stress_level'].iloc[-1]:.2f} m",
        (
            f"{scenario['stress_level'].iloc[-1] - level_current:+.2f} m"
        ),
    )

    m3.metric(
        "Lluvia histórica alta",
        f"{rain_info['daily_high']:.1f} mm/día",
    )

    if (
        flow_info[
            "high"
        ] is not None
    ):

        flow_text = (
            f"{flow_info['high']:,.0f} m³/s"
        )

    else:

        flow_text = (
            "Sin dato"
        )

    m4.metric(
        "Caudal histórico alto",
        flow_text,
    )

    # ========================================================
    # SEGUNDA FILA DE MÉTRICAS
    # ========================================================

    s1, s2, s3 = st.columns(
        3
    )

    s1.metric(
        "Objetivo de estrés",
        f"{level_target:.2f} m",
    )

    s2.metric(
        "Estaciones aguas arriba",
        upstream_info[
            "available"
        ],
    )

    upstream_delta = (
        upstream_info[
            "mean_delta_3"
        ]
    )

    s3.metric(
        "Variación media aguas arriba · 3 días",
        f"{upstream_delta:+.2f} m",
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = go.Figure()

    # --------------------------------------------------------
    # MÁXIMO HISTÓRICO MISMA FECHA
    # --------------------------------------------------------

    if (
        "level_max_historical"
        in scenario.columns
        and scenario[
            "level_max_historical"
        ].notna().any()
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "level_max_historical"
                ],
                mode=
                    "lines",
                name=
                    "Máximo histórico misma fecha",
                line=dict(
                    dash="dash",
                ),
            )
        )

    # --------------------------------------------------------
    # MÍNIMO HISTÓRICO MISMA FECHA
    # --------------------------------------------------------

    if (
        "level_min_historical"
        in scenario.columns
        and scenario[
            "level_min_historical"
        ].notna().any()
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "level_min_historical"
                ],
                mode=
                    "lines",
                name=
                    "Mínimo histórico misma fecha",
                line=dict(
                    dash="dot",
                ),
            )
        )

    # --------------------------------------------------------
    # ESCENARIO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "stress_level"
            ],
            mode=
                "lines+markers",
            name=
                "Escenario de estrés",
            line=dict(
                width=3,
            ),
        )
    )

    # --------------------------------------------------------
    # NIVEL DE PARTIDA
    # --------------------------------------------------------

    fig.add_hline(
        y=
            level_current,
        line_dash=
            "dot",
        annotation_text=
            "Nivel actual",
    )

    fig.update_layout(
        height=470,
        hovermode=
            "x unified",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    fig.update_xaxes(
        title_text=
            "Fecha",
        tickformat=
            "%d/%m",
    )

    fig.update_yaxes(
        title_text=
            "Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # TABLA
    # ========================================================

    with st.expander(
        "📋 Ver escenario día por día"
    ):

        table = scenario.copy()

        table[
            "datetime"
        ] = pd.to_datetime(
            table[
                "datetime"
            ]
        ).dt.strftime(
            "%d/%m/%Y"
        )

        rename_map = {
            "datetime":
                "Fecha",

            "stress_level":
                "Nivel escenario (m)",

            "level_min_historical":
                "Mínimo histórico (m)",

            "level_max_historical":
                "Máximo histórico (m)",

            "level_mean_historical":
                "Promedio histórico (m)",
        }

        table = table.rename(
            columns=
                rename_map
        )

        numeric_cols = [
            col
            for col
            in table.columns
            if col != "Fecha"
        ]

        for col in numeric_cols:

            table[
                col
            ] = pd.to_numeric(
                table[
                    col
                ],
                errors="coerce",
            ).round(
                2
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # ACLARACIÓN
    # ========================================================

    st.info(
        "Este escenario no predice que los máximos históricos "
        "vayan a repetirse. Los utiliza como condición de estrés "
        "para analizar la respuesta potencial del nivel del río."
    )
