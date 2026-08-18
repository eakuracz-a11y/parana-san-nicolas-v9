import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# VERSIÓN
# ============================================================

STRESS_VERSION = "V11.6"


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRESS_DAYS = 60

# Escala visual fija solicitada
Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5

MIN_HISTORY_DAYS = 90

EPS = 1e-9


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_datetime(serie):

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


def _upstream_cols(df):

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
        for c in df.columns
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
# NIVEL SAN NICOLÁS
# ============================================================

def _preparar_nivel(df):

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
# LLUVIA + CAUDAL
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
# AGUAS ARRIBA
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

    hist = nivel.copy()

    exog = _preparar_exog(
        exog_history
    )

    upstream = _preparar_upstream(
        upstream_history
    )

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
# BUSCAR TODAS LAS CRECIDAS DE 60 DÍAS
# ============================================================

def _buscar_eventos_crecida(
    hist,
    days=STRESS_DAYS,
):

    eventos = []

    if (
        hist.empty
        or len(hist) < days
    ):

        return eventos

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
            .reset_index(
                drop=True
            )
        )

        nivel = (
            block[
                "nivel"
            ]
            .dropna()
        )

        if len(
            nivel
        ) < 40:

            continue

        nivel_inicio = float(
            nivel.iloc[
                0
            ]
        )

        nivel_maximo = float(
            nivel.max()
        )

        peak_idx = int(
            block[
                "nivel"
            ]
            .idxmax()
        )

        crecimiento = (
            nivel_maximo
            - nivel_inicio
        )

        if crecimiento <= 0:

            continue

        eventos.append(
            {
                "growth":
                    crecimiento,

                "initial":
                    nivel_inicio,

                "max":
                    nivel_maximo,

                "peak_idx":
                    peak_idx,

                "block":
                    block,
            }
        )

    eventos.sort(
        key=lambda x:
            x[
                "growth"
            ],
        reverse=True,
    )

    return eventos


# ============================================================
# ELEGIR EVENTO POR PERCENTIL
# ============================================================

def _evento_percentil(
    eventos,
    percentile,
):

    if not eventos:

        return None

    growths = np.array(
        [
            e[
                "growth"
            ]
            for e in eventos
        ],
        dtype=float,
    )

    target = float(
        np.quantile(
            growths,
            percentile,
        )
    )

    idx = int(
        np.argmin(
            np.abs(
                growths
                - target
            )
        )
    )

    return eventos[
        idx
    ]


# ============================================================
# FORMA COMPLETA DEL EVENTO
# ============================================================

def _event_shape(
    event,
):

    block = (
        event[
            "block"
        ]
        .copy()
    )

    niveles = (
        block[
            "nivel"
        ]
        .to_numpy(
            dtype=float
        )
    )

    initial = float(
        niveles[
            0
        ]
    )

    delta = (
        niveles
        - initial
    )

    # ========================================================
    # FILTRAR RUIDO PEQUEÑO
    # ========================================================

    delta = (
        pd.Series(
            delta
        )
        .rolling(
            3,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    peak_idx = int(
        np.nanargmax(
            delta
        )
    )

    # ========================================================
    # FASE DE CRECIDA
    # ========================================================

    rising = (
        delta[
            :peak_idx + 1
        ]
        .copy()
    )

    # Durante la crecida evitamos bajadas pequeñas
    rising = np.maximum.accumulate(
        rising
    )

    # ========================================================
    # FASE POST-PICO
    # ========================================================

    falling = (
        delta[
            peak_idx:
        ]
        .copy()
    )

    # Conservamos la bajante real histórica,
    # pero suavizada.

    falling = (
        pd.Series(
            falling
        )
        .rolling(
            5,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    shape = np.concatenate(
        [
            rising[:-1],
            falling,
        ]
    )

    if len(shape) < STRESS_DAYS:

        shape = np.pad(
            shape,
            (
                0,
                STRESS_DAYS
                - len(shape),
            ),
            mode="edge",
        )

    return shape[
        :STRESS_DAYS
    ]


# ============================================================
# CREAR ESCENARIO DESDE EVENTO
# ============================================================

def _crear_escenario(
    hist,
    event,
    factor=1.0,
):

    if event is None:

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

    shape = _event_shape(
        event
    )

    shape = (
        shape
        * float(
            factor
        )
    )

    # ========================================================
    # NO PERMITIR NIVEL NEGATIVO
    # ========================================================

    levels = (
        nivel_actual
        + shape
    )

    levels = np.clip(
        levels,
        0.0,
        12.0,
    )

    # ========================================================
    # SUAVIZACIÓN FINAL
    # ========================================================

    levels = (
        pd.Series(
            levels
        )
        .rolling(
            3,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    # Forzar exactamente el nivel real de partida
    # en la primera jornada.

    levels[
        0
    ] = max(
        nivel_actual,
        levels[
            0
        ],
    )

    block = (
        event[
            "block"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    results = []

    prev = nivel_actual

    for i in range(
        STRESS_DAYS
    ):

        if i < len(block):

            source = block.iloc[
                i
            ]

            rain = _safe_float(
                source.get(
                    "precip_mm",
                    0.0,
                ),
                0.0,
            )

            q = _safe_float(
                source.get(
                    "caudal_m3s",
                    np.nan,
                )
            )

            up = _safe_float(
                source.get(
                    "upstream_mean",
                    np.nan,
                )
            )

            source_date = source[
                "datetime"
            ]

        else:

            rain = 0.0
            q = np.nan
            up = np.nan
            source_date = pd.NaT

        current = float(
            levels[
                i
            ]
        )

        results.append(
            {
                "datetime":
                    future_dates[
                        i
                    ],

                "prediction":
                    current,

                "nivel_base":
                    prev,

                "variacion_dia":
                    current - prev,

                "precip_mm":
                    rain,

                "caudal_m3s":
                    q,

                "upstream_mean":
                    up,

                "source_date":
                    source_date,
            }
        )

        prev = current

    return pd.DataFrame(
        results
    )


# ============================================================
# ENVOLVENTE HISTÓRICA DIARIA
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
# MÉTRICAS ESCENARIO
# ============================================================

def _scenario_metrics(
    scenario,
    nivel_actual,
):

    if scenario.empty:

        return {}

    idx = (
        scenario[
            "prediction"
        ]
        .idxmax()
    )

    max_level = float(
        scenario.loc[
            idx,
            "prediction"
        ]
    )

    max_date = pd.to_datetime(
        scenario.loc[
            idx,
            "datetime"
        ]
    )

    final_level = float(
        scenario[
            "prediction"
        ]
        .iloc[
            -1
        ]
    )

    days_to_peak = int(
        scenario.loc[
            idx
        ].name
        + 1
    )

    return {

        "max":
            max_level,

        "date":
            max_date,

        "growth":
            max_level
            - nivel_actual,

        "final":
            final_level,

        "days_to_peak":
            days_to_peak,
    }


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
        "⚠️ Escenarios históricos severos · 60 días"
    )

    st.caption(
        f"{STRESS_VERSION} · "
        "Simulación histórica desde el nivel real actual. "
        "Escenarios P90, P95 y peor creciente registrada."
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
        or len(
            hist
        )
        < MIN_HISTORY_DAYS
    ):

        st.info(
            "No hay suficiente historial para calcular "
            "los escenarios severos."
        )

        return

    nivel_actual = _safe_last(
        hist[
            "nivel"
        ]
    )

    if pd.isna(
        nivel_actual
    ):

        st.info(
            "No existe un nivel actual válido."
        )

        return

    # ========================================================
    # EVENTOS HISTÓRICOS
    # ========================================================

    eventos = _buscar_eventos_crecida(
        hist
    )

    if not eventos:

        st.info(
            "No se encontraron eventos históricos "
            "de creciente suficientes."
        )

        return

    peor_evento = eventos[
        0
    ]

    evento_p95 = _evento_percentil(
        eventos,
        0.95,
    )

    evento_p90 = _evento_percentil(
        eventos,
        0.90,
    )

    # ========================================================
    # ESCENARIOS
    # ========================================================

    scenario_worst = _crear_escenario(
        hist,
        peor_evento,
        factor=1.00,
    )

    scenario_p95 = _crear_escenario(
        hist,
        evento_p95,
        factor=1.00,
    )

    scenario_p90 = _crear_escenario(
        hist,
        evento_p90,
        factor=1.00,
    )

    if scenario_worst.empty:

        st.info(
            "No fue posible construir el peor escenario."
        )

        return

    # ========================================================
    # ENVOLVENTE
    # ========================================================

    envelope = _envolvente_historica_diaria(
        df,
        scenario_worst[
            "datetime"
        ],
    )

    if not envelope.empty:

        scenario_worst = (
            scenario_worst
            .merge(
                envelope[
                    [
                        "datetime",
                        "nivel_min_historico",
                        "nivel_max_historico",
                        "nivel_promedio_historico",
                    ]
                ],
                on="datetime",
                how="left",
            )
        )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    m_worst = _scenario_metrics(
        scenario_worst,
        nivel_actual,
    )

    m95 = _scenario_metrics(
        scenario_p95,
        nivel_actual,
    )

    m90 = _scenario_metrics(
        scenario_p90,
        nivel_actual,
    )

    st.markdown(
        "**Resumen de riesgo histórico**"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
    )

    c2.metric(
        "P90 máximo",
        (
            f"{m90['max']:.2f} m"
            if m90
            else "--"
        ),
        (
            f"+{m90['growth']:.2f} m"
            if m90
            else None
        ),
    )

    c3.metric(
        "P95 máximo",
        (
            f"{m95['max']:.2f} m"
            if m95
            else "--"
        ),
        (
            f"+{m95['growth']:.2f} m"
            if m95
            else None
        ),
    )

    c4.metric(
        "Peor caso",
        f"{m_worst['max']:.2f} m",
        f"+{m_worst['growth']:.2f} m",
    )

    # ========================================================
    # SEGUNDA FILA
    # ========================================================

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Días hasta pico",
        m_worst[
            "days_to_peak"
        ],
    )

    d2.metric(
        "Fecha pico",
        m_worst[
            "date"
        ].strftime(
            "%d/%m/%Y"
        ),
    )

    d3.metric(
        "Nivel día 60",
        f"{m_worst['final']:.2f} m",
        f"{m_worst['final'] - nivel_actual:+.2f} m",
    )

    historical_growth = float(
        peor_evento[
            "growth"
        ]
    )

    d4.metric(
        "Crecida histórica patrón",
        f"+{historical_growth:.2f} m",
    )

    # ========================================================
    # MÁXIMO HISTÓRICO DEL PERÍODO
    # ========================================================

    max_hist_periodo = np.nan

    if (
        "nivel_max_historico"
        in scenario_worst.columns
        and scenario_worst[
            "nivel_max_historico"
        ]
        .notna()
        .any()
    ):

        max_hist_periodo = float(
            scenario_worst[
                "nivel_max_historico"
            ]
            .max()
        )

    if pd.notna(
        max_hist_periodo
    ):

        st.caption(
            "Máximo histórico registrado para las fechas "
            f"del horizonte: **{max_hist_periodo:.2f} m**."
        )

    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    fig = go.Figure()

    # Máximo histórico diario
    if (
        "nivel_max_historico"
        in scenario_worst.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario_worst[
                    "datetime"
                ],
                y=scenario_worst[
                    "nivel_max_historico"
                ],
                mode="lines",
                name="Máximo histórico del día",
                line=dict(
                    color="#d62728",
                    width=2,
                ),
            )
        )

    # Mínimo histórico diario
    if (
        "nivel_min_historico"
        in scenario_worst.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario_worst[
                    "datetime"
                ],
                y=scenario_worst[
                    "nivel_min_historico"
                ],
                mode="lines",
                name="Mínimo histórico del día",
                line=dict(
                    color="#2ca02c",
                    width=2,
                ),
            )
        )

    # P90
    if not scenario_p90.empty:

        fig.add_trace(
            go.Scatter(
                x=scenario_p90[
                    "datetime"
                ],
                y=scenario_p90[
                    "prediction"
                ],
                mode="lines",
                name="Escenario P90",
                line=dict(
                    color="#17becf",
                    width=2,
                    dash="dot",
                ),
            )
        )

    # P95
    if not scenario_p95.empty:

        fig.add_trace(
            go.Scatter(
                x=scenario_p95[
                    "datetime"
                ],
                y=scenario_p95[
                    "prediction"
                ],
                mode="lines",
                name="Escenario P95",
                line=dict(
                    color="#9467bd",
                    width=3,
                    dash="dash",
                ),
            )
        )

    # Peor caso
    fig.add_trace(
        go.Scatter(
            x=scenario_worst[
                "datetime"
            ],
            y=scenario_worst[
                "prediction"
            ],
            mode="lines+markers",
            name="Peor caso histórico",
            line=dict(
                color="#1f77b4",
                width=4,
            ),
            marker=dict(
                size=5,
            ),
        )
    )

    # Nivel actual
    fig.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        annotation_text=(
            f"Nivel real de partida: "
            f"{nivel_actual:.2f} m"
        ),
    )

    # Pico
    fig.add_vline(
        x=m_worst[
            "date"
        ],
        line_dash="dot",
        line_width=1,
    )

    fig.add_annotation(
        x=m_worst[
            "date"
        ],
        y=m_worst[
            "max"
        ],
        text=(
            "Pico peor caso"
            "<br>"
            f"{m_worst['max']:.2f} m"
        ),
        showarrow=True,
        arrowhead=2,
        yshift=25,
    )

    # ========================================================
    # ESCALA FIJA 0–7 m
    # ========================================================

    fig.update_layout(
        height=550,
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
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
        fixedrange=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # DÍAS SOBRE MÁXIMO HISTÓRICO
    # ========================================================

    days_above = 0

    if (
        "nivel_max_historico"
        in scenario_worst.columns
    ):

        comparison = (
            scenario_worst[
                "prediction"
            ]
            > scenario_worst[
                "nivel_max_historico"
            ]
        )

        days_above = int(
            comparison
            .fillna(False)
            .sum()
        )

    if days_above > 0:

        st.warning(
            f"⚠️ El peor escenario supera el máximo histórico "
            f"diario de referencia durante **{days_above} días**. "
            "Debe interpretarse como una simulación extrema y no "
            "como un pronóstico esperado."
        )

    # ========================================================
    # GRÁFICO DEL EVENTO HISTÓRICO ORIGEN
    # ========================================================

    st.markdown(
        "**Evento histórico utilizado como patrón de peor caso**"
    )

    block = (
        peor_evento[
            "block"
        ]
        .copy()
    )

    historical_fig = go.Figure()

    historical_fig.add_trace(
        go.Scatter(
            x=block[
                "datetime"
            ],
            y=block[
                "nivel"
            ],
            mode="lines+markers",
            name="Nivel histórico",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=4,
            ),
        )
    )

    historical_fig.update_layout(
        height=320,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=5,
        ),
        showlegend=False,
    )

    historical_fig.update_xaxes(
        title_text="Fecha histórica",
        tickformat="%d/%m/%Y",
    )

    # ========================================================
    # ESCALA FIJA 0–7 m
    # ========================================================

    historical_fig.update_yaxes(
        title_text="Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True,
    )

    # ========================================================
    # VARIABLES DEL PEOR EVENTO
    # ========================================================

    rain = (
        block[
            "precip_mm"
        ]
        .fillna(0.0)
    )

    rain_total = float(
        rain.sum()
    )

    rain_max = float(
        rain.max()
    )

    q_max = (
        float(
            block[
                "caudal_m3s"
            ]
            .max()
        )
        if block[
            "caudal_m3s"
        ]
        .notna()
        .any()
        else np.nan
    )

    up_max = (
        float(
            block[
                "upstream_mean"
            ]
            .max()
        )
        if block[
            "upstream_mean"
        ]
        .notna()
        .any()
        else np.nan
    )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Lluvia evento 60 d",
        f"{_format_number(rain_total, 1)} mm",
    )

    v2.metric(
        "Máxima lluvia diaria",
        f"{_format_number(rain_max, 1)} mm",
    )

    v3.metric(
        "Caudal máximo",
        (
            f"{_format_number(q_max, 0)} m³/s"
            if pd.notna(q_max)
            else "--"
        ),
    )

    v4.metric(
        "Aguas arriba máx. medio",
        (
            f"{up_max:.2f} m"
            if pd.notna(up_max)
            else "--"
        ),
    )

    # ========================================================
    # LLUVIA / CAUDAL
    # ========================================================

    lluvia_col, caudal_col = st.columns(
        2
    )

    with lluvia_col:

        st.markdown(
            "**🌧️ Lluvia del evento histórico**"
        )

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=block[
                    "datetime"
                ],
                y=block[
                    "precip_mm"
                ],
            )
        )

        rain_fig.update_layout(
            height=280,
            margin=dict(
                l=5,
                r=5,
                t=5,
                b=5,
            ),
            showlegend=False,
            yaxis_title="mm/día",
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    with caudal_col:

        st.markdown(
            "**💧 Caudal del evento histórico**"
        )

        if (
            block[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            q_fig = go.Figure()

            q_fig.add_trace(
                go.Scatter(
                    x=block[
                        "datetime"
                    ],
                    y=block[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                )
            )

            q_fig.update_layout(
                height=280,
                margin=dict(
                    l=5,
                    r=5,
                    t=5,
                    b=5,
                ),
                showlegend=False,
                yaxis_title="m³/s",
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
                "Sin caudal histórico disponible."
            )

    # ========================================================
    # TABLA DE AUDITORÍA
    # ========================================================

    with st.expander(
        "🔎 Auditoría del peor escenario · 60 días"
    ):

        tabla = scenario_worst.copy()

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
            .round(
                2
            )
        )

        tabla[
            "Variación diaria"
        ] = (
            tabla[
                "variacion_dia"
            ]
            .round(
                3
            )
        )

        tabla[
            "Nivel escenario"
        ] = (
            tabla[
                "prediction"
            ]
            .round(
                2
            )
        )

        tabla[
            "Lluvia"
        ] = (
            tabla[
                "precip_mm"
            ]
            .round(
                1
            )
        )

        tabla[
            "Caudal"
        ] = (
            tabla[
                "caudal_m3s"
            ]
            .round(
                0
            )
        )

        tabla[
            "Aguas arriba"
        ] = (
            tabla[
                "upstream_mean"
            ]
            .round(
                2
            )
        )

        tabla[
            "Fecha histórica origen"
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
            "Variación diaria",
            "Nivel escenario",
            "Fecha histórica origen",
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
                .round(
                    2
                )
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
        "ℹ️ Interpretación de P90, P95 y peor caso"
    ):

        st.markdown(
            f"""
            Todos los escenarios parten del nivel real disponible:

            **{nivel_actual:.2f} m**

            **P90** representa una creciente ubicada aproximadamente
            en el percentil 90 de las crecientes históricas de
            60 días encontradas.

            **P95** representa una creciente todavía más severa.

            **Peor caso histórico** reproduce la forma completa de
            la mayor creciente histórica encontrada en el historial,
            trasladándola al nivel actual.

            A diferencia de la versión anterior, después del pico
            el escenario puede **estabilizarse o comenzar una bajante**
            siguiendo el comportamiento observado históricamente.

            Las líneas roja y verde corresponden al máximo y mínimo
            histórico para cada fecha del calendario.

            Todos los gráficos de nivel hidrométrico están fijados
            en una escala de **0 a 7 metros** para facilitar la
            comparación visual entre paneles.
            """
        )

    st.warning(
        "Los escenarios P90, P95 y peor caso son simulaciones "
        "históricas de estrés. No constituyen pronósticos ni alertas "
        "oficiales."
    )
