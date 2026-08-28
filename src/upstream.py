import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/stress_ui.py
# V11.7
#
# ESCENARIO HIDROLÓGICO EXTENDIDO
#
# CAMBIO PRINCIPAL:
# El escenario puede comenzar desde el nivel proyectado
# al día 30 para evitar saltos artificiales entre:
#
# 15 días -> 30 días -> 60 días
#
# ============================================================


STRESS_DAYS = 60

Y_MIN = 0.0
Y_MAX = 7.0


# ============================================================
# UTILIDADES
# ============================================================


def _to_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _to_datetime(series):

    x = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return x.dt.tz_localize(None)


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


def _safe_quantile(
    values,
    q,
    default=0.0,
):

    if values is None:
        return default

    values = (
        _to_numeric(
            pd.Series(values)
        )
        .dropna()
    )

    if values.empty:
        return default

    try:

        value = float(
            values.quantile(q)
        )

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# NIVEL
# ============================================================


def _get_level_column(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return None

    candidates = [
        "nivel",
        "value",
        "nivel_san_nicolas",
        "prediction",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def _prepare_level_history(df):

    level_col = (
        _get_level_column(
            df
        )
    )

    if (
        level_col is None
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame()

    x = df[
        [
            "datetime",
            level_col,
        ]
    ].copy()

    x[
        "datetime"
    ] = _to_datetime(
        x[
            "datetime"
        ]
    )

    x[
        "nivel"
    ] = _to_numeric(
        x[
            level_col
        ]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    if x.empty:
        return pd.DataFrame()

    x[
        "date"
    ] = (
        x[
            "datetime"
        ]
        .dt.normalize()
    )

    x = (
        x.groupby(
            "date",
            as_index=False,
        )[
            "nivel"
        ]
        .mean()
        .rename(
            columns={
                "date":
                    "datetime"
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# EXÓGENAS
# ============================================================


def _prepare_exogenous(
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

        return pd.DataFrame()

    x = exog_history.copy()

    x[
        "datetime"
    ] = _to_datetime(
        x[
            "datetime"
        ]
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    x[
        "date"
    ] = (
        x[
            "datetime"
        ]
        .dt.normalize()
    )

    agg = {}

    if "precip_mm" in x.columns:

        x[
            "precip_mm"
        ] = _to_numeric(
            x[
                "precip_mm"
            ]
        )

        agg[
            "precip_mm"
        ] = "sum"

    if "caudal_m3s" in x.columns:

        x[
            "caudal_m3s"
        ] = _to_numeric(
            x[
                "caudal_m3s"
            ]
        )

        agg[
            "caudal_m3s"
        ] = "mean"

    if not agg:

        return pd.DataFrame()

    x = (
        x.groupby(
            "date",
            as_index=False,
        )
        .agg(
            agg
        )
        .rename(
            columns={
                "date":
                    "datetime"
            }
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# HISTÓRICO NIVEL
# ============================================================


def _historical_level_reference(
    df,
    future_dates,
):

    level = (
        _prepare_level_history(
            df
        )
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "historical_level_max_m"
    ] = np.nan

    if level.empty:

        return result

    level[
        "doy"
    ] = (
        level[
            "datetime"
        ]
        .dt.dayofyear
    )

    global_reference = (
        _safe_quantile(
            level[
                "nivel"
            ],
            0.95,
            default=float(
                level[
                    "nivel"
                ].max()
            ),
        )
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        distance = np.minimum(
            np.abs(
                level[
                    "doy"
                ]
                - doy
            ),
            365
            - np.abs(
                level[
                    "doy"
                ]
                - doy
            ),
        )

        seasonal = (
            level[
                distance <= 10
            ][
                "nivel"
            ]
            .dropna()
        )

        if not seasonal.empty:

            value = float(
                seasonal.max()
            )

        else:

            value = (
                global_reference
            )

        values.append(
            float(
                np.clip(
                    value,
                    Y_MIN,
                    Y_MAX,
                )
            )
        )

    result[
        "historical_level_max_m"
    ] = values

    return result


# ============================================================
# LLUVIA HISTÓRICA
# ============================================================


def _historical_rain_reference(
    exog_history,
    future_dates,
):

    exog = (
        _prepare_exogenous(
            exog_history
        )
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "rain_historical_max_mm"
    ] = 0.0

    if (
        exog.empty
        or "precip_mm"
        not in exog.columns
    ):

        return result

    rain = exog[
        [
            "datetime",
            "precip_mm",
        ]
    ].copy()

    rain[
        "precip_mm"
    ] = _to_numeric(
        rain[
            "precip_mm"
        ]
    )

    rain = rain.dropna(
        subset=[
            "precip_mm"
        ]
    )

    if rain.empty:

        return result

    rain[
        "doy"
    ] = (
        rain[
            "datetime"
        ]
        .dt.dayofyear
    )

    global_reference = (
        _safe_quantile(
            rain[
                "precip_mm"
            ],
            0.95,
            default=0.0,
        )
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        distance = np.minimum(
            np.abs(
                rain[
                    "doy"
                ]
                - doy
            ),
            365
            - np.abs(
                rain[
                    "doy"
                ]
                - doy
            ),
        )

        seasonal = (
            rain[
                distance <= 7
            ][
                "precip_mm"
            ]
            .dropna()
        )

        if not seasonal.empty:

            value = float(
                seasonal.max()
            )

        else:

            value = (
                global_reference
            )

        values.append(
            max(
                0.0,
                _safe_float(
                    value,
                    0.0,
                ),
            )
        )

    result[
        "rain_historical_max_mm"
    ] = values

    return result


# ============================================================
# CAUDAL HISTÓRICO
# ============================================================


def _historical_flow_reference(
    exog_history,
    future_dates,
):

    exog = (
        _prepare_exogenous(
            exog_history
        )
    )

    result = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates
                )
        }
    )

    result[
        "flow_historical_max_m3s"
    ] = np.nan

    if (
        exog.empty
        or "caudal_m3s"
        not in exog.columns
    ):

        return result

    q = exog[
        [
            "datetime",
            "caudal_m3s",
        ]
    ].copy()

    q[
        "caudal_m3s"
    ] = _to_numeric(
        q[
            "caudal_m3s"
        ]
    )

    q = q.dropna(
        subset=[
            "caudal_m3s"
        ]
    )

    if q.empty:

        return result

    q[
        "doy"
    ] = (
        q[
            "datetime"
        ]
        .dt.dayofyear
    )

    global_reference = (
        _safe_quantile(
            q[
                "caudal_m3s"
            ],
            0.95,
            default=np.nan,
        )
    )

    values = []

    for future_date in result[
        "datetime"
    ]:

        doy = int(
            future_date.dayofyear
        )

        distance = np.minimum(
            np.abs(
                q[
                    "doy"
                ]
                - doy
            ),
            365
            - np.abs(
                q[
                    "doy"
                ]
                - doy
            ),
        )

        seasonal = (
            q[
                distance <= 10
            ][
                "caudal_m3s"
            ]
            .dropna()
        )

        if not seasonal.empty:

            value = float(
                seasonal.max()
            )

        else:

            value = (
                global_reference
            )

        values.append(
            _safe_float(
                value,
                np.nan,
            )
        )

    result[
        "flow_historical_max_m3s"
    ] = values

    return result


# ============================================================
# SEÑAL AGUAS ARRIBA
# ============================================================


def _upstream_signal(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
    ):

        return 0.0

    level_cols = [
        col
        for col
        in upstream_history.columns
        if str(
            col
        ).startswith(
            "nivel_"
        )
    ]

    signals = []

    for col in level_cols:

        values = (
            _to_numeric(
                upstream_history[
                    col
                ]
            )
            .dropna()
        )

        if len(values) < 2:
            continue

        if len(values) >= 7:

            delta = float(
                values.iloc[-1]
                - values.iloc[-7]
            )

        else:

            delta = float(
                values.iloc[-1]
                - values.iloc[0]
            )

        signals.append(
            delta
        )

    if not signals:

        return 0.0

    return float(
        np.clip(
            np.nanmedian(
                signals
            ),
            -1.5,
            1.5,
        )
    )


# ============================================================
# TENDENCIA LOCAL
# ============================================================


def _recent_level_slope(
    level_history,
):

    if (
        level_history is None
        or level_history.empty
    ):

        return 0.0

    values = (
        _to_numeric(
            level_history[
                "nivel"
            ]
        )
        .dropna()
        .tail(
            10
        )
    )

    if len(values) < 3:

        return 0.0

    x = np.arange(
        len(values),
        dtype=float,
    )

    y = values.to_numpy(
        dtype=float
    )

    try:

        slope = float(
            np.polyfit(
                x,
                y,
                1,
            )[0]
        )

    except Exception:

        slope = 0.0

    return float(
        np.clip(
            slope,
            -0.10,
            0.10,
        )
    )


# ============================================================
# SEÑAL DE CAUDAL
# ============================================================


def _flow_signal(
    exog_history,
):

    exog = (
        _prepare_exogenous(
            exog_history
        )
    )

    if (
        exog.empty
        or "caudal_m3s"
        not in exog.columns
    ):

        return {
            "trend": 0.0,
            "current": np.nan,
        }

    q = (
        _to_numeric(
            exog[
                "caudal_m3s"
            ]
        )
        .dropna()
    )

    if q.empty:

        return {
            "trend": 0.0,
            "current": np.nan,
        }

    current = float(
        q.iloc[-1]
    )

    if len(q) >= 7:

        base = float(
            q.iloc[-7]
        )

    else:

        base = float(
            q.iloc[0]
        )

    if (
        np.isfinite(base)
        and abs(base) > 1e-9
    ):

        trend = (
            current
            - base
        ) / abs(
            base
        )

    else:

        trend = 0.0

    return {
        "trend":
            float(
                np.clip(
                    trend,
                    -1.0,
                    1.0,
                )
            ),

        "current":
            current,
    }


# ============================================================
# CONSTRUIR ESCENARIO
# ============================================================


def build_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
    anchor_date=None,
    anchor_level=None,
    anchor_day=0,
):
    """
    Construye el escenario extendido.

    anchor_date:
        fecha desde donde comienza el escenario.

    anchor_level:
        nivel inicial del escenario.

    anchor_day:
        número de día dentro de la proyección total.

        Ejemplo:
        anchor_day=30
        genera días 31 a 60.

    Si no se informa ancla:
        mantiene compatibilidad con versiones anteriores
        y comienza desde el último nivel observado.
    """

    level_history = (
        _prepare_level_history(
            df
        )
    )

    if level_history.empty:

        return pd.DataFrame()


    # ========================================================
    # PUNTO DE INICIO
    # ========================================================

    last_observed_date = (
        pd.Timestamp(
            level_history[
                "datetime"
            ].iloc[-1]
        )
    )

    last_observed_level = float(
        level_history[
            "nivel"
        ].iloc[-1]
    )


    if anchor_date is None:

        start_date = (
            last_observed_date
        )

    else:

        start_date = (
            pd.Timestamp(
                anchor_date
            )
        )


    if anchor_level is None:

        start_level = (
            last_observed_level
        )

    else:

        start_level = float(
            anchor_level
        )


    start_level = float(
        np.clip(
            start_level,
            Y_MIN,
            Y_MAX,
        )
    )


    days = int(
        max(
            days,
            anchor_day + 1,
        )
    )


    remaining_days = (
        days
        - int(
            anchor_day
        )
    )


    if remaining_days <= 0:

        return pd.DataFrame()


    future_dates = (
        pd.date_range(
            start=(
                start_date
                + pd.Timedelta(
                    days=1
                )
            ),
            periods=
                remaining_days,
            freq="D",
        )
    )


    # ========================================================
    # REFERENCIAS HISTÓRICAS
    # ========================================================

    scenario = pd.DataFrame(
        {
            "datetime":
                future_dates
        }
    )


    level_reference = (
        _historical_level_reference(
            df,
            future_dates,
        )
    )

    rain_reference = (
        _historical_rain_reference(
            exog_history,
            future_dates,
        )
    )

    flow_reference = (
        _historical_flow_reference(
            exog_history,
            future_dates,
        )
    )


    scenario = scenario.merge(
        level_reference,
        on="datetime",
        how="left",
    )

    scenario = scenario.merge(
        rain_reference,
        on="datetime",
        how="left",
    )

    scenario = scenario.merge(
        flow_reference,
        on="datetime",
        how="left",
    )


    # ========================================================
    # SEÑALES
    # ========================================================

    upstream_signal = (
        _upstream_signal(
            upstream_history
        )
    )

    local_slope = (
        _recent_level_slope(
            level_history
        )
    )

    flow_signal = (
        _flow_signal(
            exog_history
        )
    )


    rain_reference_value = (
        _safe_quantile(
            scenario[
                "rain_historical_max_mm"
            ],
            0.80,
            default=1.0,
        )
    )

    if rain_reference_value <= 0:

        rain_reference_value = (
            1.0
        )


    flow_reference_value = (
        _safe_quantile(
            scenario[
                "flow_historical_max_m3s"
            ],
            0.50,
            default=np.nan,
        )
    )


    # ========================================================
    # CURVA
    # ========================================================

    levels = []

    daily_changes = []

    previous_level = (
        start_level
    )


    for i, row in scenario.iterrows():

        extension_day = (
            i + 1
        )

        total_day = (
            int(
                anchor_day
            )
            + extension_day
        )


        # ----------------------------------------------------
        # TENDENCIA HEREDADA
        #
        # Tiene mayor peso al comienzo y se amortigua.
        # ----------------------------------------------------

        persistence = (
            local_slope
            * np.exp(
                -extension_day
                / 18.0
            )
        )


        # ----------------------------------------------------
        # AGUAS ARRIBA
        # ----------------------------------------------------

        upstream_component = (
            0.018
            * upstream_signal
            * (
                1.0
                - np.exp(
                    -extension_day
                    / 10.0
                )
            )
        )


        # ----------------------------------------------------
        # CAUDAL ACTUAL
        # ----------------------------------------------------

        flow_component = (
            0.010
            * flow_signal.get(
                "trend",
                0.0,
            )
        )


        # ----------------------------------------------------
        # CAUDAL HISTÓRICO ESTACIONAL
        # ----------------------------------------------------

        future_flow = (
            _safe_float(
                row.get(
                    "flow_historical_max_m3s"
                ),
                np.nan,
            )
        )

        if (
            np.isfinite(
                future_flow
            )
            and np.isfinite(
                flow_reference_value
            )
            and flow_reference_value > 0
        ):

            flow_ratio = (
                future_flow
                / flow_reference_value
            )

            historical_flow_component = (
                0.008
                * np.clip(
                    flow_ratio - 1.0,
                    -0.5,
                    1.5,
                )
            )

        else:

            historical_flow_component = (
                0.0
            )


        # ----------------------------------------------------
        # LLUVIA HISTÓRICA ESTACIONAL
        # ----------------------------------------------------

        future_rain = max(
            0.0,
            _safe_float(
                row.get(
                    "rain_historical_max_mm"
                ),
                0.0,
            ),
        )

        rain_ratio = (
            future_rain
            / rain_reference_value
        )

        rain_component = (
            0.006
            * np.clip(
                rain_ratio,
                0.0,
                3.0,
            )
        )


        # ----------------------------------------------------
        # CAMBIO PROPUESTO
        # ----------------------------------------------------

        raw_change = (
            persistence
            + upstream_component
            + flow_component
            + historical_flow_component
            + rain_component
        )


        # ----------------------------------------------------
        # REFERENCIA HISTÓRICA
        #
        # Se usa como atracción suave, NO como valor objetivo
        # obligatorio.
        # ----------------------------------------------------

        historical_level = (
            _safe_float(
                row.get(
                    "historical_level_max_m"
                ),
                previous_level,
            )
        )


        difference_to_history = (
            historical_level
            - previous_level
        )


        historical_component = (
            difference_to_history
            * 0.008
        )


        raw_change += (
            historical_component
        )


        # ====================================================
        # LÍMITES DE VARIACIÓN
        #
        # Evita caídas o aumentos exagerados.
        #
        # Para horizonte 31-60 preferimos una transición
        # progresiva.
        # ====================================================

        max_drop = -0.035
        max_rise = 0.080


        # Si aguas arriba está claramente creciendo,
        # evitamos una caída fuerte.
        if upstream_signal > 0.10:

            max_drop = -0.015


        # Si el caudal está creciendo,
        # también limitamos la caída.
        if (
            flow_signal.get(
                "trend",
                0.0,
            )
            > 0.05
        ):

            max_drop = max(
                max_drop,
                -0.012,
            )


        daily_change = float(
            np.clip(
                raw_change,
                max_drop,
                max_rise,
            )
        )


        # ----------------------------------------------------
        # SUAVIZADO RESPECTO AL DÍA ANTERIOR
        # ----------------------------------------------------

        if daily_changes:

            previous_change = (
                daily_changes[-1]
            )

            daily_change = (
                0.65
                * previous_change
                + 0.35
                * daily_change
            )


        proposed_level = (
            previous_level
            + daily_change
        )


        proposed_level = float(
            np.clip(
                proposed_level,
                Y_MIN,
                Y_MAX,
            )
        )


        levels.append(
            proposed_level
        )

        daily_changes.append(
            daily_change
        )

        previous_level = (
            proposed_level
        )


    # ========================================================
    # RESULTADO
    # ========================================================

    scenario[
        "stress_level"
    ] = levels


    scenario[
        "daily_change"
    ] = daily_changes


    scenario[
        "scenario_day"
    ] = np.arange(
        int(
            anchor_day
        ) + 1,
        days + 1,
    )


    scenario[
        "upstream_signal"
    ] = upstream_signal


    scenario[
        "current_flow_m3s"
    ] = (
        flow_signal.get(
            "current",
            np.nan,
        )
    )


    scenario[
        "scenario_type"
    ] = (
        "historical_hydrological"
    )


    scenario[
        "anchor_level"
    ] = (
        start_level
    )


    return scenario


# ============================================================
# API PARA APP.PY
# ============================================================


def get_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
    anchor_date=None,
    anchor_level=None,
    anchor_day=0,
):

    return build_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
        days=days,
        anchor_date=
            anchor_date,
        anchor_level=
            anchor_level,
        anchor_day=
            anchor_day,
    )


# ============================================================
# RENDER OPCIONAL
# ============================================================


def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
    days=STRESS_DAYS,
    anchor_date=None,
    anchor_level=None,
    anchor_day=0,
):

    st.subheader(
        "⚠️ Escenario hidrológico extendido"
    )


    scenario = (
        build_stress_scenario(
            df=df,
            models=models,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
            days=days,
            anchor_date=
                anchor_date,
            anchor_level=
                anchor_level,
            anchor_day=
                anchor_day,
        )
    )


    if scenario.empty:

        st.info(
            "No hay suficientes datos "
            "para construir el escenario."
        )

        return scenario


    level_history = (
        _prepare_level_history(
            df
        )
    )


    observed_level = float(
        level_history[
            "nivel"
        ].iloc[-1]
    )


    first_level = float(
        scenario[
            "anchor_level"
        ].iloc[0]
    )


    final_level = float(
        scenario[
            "stress_level"
        ].iloc[-1]
    )


    maximum_level = float(
        scenario[
            "stress_level"
        ].max()
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Nivel inicio escenario",
        f"{first_level:.2f} m",
    )


    c2.metric(
        "Nivel final",
        f"{final_level:.2f} m",
        f"{final_level - first_level:+.2f} m",
    )


    c3, c4 = st.columns(
        2
    )


    c3.metric(
        "Máximo escenario",
        f"{maximum_level:.2f} m",
    )


    c4.metric(
        "Nivel observado actual",
        f"{observed_level:.2f} m",
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=scenario[
                "datetime"
            ],
            y=scenario[
                "stress_level"
            ],
            mode="lines+markers",
            name="Escenario",
        )
    )


    if (
        "historical_level_max_m"
        in scenario.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario[
                    "datetime"
                ],
                y=scenario[
                    "historical_level_max_m"
                ],
                mode="lines",
                line=dict(
                    dash="dot"
                ),
                name="Máximo histórico estacional",
            )
        )


    fig.update_layout(
        height=420,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10,
        ),
    )


    fig.update_yaxes(
        title_text="Nivel (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=0.5,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    return scenario
