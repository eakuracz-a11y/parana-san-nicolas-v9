# ============================================================
# PARANÁ · SAN NICOLÁS
# app.py
# V11.14 COMPLETO
#
# Dashboard Streamlit
#
# Integra:
# ------------------------------------------------------------
# - Nivel INA San Nicolás
# - Niveles aguas arriba
# - Caudales observados / reconstruidos
# - Lluvia por estación
# - Pronóstico ML 60 días
# - Propagación Corrientes -> San Nicolás
# - Escenarios históricos:
#       Probable
#       Adverso
#       Extremo histórico
# - Comparativa año contra año
# - Diagnóstico de cobertura del modelo
#
# ============================================================


from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# IMPORTACIONES DEL PROYECTO
# ============================================================

from src.ina import observed

from src.upstream import (
    get_upstream_history,
)

from src.exogenous import (
    get_exogenous_data,
)

from src.hydrology import (
    analizar_corrientes_san_nicolas,
)

from src.model import (
    train,
    predict,
)


# ============================================================
# VERSIÓN
# ============================================================

APP_VERSION = "V11.16"

APP_SUBTITLE = (
    "Pronóstico hidrológico multivariable · "
    "caudal + lluvia + propagación + escenarios históricos"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

FORECAST_DAYS = 60

DEFAULT_VISIBLE_HISTORY_DAYS = 120

DEFAULT_TRAINING_DAYS = 365 * 8

MIN_TRAINING_DAYS = 365 * 3

MAX_TRAINING_DAYS = 365 * 15

# Histórico independiente para relaciones hidrológicas.
# Se usa una ventana más extensa que la elegida para entrenamiento.
HYDROLOGY_HISTORY_YEARS = 20


# ============================================================
# ESTACIONES
# ============================================================

STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]


LEVEL_COLUMNS = {

    "Corrientes":
        "nivel_corrientes",

    "Goya":
        "nivel_goya",

    "La Paz":
        "nivel_la_paz",

    "Paraná":
        "nivel_parana",

    "Diamante":
        "nivel_diamante",

    "Rosario":
        "nivel_rosario",

    "Villa Constitución":
        "nivel_villa_constitucion",

    "San Nicolás":
        "nivel_san_nicolas",
}


FLOW_COLUMNS = {

    "Corrientes":
        "q_corrientes",

    "Goya":
        "q_goya",

    "La Paz":
        "q_la_paz",

    "Paraná":
        "q_parana",

    "Diamante":
        "q_diamante",

    "Rosario":
        "q_rosario",

    "Villa Constitución":
        "q_villa_constitucion",

    "San Nicolás":
        "q_san_nicolas",
}


RAIN_COLUMNS = {

    "Corrientes":
        "rain_corrientes",

    "Goya":
        "rain_goya",

    "La Paz":
        "rain_la_paz",

    "Paraná":
        "rain_parana",

    "Diamante":
        "rain_diamante",

    "Rosario":
        "rain_rosario",

    "Villa Constitución":
        "rain_villa_constitucion",

    "San Nicolás":
        "rain_san_nicolas",
}


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(
    page_title=
        "Paraná · San Nicolás",

    page_icon=
        "🌊",

    layout=
        "wide",

    initial_sidebar_state=
        "expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1 {
        margin-bottom: 0.1rem;
    }

    h2 {
        margin-top: 1.4rem;
    }

    h3 {
        margin-top: 1rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(127,127,127,0.06);
        border: 1px solid rgba(127,127,127,0.14);
        padding: 12px 14px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.88rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .small-note {
        opacity: 0.72;
        font-size: 0.82rem;
    }

    .status-box {
        border: 1px solid rgba(127,127,127,0.17);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }

    div[data-testid="stDataFrame"] {
        font-size: 0.88rem;
    }

    @media (max-width: 800px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }

        h1 {
            font-size: 1.75rem !important;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(
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


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def naive_datetime(values):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def normalize_frame(
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

        return pd.DataFrame()

    result = df.copy()

    if "datetime" in result.columns:

        result[
            "datetime"
        ] = naive_datetime(
            result[
                "datetime"
            ]
        )

        result = (
            result
            .dropna(
                subset=[
                    "datetime"
                ]
            )
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

    return result


def fmt_level(
    value,
):

    value = safe_float(
        value
    )

    if not np.isfinite(
        value
    ):

        return "—"

    return (
        f"{value:.2f} m"
    )


def fmt_flow(
    value,
):

    value = safe_float(
        value
    )

    if not np.isfinite(
        value
    ):

        return "—"

    return (
        f"{value:,.0f} m³/s"
        .replace(",", ".")
    )


def fmt_delta_level(
    value,
):

    value = safe_float(
        value
    )

    if not np.isfinite(
        value
    ):

        return "—"

    arrow = (
        "↑"
        if value > 0.01
        else (
            "↓"
            if value < -0.01
            else "→"
        )
    )

    sign = (
        "+"
        if value > 0
        else ""
    )

    return (
        f"{arrow} {sign}{value:.2f} m"
    )


def fmt_percent(
    value,
):

    value = safe_float(
        value
    )

    if not np.isfinite(
        value
    ):

        return "—"

    return (
        f"{value * 100:+.1f}%"
    )


def current_value(
    df,
    column,
):

    if (
        df is None
        or df.empty
        or column
        not in df.columns
    ):

        return np.nan

    values = (
        numeric(
            df[
                column
            ]
        )
        .dropna()
    )

    if values.empty:

        return np.nan

    return float(
        values.iloc[-1]
    )


def delta_days(
    df,
    column,
    days,
):

    if (
        df is None
        or df.empty
        or column
        not in df.columns
    ):

        return np.nan

    values = (
        numeric(
            df[
                column
            ]
        )
        .dropna()
    )

    if len(values) <= days:

        return np.nan

    return float(
        values.iloc[-1]
        -
        values.iloc[
            -(days + 1)
        ]
    )


# ============================================================
# NORMALIZACIÓN PARA COMPARAR ESTACIONES
# ============================================================

def normalized_change(
    series,
):
    """
    Convierte una serie a cambio respecto del primer dato válido.
    Permite comparar formas de Corrientes y San Nicolás sin asumir
    que sus ceros hidrométricos son equivalentes.
    """
    x = numeric(series)

    valid = x.dropna()

    if valid.empty:
        return x * np.nan

    base = float(valid.iloc[0])

    return x - base


# ============================================================
# EXTRAER SN
# ============================================================

def prepare_sn_observed(
    df,
):

    df = normalize_frame(
        df
    )

    if df.empty:

        return pd.DataFrame()

    level_col = None

    for candidate in [
        "nivel",
        "value",
        "nivel_san_nicolas",
    ]:

        if candidate in df.columns:

            level_col = candidate
            break

    if level_col is None:

        return pd.DataFrame()

    result = df[
        [
            "datetime",
            level_col,
        ]
    ].copy()

    result[
        "nivel"
    ] = numeric(
        result[
            level_col
        ]
    )

    result = (
        result
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return result[
        [
            "datetime",
            "nivel",
        ]
    ]


# ============================================================
# ESTADO DEL NIVEL
# ============================================================

def level_state(
    delta_7,
):

    delta_7 = safe_float(
        delta_7
    )

    if not np.isfinite(
        delta_7
    ):

        return "Sin tendencia"

    if delta_7 > 0.10:

        return "🔵 ↑ Creciente"

    if delta_7 < -0.10:

        return "🔴 ↓ Decreciente"

    return "🟢 → Estable"


# ============================================================
# ESTADO DEL CAUDAL
# ============================================================

def flow_state(
    history,
):

    if (
        history is None
        or history.empty
        or "caudal_m3s"
        not in history.columns
    ):

        return "Sin tendencia"

    q = (
        numeric(
            history[
                "caudal_m3s"
            ]
        )
        .dropna()
    )

    if len(q) < 8:

        return "Sin tendencia"

    current = float(
        q.iloc[-1]
    )

    previous = float(
        q.iloc[-8]
    )

    if previous <= 0:

        return "Sin tendencia"

    variation = (
        current
        - previous
    ) / previous

    if variation > 0.05:

        return "↑ Creciente"

    if variation < -0.05:

        return "↓ Decreciente"

    return "→ Estable"


# ============================================================
# DYNAMIC Y RANGE
# ============================================================

def dynamic_level_range(
    *series_list,
):

    values = []

    for series in series_list:

        if series is None:
            continue

        try:

            x = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                )
                .dropna()
            )

            if not x.empty:

                values.extend(
                    x.tolist()
                )

        except Exception:
            continue

    if not values:

        return None

    low = float(
        np.nanmin(
            values
        )
    )

    high = float(
        np.nanmax(
            values
        )
    )

    span = max(
        high - low,
        0.30,
    )

    padding = max(
        0.15,
        span * 0.12,
    )

    return [
        low - padding,
        high + padding,
    ]


# ============================================================
# SESIÓN
# ============================================================

if "data_loaded" not in st.session_state:

    st.session_state[
        "data_loaded"
    ] = False


# ============================================================
# CABECERA
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · {APP_SUBTITLE}"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Configuración"
    )

    base_date = st.date_input(

        "Fecha base",

        value=
            date.today(),

        max_value=
            date.today(),
    )

    visible_days = st.slider(

        "Historia visible",

        min_value=30,

        max_value=730,

        value=
            DEFAULT_VISIBLE_HISTORY_DAYS,

        step=30,
    )

    training_years = st.slider(

        "Años para entrenamiento",

        min_value=3,

        max_value=15,

        value=8,

        step=1,
    )

    st.caption(
        "El módulo hidrológico utiliza hasta 20 años para relaciones "
        "Corrientes → San Nicolás; el entrenamiento conserva la ventana "
        "seleccionada para estabilidad y velocidad."
    )

    update_clicked = st.button(
        "🔄 Actualizar datos y modelo",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.caption(
        "Horizonte"
    )

    st.write(
        "15 días · pronóstico"
    )

    st.write(
        "30 días · proyección"
    )

    st.write(
        "45 días · escenario extendido"
    )

    st.write(
        "60 días · tendencia hidrológica"
    )


# ============================================================
# CARGAR DATOS
# ============================================================

if (
    update_clicked
    or not st.session_state[
        "data_loaded"
    ]
):

    with st.spinner(
        "Consultando INA, lluvia, caudales y entrenando modelo..."
    ):

        try:

            base_ts = pd.Timestamp(
                base_date
            ).normalize()

            # =================================================
            # PERÍODOS
            # =================================================

            visible_start = (
                base_ts
                - pd.Timedelta(
                    days=
                        visible_days
                )
            )

            training_days = int(
                np.clip(
                    training_years
                    * 365,

                    MIN_TRAINING_DAYS,

                    MAX_TRAINING_DAYS,
                )
            )

            training_start = (
                base_ts
                - pd.Timedelta(
                    days=
                        training_days
                )
            )

            hydrology_start = (
                base_ts
                - pd.DateOffset(
                    years=HYDROLOGY_HISTORY_YEARS
                )
            ).normalize()

            # =================================================
            # SAN NICOLÁS
            # =================================================

            sn_raw, sn_error = observed(

                hydrology_start.strftime(
                    "%Y-%m-%d"
                ),

                base_ts.strftime(
                    "%Y-%m-%d"
                ),
            )

            if sn_error:

                raise RuntimeError(
                    sn_error
                )

            sn_hydrology_history = prepare_sn_observed(
                sn_raw
            )

            if sn_hydrology_history.empty:

                raise RuntimeError(
                    "INA no devolvió niveles válidos para San Nicolás."
                )

            sn_history = sn_hydrology_history[
                sn_hydrology_history["datetime"] >= training_start
            ].copy().reset_index(drop=True)

            if sn_history.empty:
                sn_history = sn_hydrology_history.copy()

            # =================================================
            # UPSTREAM
            # =================================================

            (
                upstream_history,
                upstream_meta,
            ) = get_upstream_history(

                hydrology_start.strftime(
                    "%Y-%m-%d"
                ),

                base_ts.strftime(
                    "%Y-%m-%d"
                ),
            )

            upstream_hydrology_history = (
                normalize_frame(
                    upstream_history
                )
            )

            upstream_history = upstream_hydrology_history[
                upstream_hydrology_history["datetime"] >= training_start
            ].copy().reset_index(drop=True)

            # =================================================
            # EXÓGENAS
            # =================================================

            # Niveles diarios disponibles para reconstruir caudales
            # cuando INA no publica una serie de caudal utilizable.
            level_history_for_flow = upstream_history.copy()

            sn_levels_for_flow = sn_history[["datetime", "nivel"]].copy()
            sn_levels_for_flow = sn_levels_for_flow.rename(
                columns={"nivel": "nivel_san_nicolas"}
            )

            if level_history_for_flow.empty:
                level_history_for_flow = sn_levels_for_flow
            else:
                level_history_for_flow = level_history_for_flow.merge(
                    sn_levels_for_flow,
                    on="datetime",
                    how="outer",
                )

            (
                exog_history,
                exog_future,
                exog_meta,
            ) = get_exogenous_data(

                training_start.strftime(
                    "%Y-%m-%d"
                ),

                base_ts.strftime(
                    "%Y-%m-%d"
                ),

                forecast_days=
                    FORECAST_DAYS,

                level_history=
                    level_history_for_flow,
            )

            exog_history = normalize_frame(
                exog_history
            )

            exog_future = normalize_frame(
                exog_future
            )

            # =================================================
            # HIDROLOGÍA
            # =================================================

            hydrology = (
                analizar_corrientes_san_nicolas(

                    sn_hydrology_history,

                    upstream_hydrology_history,

                    exog_history=
                        exog_history,

                    exog_future=
                        exog_future,

                    days=
                        FORECAST_DAYS,
                )
            )

            # =================================================
            # MODELO
            # =================================================

            models, metrics = train(

                sn_history,

                exog_history=
                    exog_history,

                upstream_history=
                    upstream_history,

                hydrology=
                    hydrology,
            )

            # =================================================
            # PRONÓSTICO 60 DÍAS
            # =================================================

            forecast = predict(

                sn_history,

                models,

                days=
                    FORECAST_DAYS,

                exog_future=
                    exog_future,

                upstream_future=
                    None,

                hydrology=
                    hydrology,
            )

            forecast = normalize_frame(
                forecast
            )

            if forecast.empty:

                raise RuntimeError(
                    "El modelo no generó pronóstico."
                )

            # =================================================
            # GUARDAR
            # =================================================

            st.session_state[
                "sn_history"
            ] = sn_history

            st.session_state[
                "upstream_history"
            ] = upstream_history

            st.session_state[
                "upstream_meta"
            ] = upstream_meta

            st.session_state[
                "exog_history"
            ] = exog_history

            st.session_state[
                "exog_future"
            ] = exog_future

            st.session_state[
                "exog_meta"
            ] = exog_meta

            st.session_state[
                "hydrology"
            ] = hydrology

            st.session_state[
                "models"
            ] = models

            st.session_state[
                "metrics"
            ] = metrics

            st.session_state[
                "forecast"
            ] = forecast

            st.session_state[
                "base_date"
            ] = base_ts

            st.session_state[
                "visible_start"
            ] = visible_start

            st.session_state[
                "last_update"
            ] = datetime.now()

            st.session_state[
                "data_loaded"
            ] = True

            st.session_state[
                "load_error"
            ] = None

        except Exception as exc:

            st.session_state[
                "load_error"
            ] = str(
                exc
            )

            st.session_state[
                "data_loaded"
            ] = False


# ============================================================
# ERROR
# ============================================================

if not st.session_state.get(
    "data_loaded",
    False,
):

    error = st.session_state.get(
        "load_error"
    )

    if error:

        st.error(
            "No fue posible actualizar los datos y el modelo.\n\n"
            + error
        )

    else:

        st.info(
            "Presioná «Actualizar datos y modelo»."
        )

    st.stop()


# ============================================================
# RECUPERAR ESTADO
# ============================================================

sn_history = st.session_state[
    "sn_history"
]

upstream_history = st.session_state[
    "upstream_history"
]

upstream_meta = st.session_state[
    "upstream_meta"
]

exog_history = st.session_state[
    "exog_history"
]

exog_future = st.session_state[
    "exog_future"
]

exog_meta = st.session_state[
    "exog_meta"
]

hydrology = st.session_state[
    "hydrology"
]

models = st.session_state[
    "models"
]

metrics = st.session_state[
    "metrics"
]

forecast = st.session_state[
    "forecast"
]

base_ts = st.session_state[
    "base_date"
]

visible_start = (
    base_ts
    - pd.Timedelta(
        days=
            visible_days
    )
)


# ============================================================
# ÉXITO
# ============================================================

st.success(
    "Datos y modelo actualizados correctamente."
)


# ============================================================
# ESTADO ACTUAL
# ============================================================

current_level = current_value(
    sn_history,
    "nivel",
)

delta_1 = delta_days(
    sn_history,
    "nivel",
    1,
)

delta_7 = delta_days(
    sn_history,
    "nivel",
    7,
)


current_flow = current_value(
    exog_history,
    "caudal_m3s",
)


flow_station = exog_meta.get(
    "main_flow_station"
)

flow_status = flow_state(
    exog_history
)


hydro_estimate = hydrology.get(
    "current_estimate",
    {}
)

delay = hydro_estimate.get(
    "delay_days"
)

delay_min = hydro_estimate.get(
    "delay_min"
)

delay_max = hydro_estimate.get(
    "delay_max"
)

correlation = safe_float(
    hydro_estimate.get(
        "correlation"
    )
)


# ============================================================
# TARJETAS PRINCIPALES
# ============================================================

st.subheader(
    "Estado actual"
)

c1, c2, c3, c4 = st.columns(
    4
)

with c1:

    st.metric(

        "San Nicolás",

        fmt_level(
            current_level
        ),

        fmt_delta_level(
            delta_1
        ),
    )


with c2:

    st.metric(

        "Tendencia 7 días",

        level_state(
            delta_7
        ),

        fmt_delta_level(
            delta_7
        ),
    )


with c3:

    st.metric(

        "Caudal",

        fmt_flow(
            current_flow
        ),

        flow_status,
    )

    if flow_station:

        st.caption(
            f"Serie de referencia: {flow_station}"
        )


with c4:

    if delay is not None:

        st.metric(
            "Demora Corrientes → SN",
            f"{delay} días",
            (
                f"rango {delay_min}–{delay_max} días"
                if (
                    delay_min is not None
                    and delay_max is not None
                )
                else "—"
            ),
        )

        if np.isfinite(correlation):
            st.caption(
                f"Correlación anual robusta: {correlation:.2f}"
            )

    else:

        st.metric(
            "Demora Corrientes → SN",
            "—",
        )


# ============================================================
# TRADUCCIÓN CORRIENTES -> SAN NICOLÁS
# ============================================================

response_ratio = safe_float(
    hydro_estimate.get(
        "response_m_per_m"
    )
)

corrientes_change_7d = safe_float(
    hydro_estimate.get(
        "corrientes_change_7d"
    )
)

expected_sn_change = safe_float(
    hydro_estimate.get(
        "expected_sn_change"
    )
)

if (
    np.isfinite(response_ratio)
    or np.isfinite(corrientes_change_7d)
):
    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "Corrientes · variación 7 días",
            (
                f"{corrientes_change_7d:+.2f} m"
                if np.isfinite(corrientes_change_7d)
                else "—"
            ),
        )

    with t2:
        st.metric(
            "Respuesta histórica SN / Corrientes",
            (
                f"{response_ratio:.2f} m/m"
                if np.isfinite(response_ratio)
                else "—"
            ),
        )

    with t3:
        st.metric(
            "Impacto estimado en San Nicolás",
            (
                f"{expected_sn_change:+.2f} m"
                if np.isfinite(expected_sn_change)
                else "—"
            ),
        )

    st.caption(
        "La traducción se aplica sobre la VARIACIÓN de nivel de Corrientes, "
        "no sobre su altura absoluta. El efecto se ubica temporalmente dentro "
        "del rango histórico de propagación mostrado arriba."
    )


# ============================================================
# GRÁFICO PRINCIPAL
# ============================================================

st.subheader(
    "Nivel de San Nicolás · observado y proyección"
)


visible_observed = sn_history[
    sn_history[
        "datetime"
    ]
    >= visible_start
].copy()


fig = go.Figure()


# ============================================================
# OBSERVADO
# ============================================================

fig.add_trace(
    go.Scatter(

        x=
            visible_observed[
                "datetime"
            ],

        y=
            visible_observed[
                "nivel"
            ],

        mode=
            "lines",

        name=
            "Observado",

        line=dict(
            width=3,
            color="#90caf9",
        ),
    )
)


# ============================================================
# HORIZONTES DEL PRONÓSTICO CENTRAL
# ============================================================

segments = [

    (
        1,
        15,
        "Pronóstico 1–15 días",
        "#2196f3",
    ),

    (
        16,
        30,
        "Proyección 16–30 días",
        "#f3b5b5",
    ),

    (
        31,
        45,
        "Escenario 31–45 días",
        "#ff7043",
    ),

    (
        46,
        60,
        "Tendencia 46–60 días",
        "#7ddc9a",
    ),
]


for start_day, end_day, label, line_color in segments:

    segment = forecast[
        (
            forecast[
                "horizon_day"
            ]
            >= start_day
        )
        &
        (
            forecast[
                "horizon_day"
            ]
            <= end_day
        )
    ]

    if segment.empty:
        continue

    fig.add_trace(
        go.Scatter(

            x=
                segment[
                    "datetime"
                ],

            y=
                segment[
                    "prediction"
                ],

            mode=
                "lines",

            name=
                label,

            line=dict(
                width=3,
                color=line_color,
            ),
        )
    )


# ============================================================
# INCERTIDUMBRE
# ============================================================

if (
    "lower" in forecast.columns
    and "upper" in forecast.columns
):
    upper = numeric(
        forecast["upper"]
    )
    lower = numeric(
        forecast["lower"]
    )

    valid_uncertainty = (
        upper.notna()
        & lower.notna()
    )

    if valid_uncertainty.any():
        fig.add_trace(
            go.Scatter(
                x=forecast["datetime"],
                y=upper,
                mode="lines",
                line=dict(
                    width=0,
                    color="rgba(148,163,184,0.0)",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast["datetime"],
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(148,163,184,0.14)",
                line=dict(
                    width=0,
                    color="rgba(148,163,184,0.0)",
                ),
                name="Incertidumbre",
                hoverinfo="skip",
            )
        )


# ============================================================
# ESCENARIO ADVERSO
# ============================================================

if (
    "scenario_adverse" in forecast.columns
    and numeric(
        forecast["scenario_adverse"]
    ).notna().any()
):
    fig.add_trace(
        go.Scatter(
            x=forecast["datetime"],
            y=forecast["scenario_adverse"],
            mode="lines",
            name="Escenario adverso",
            line=dict(
                color="rgba(250,204,21,0.25)",
                width=3.0,
                dash="dash",
            ),
        )
    )


# ============================================================
# PEOR ESCENARIO / EXTREMO HISTÓRICO
# ============================================================

if (
    "scenario_extreme" in forecast.columns
    and numeric(
        forecast["scenario_extreme"]
    ).notna().any()
):
    fig.add_trace(
        go.Scatter(
            x=forecast["datetime"],
            y=forecast["scenario_extreme"],
            mode="lines",
            name="Peor escenario · extremo histórico",
            line=dict(
                color="#ef4444",
                width=3.5,
                dash="dot",
            ),
        )
    )


y_range = dynamic_level_range(

    visible_observed[
        "nivel"
    ],

    forecast[
        "prediction"
    ],

    forecast.get(
        "scenario_adverse"
    ),

    forecast.get(
        "scenario_extreme"
    ),
)


fig.update_layout(

    height=520,

    hovermode=
        "x unified",

    margin=dict(
        l=20,
        r=20,
        t=25,
        b=20,
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    xaxis_title=
        "Fecha",

    yaxis_title=
        "Nivel [m]",
)


fig.update_yaxes(
    range=[0, 7],
    dtick=0.5,
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


st.caption(
    "Los escenarios adverso y extremo representan eventos "
    "históricos comparables y no una predicción meteorológica "
    "determinística a 60 días."
)


# ============================================================
# VALORES A 15 / 30 / 45 / 60
# ============================================================

st.subheader(
    "Horizontes del pronóstico"
)


horizon_columns = st.columns(
    4
)


for column, day in zip(
    horizon_columns,
    [
        15,
        30,
        45,
        60,
    ],
):

    match = forecast[
        forecast[
            "horizon_day"
        ]
        == day
    ]

    with column:

        if match.empty:

            st.metric(
                f"{day} días",
                "—",
            )

            continue

        row = match.iloc[
            0
        ]

        prediction = safe_float(
            row.get(
                "prediction"
            )
        )

        adverse = safe_float(
            row.get(
                "scenario_adverse"
            )
        )

        extreme = safe_float(
            row.get(
                "scenario_extreme"
            )
        )

        st.metric(

            f"{day} días",

            fmt_level(
                prediction
            ),

            (
                f"vs actual "
                f"{prediction - current_level:+.2f} m"
                if (
                    np.isfinite(
                        prediction
                    )
                    and np.isfinite(
                        current_level
                    )
                )
                else None
            ),
        )

        if np.isfinite(
            adverse
        ):

            st.caption(
                f"Adverso: {adverse:.2f} m"
            )

        if np.isfinite(
            extreme
        ):

            st.caption(
                f"Extremo: {extreme:.2f} m"
            )


# ============================================================
# ESCENARIOS
# ============================================================

st.subheader(
    "Escenarios históricos"
)


scenario_rows = []


for day in [
    15,
    30,
    45,
    60,
]:

    match = forecast[
        forecast[
            "horizon_day"
        ]
        == day
    ]

    if match.empty:
        continue

    row = match.iloc[
        0
    ]

    scenario_rows.append(
        {
            "Horizonte":
                f"{day} días",

            "Pronóstico central [m]":
                safe_float(
                    row.get(
                        "prediction"
                    )
                ),

            "Probable histórico [m]":
                safe_float(
                    row.get(
                        "scenario_probable"
                    )
                ),

            "Adverso [m]":
                safe_float(
                    row.get(
                        "scenario_adverse"
                    )
                ),

            "Extremo histórico [m]":
                safe_float(
                    row.get(
                        "scenario_extreme"
                    )
                ),
        }
    )


scenario_table = pd.DataFrame(
    scenario_rows
)


if not scenario_table.empty:

    st.dataframe(
        scenario_table.style.format(
            {
                "Pronóstico central [m]":
                    "{:.2f}",

                "Probable histórico [m]":
                    "{:.2f}",

                "Adverso [m]":
                    "{:.2f}",

                "Extremo histórico [m]":
                    "{:.2f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CORREDOR AGUAS ARRIBA
# ============================================================

st.subheader(
    "Estado del corredor aguas arriba"
)


corridor_rows = []


for station in STATIONS:

    level_col = LEVEL_COLUMNS[
        station
    ]

    flow_col = FLOW_COLUMNS[
        station
    ]

    rain_col = RAIN_COLUMNS[
        station
    ]

    if station == "San Nicolás":

        level_history = sn_history

        level_value = current_value(
            sn_history,
            "nivel",
        )

        d1 = delta_days(
            sn_history,
            "nivel",
            1,
        )

        d7 = delta_days(
            sn_history,
            "nivel",
            7,
        )

    else:

        level_history = upstream_history

        level_value = current_value(
            upstream_history,
            level_col,
        )

        d1 = delta_days(
            upstream_history,
            level_col,
            1,
        )

        d7 = delta_days(
            upstream_history,
            level_col,
            7,
        )

    flow_value = current_value(
        exog_history,
        flow_col,
    )

    rain_7 = np.nan

    if (
        rain_col
        in exog_history.columns
    ):

        rain_values = (
            numeric(
                exog_history[
                    rain_col
                ]
            )
            .fillna(0.0)
            .tail(7)
        )

        rain_7 = float(
            rain_values.sum()
        )

    if np.isfinite(
        d7
    ):

        if d7 > 0.05:

            state = "↑ Creciendo"

        elif d7 < -0.05:

            state = "↓ Bajando"

        else:

            state = "→ Estable"

    else:

        state = "Sin datos"

    corridor_rows.append(
        {
            "Estación":
                station,

            "Nivel [m]":
                level_value,

            "Δ 1 día [m]":
                d1,

            "Δ 7 días [m]":
                d7,

            "Estado":
                state,

            "Caudal [m³/s]":
                flow_value,

            "Lluvia 7d [mm]":
                rain_7,
        }
    )


corridor_table = pd.DataFrame(
    corridor_rows
)


st.dataframe(

    corridor_table.style.format(
        {
            "Nivel [m]":
                "{:.2f}",

            "Δ 1 día [m]":
                "{:+.2f}",

            "Δ 7 días [m]":
                "{:+.2f}",

            "Caudal [m³/s]":
                "{:,.0f}",

            "Lluvia 7d [mm]":
                "{:.1f}",
        },
        na_rep="—",
    ),

    use_container_width=True,

    hide_index=True,
)


# ============================================================
# CALIDAD DE LOS CAUDALES
# ============================================================

st.subheader(
    "Caudales · observado / reconstruido"
)


flow_quality_rows = []


for station in STATIONS:

    col = FLOW_COLUMNS[
        station
    ]

    source_col = (
        col
        + "_source"
    )

    quality_col = (
        col
        + "_quality"
    )

    value = current_value(
        exog_history,
        col,
    )

    source = "Sin datos"

    quality = np.nan

    if (
        not exog_history.empty
        and source_col
        in exog_history.columns
    ):

        valid = exog_history[
            [
                "datetime",
                col,
                source_col,
            ]
            + (
                [
                    quality_col
                ]
                if quality_col
                in exog_history.columns
                else []
            )
        ].copy()

        valid[
            col
        ] = numeric(
            valid[
                col
            ]
        )

        valid = valid.dropna(
            subset=[
                col
            ]
        )

        if not valid.empty:

            last = valid.iloc[
                -1
            ]

            source = str(
                last.get(
                    source_col,
                    "desconocido",
                )
            )

            if quality_col in valid.columns:

                quality = safe_float(
                    last.get(
                        quality_col
                    )
                )

    flow_quality_rows.append(
        {
            "Estación":
                station,

            "Caudal [m³/s]":
                value,

            "Origen":
                source,

            "Calidad":
                quality,
        }
    )


flow_quality_table = pd.DataFrame(
    flow_quality_rows
)


st.dataframe(

    flow_quality_table.style.format(
        {
            "Caudal [m³/s]":
                "{:,.0f}",

            "Calidad":
                "{:.0%}",
        },
        na_rep="—",
    ),

    use_container_width=True,

    hide_index=True,
)


st.caption(
    "Calidad 100% = observación directa. Valores interpolados "
    "o reconstruidos tienen menor calidad y reciben menor peso "
    "en el modelo."
)


# ============================================================
# GRÁFICO CAUDAL V11.13
# ============================================================

st.subheader(
    "Caudal histórico y proyección"
)


available_flow_stations = []


for station in STATIONS:

    col = FLOW_COLUMNS[
        station
    ]

    if (
        col in exog_history.columns
        and numeric(
            exog_history[
                col
            ]
        )
        .notna()
        .any()
    ):

        available_flow_stations.append(
            station
        )


if available_flow_stations:

    default_flow_index = 0

    if (
        flow_station
        in available_flow_stations
    ):

        default_flow_index = (
            available_flow_stations.index(
                flow_station
            )
        )

    selected_flow_station = st.selectbox(
        "Estación de caudal",
        available_flow_stations,
        index=default_flow_index,
        key="selected_flow_station",
    )

    selected_flow_col = FLOW_COLUMNS[
        selected_flow_station
    ]

    selected_source_col = (
        selected_flow_col
        + "_source"
    )

    selected_quality_col = (
        selected_flow_col
        + "_quality"
    )

    flow_hist = exog_history[
        exog_history[
            "datetime"
        ]
        >= visible_start
    ].copy()

    flow_hist[
        selected_flow_col
    ] = numeric(
        flow_hist[
            selected_flow_col
        ]
    )

    # --------------------------------------------------------
    # RESUMEN DEL CAUDAL SELECCIONADO
    # --------------------------------------------------------

    q_current = current_value(
        exog_history,
        selected_flow_col,
    )

    q_delta_7 = delta_days(
        exog_history,
        selected_flow_col,
        7,
    )

    q_future = exog_future.copy()

    if selected_flow_col in q_future.columns:
        q_future[selected_flow_col] = numeric(
            q_future[selected_flow_col]
        )

    q15 = np.nan
    q30 = np.nan
    q60 = np.nan
    qmax15 = np.nan

    if (
        not q_future.empty
        and selected_flow_col
        in q_future.columns
    ):

        future_values = q_future[
            selected_flow_col
        ]

        if len(future_values) >= 15:
            q15 = safe_float(
                future_values.iloc[14]
            )
            qmax15 = safe_float(
                future_values.iloc[:15].max()
            )

        if len(future_values) >= 30:
            q30 = safe_float(
                future_values.iloc[29]
            )

        if len(future_values) >= 60:
            q60 = safe_float(
                future_values.iloc[59]
            )

    last_source = "Sin datos"
    last_quality = np.nan

    if (
        selected_source_col
        in exog_history.columns
    ):

        source_rows = exog_history[
            [
                selected_flow_col,
                selected_source_col,
            ]
            + (
                [selected_quality_col]
                if selected_quality_col
                in exog_history.columns
                else []
            )
        ].copy()

        source_rows[
            selected_flow_col
        ] = numeric(
            source_rows[
                selected_flow_col
            ]
        )

        source_rows = source_rows.dropna(
            subset=[selected_flow_col]
        )

        if not source_rows.empty:
            last_row = source_rows.iloc[-1]
            last_source = str(
                last_row.get(
                    selected_source_col,
                    "desconocido",
                )
            )

            if selected_quality_col in source_rows.columns:
                last_quality = safe_float(
                    last_row.get(
                        selected_quality_col
                    )
                )

    fm1, fm2, fm3, fm4, fm5 = st.columns(5)

    with fm1:
        st.metric(
            "Caudal actual",
            fmt_flow(q_current),
            (
                fmt_flow(q_delta_7)
                if np.isfinite(q_delta_7)
                else None
            ),
        )
        st.caption(
            f"Origen: {last_source}"
            + (
                f" · calidad {last_quality:.0%}"
                if np.isfinite(last_quality)
                else ""
            )
        )

    with fm2:
        st.metric(
            "Máximo 15 días",
            fmt_flow(qmax15),
        )

    with fm3:
        st.metric(
            "Día 15",
            fmt_flow(q15),
        )

    with fm4:
        st.metric(
            "Día 30",
            fmt_flow(q30),
        )

    with fm5:
        st.metric(
            "Día 60 · tendencia",
            fmt_flow(q60),
        )

    flow_fig = go.Figure()

    # --------------------------------------------------------
    # HISTÓRICO OBSERVADO VS RECONSTRUIDO
    # --------------------------------------------------------

    if selected_source_col in flow_hist.columns:

        source_text = (
            flow_hist[
                selected_source_col
            ]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        observed_mask = (
            source_text == "observado"
        )

        reconstructed_mask = (
            flow_hist[
                selected_flow_col
            ].notna()
            & ~observed_mask
        )

        observed_y = flow_hist[
            selected_flow_col
        ].where(observed_mask)

        reconstructed_y = flow_hist[
            selected_flow_col
        ].where(reconstructed_mask)

        if observed_y.notna().any():
            flow_fig.add_trace(
                go.Scatter(
                    x=flow_hist["datetime"],
                    y=observed_y,
                    mode="lines",
                    name="Observado",
                    line=dict(width=2.6),
                    connectgaps=False,
                )
            )

        if reconstructed_y.notna().any():
            flow_fig.add_trace(
                go.Scatter(
                    x=flow_hist["datetime"],
                    y=reconstructed_y,
                    mode="lines",
                    name="Reconstruido / interpolado",
                    line=dict(
                        width=2.0,
                        dash="dot",
                    ),
                    connectgaps=False,
                )
            )

    else:

        flow_fig.add_trace(
            go.Scatter(
                x=flow_hist["datetime"],
                y=flow_hist[selected_flow_col],
                mode="lines",
                name="Histórico",
            )
        )

    # --------------------------------------------------------
    # FUTURO POR HORIZONTE
    # --------------------------------------------------------

    if (
        selected_flow_col
        in exog_future.columns
        and not exog_future.empty
    ):

        future_plot = exog_future.copy()
        future_plot[selected_flow_col] = numeric(
            future_plot[selected_flow_col]
        )

        if "flow_horizon_day" not in future_plot.columns:
            future_plot["flow_horizon_day"] = np.arange(
                1,
                len(future_plot) + 1,
            )

        flow_segments = [
            (
                1,
                15,
                "Pronóstico 1–15 días",
                "solid",
                3.0,
            ),
            (
                16,
                30,
                "Proyección 16–30 días",
                "dash",
                2.6,
            ),
            (
                31,
                60,
                "Tendencia 31–60 días",
                "dot",
                2.3,
            ),
        ]

        for (
            start_day,
            end_day,
            label,
            dash,
            width,
        ) in flow_segments:

            segment = future_plot[
                (
                    future_plot[
                        "flow_horizon_day"
                    ] >= start_day
                )
                &
                (
                    future_plot[
                        "flow_horizon_day"
                    ] <= end_day
                )
            ].copy()

            if segment.empty:
                continue

            flow_fig.add_trace(
                go.Scatter(
                    x=segment["datetime"],
                    y=segment[selected_flow_col],
                    mode="lines",
                    name=label,
                    line=dict(
                        width=width,
                        dash=dash,
                    ),
                )
            )

    flow_fig.update_layout(
        height=430,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        yaxis_title="Caudal [m³/s]",
        xaxis_title="Fecha",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        flow_fig,
        use_container_width=True,
    )

    st.caption(
        "V11.13: 1–15 días = pronóstico hidrológico; "
        "16–30 días = proyección; 31–60 días = tendencia. "
        "El tramo extendido no debe interpretarse como un pronóstico "
        "meteorológico determinístico."
    )

else:

    st.info(
        "No hay series de caudal disponibles."
    )


# ============================================================
# LLUVIAS
# ============================================================

st.subheader(
    "Lluvia por punto del corredor"
)


available_rain_stations = [
    station
    for station in STATIONS
    if (
        RAIN_COLUMNS[
            station
        ]
        in exog_history.columns
    )
]


if available_rain_stations:

    selected_rain_station = st.selectbox(

        "Estación de lluvia",

        available_rain_stations,

        index=0,

        key=
            "selected_rain_station",
    )

    rain_col = RAIN_COLUMNS[
        selected_rain_station
    ]

    rain_hist = exog_history[
        exog_history[
            "datetime"
        ]
        >= visible_start
    ].copy()

    rain_fig = go.Figure()

    rain_fig.add_trace(
        go.Bar(

            x=
                rain_hist[
                    "datetime"
                ],

            y=
                rain_hist[
                    rain_col
                ],

            name=
                "Histórico",
        )
    )

    if (
        rain_col
        in exog_future.columns
    ):

        rain_future_plot = exog_future.copy()

        if (
            "rain_forecast_available"
            in rain_future_plot.columns
        ):
            rain_future_plot = rain_future_plot[
                rain_future_plot[
                    "rain_forecast_available"
                ].fillna(False)
            ]

        else:
            rain_future_plot = rain_future_plot.head(16)

        rain_fig.add_trace(
            go.Bar(

                x=
                    rain_future_plot[
                        "datetime"
                    ],

                y=
                    rain_future_plot[
                        rain_col
                    ],

                name=
                    "Pronóstico meteorológico disponible",
        )
    )

    rain_fig.update_layout(

        height=380,

        hovermode=
            "x unified",

        barmode=
            "overlay",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),

        yaxis_title=
            "Precipitación [mm]",

        xaxis_title=
            "Fecha",
    )

    st.plotly_chart(
        rain_fig,
        use_container_width=True,
    )


# ============================================================
# PROPAGACIÓN CORRIENTES → SAN NICOLÁS
# ============================================================

st.subheader(
    "Propagación Corrientes → San Nicolás"
)


prop1, prop2, prop3, prop4 = st.columns(
    4
)


with prop1:

    st.metric(
        "Retardo estimado",
        (
            f"{delay} días"
            if delay is not None
            else "—"
        ),
    )


with prop2:

    st.metric(
        "Rango probable",
        (
            f"{delay_min}–{delay_max} días"
            if (
                delay_min is not None
                and delay_max is not None
            )
            else "—"
        ),
    )


with prop3:

    st.metric(
        "Correlación",
        (
            f"{correlation:.2f}"
            if np.isfinite(
                correlation
            )
            else "—"
        ),
    )


with prop4:

    similar_count = (
        hydro_estimate.get(
            "similar_event_count",
            0,
        )
    )

    st.metric(
        "Eventos similares",
        str(
            similar_count
        ),
    )


# ============================================================
# CORRIENTES VS SAN NICOLÁS NORMALIZADO
# ============================================================

if (
    "nivel_corrientes"
    in upstream_history.columns
):

    corrientes = upstream_history[
        [
            "datetime",
            "nivel_corrientes",
        ]
    ].copy()

    sn_compare = sn_history[
        [
            "datetime",
            "nivel",
        ]
    ].copy()

    compare = corrientes.merge(
        sn_compare,
        on="datetime",
        how="inner",
    )

    compare[
        "nivel_corrientes"
    ] = numeric(
        compare[
            "nivel_corrientes"
        ]
    )

    compare[
        "nivel"
    ] = numeric(
        compare[
            "nivel"
        ]
    )

    compare = compare.dropna()

    if len(compare) >= 30:

        corr_med = (
            compare[
                "nivel_corrientes"
            ]
            .rolling(
                30,
                min_periods=10,
            )
            .median()
        )

        sn_med = (
            compare[
                "nivel"
            ]
            .rolling(
                30,
                min_periods=10,
            )
            .median()
        )

        corr_std = (
            compare[
                "nivel_corrientes"
            ]
            .rolling(
                60,
                min_periods=20,
            )
            .std()
        )

        sn_std = (
            compare[
                "nivel"
            ]
            .rolling(
                60,
                min_periods=20,
            )
            .std()
        )

        compare[
            "corrientes_anom"
        ] = (
            (
                compare[
                    "nivel_corrientes"
                ]
                - corr_med
            )
            /
            corr_std.replace(
                0,
                np.nan,
            )
        )

        compare[
            "sn_anom"
        ] = (
            (
                compare[
                    "nivel"
                ]
                - sn_med
            )
            /
            sn_std.replace(
                0,
                np.nan,
            )
        )

        compare_visible = compare[
            compare[
                "datetime"
            ]
            >= (
                base_ts
                - pd.Timedelta(
                    days=730
                )
            )
        ]

        comparison_fig = go.Figure()

        comparison_fig.add_trace(
            go.Scatter(

                x=
                    compare_visible[
                        "datetime"
                    ],

                y=
                    compare_visible[
                        "corrientes_anom"
                    ],

                mode=
                    "lines",

                name=
                    "Corrientes · anomalía",
            )
        )

        comparison_fig.add_trace(
            go.Scatter(

                x=
                    compare_visible[
                        "datetime"
                    ],

                y=
                    compare_visible[
                        "sn_anom"
                    ],

                mode=
                    "lines",

                name=
                    "San Nicolás · anomalía",
            )
        )

        comparison_fig.update_layout(

            height=400,

            hovermode=
                "x unified",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),

            yaxis_title=
                "Anomalía normalizada",

            xaxis_title=
                "Fecha",
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
        )

        st.caption(
            "La comparación está normalizada porque Corrientes "
            "y San Nicolás utilizan ceros hidrométricos diferentes."
        )


# ============================================================
# EVENTOS HISTÓRICOS SIMILARES
# ============================================================

st.subheader(
    "Eventos históricos comparables"
)


similar_events = hydrology.get(
    "similar_events"
)


if (
    isinstance(
        similar_events,
        pd.DataFrame,
    )
    and not similar_events.empty
):

    event_columns = [
        col
        for col in [
            "start_date",
            "peak_date",
            "start_level_sn",
            "peak_level_sn",
            "rise_sn",
            "rise_days",
            "similarity_distance",
            "similarity_weight",
        ]
        if col
        in similar_events.columns
    ]

    events_display = (
        similar_events[
            event_columns
        ]
        .head(15)
        .copy()
    )

    events_display = events_display.rename(
        columns={
            "start_date":
                "Inicio",

            "peak_date":
                "Pico",

            "start_level_sn":
                "Nivel inicial SN",

            "peak_level_sn":
                "Pico SN",

            "rise_sn":
                "Crecimiento",

            "rise_days":
                "Duración",

            "similarity_distance":
                "Distancia",

            "similarity_weight":
                "Peso",
        }
    )

    st.dataframe(
        events_display,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Todavía no se detectaron suficientes eventos históricos comparables."
    )


# ============================================================
# COMPARATIVA AÑO CONTRA AÑO
# ============================================================

st.subheader(
    "Comparativa año contra año"
)


years_available = sorted(
    sn_history[
        "datetime"
    ]
    .dt
    .year
    .dropna()
    .astype(int)
    .unique()
    .tolist(),
    reverse=True,
)


if years_available:

    current_year = int(
        base_ts.year
    )

    default_years = [
        year
        for year in years_available
        if year
        >= current_year - 4
    ][:5]

    selected_years = st.multiselect(

        "Años a comparar",

        options=
            years_available,

        default=
            default_years,
    )

    yc1, yc2 = st.columns(
        2
    )

    with yc1:

        comparison_start = st.date_input(

            "Inicio de ventana estacional",

            value=
                date(
                    2026,
                    1,
                    1,
                ),

            key=
                "comparison_start",
        )

    with yc2:

        comparison_end = st.date_input(

            "Fin de ventana estacional",

            value=
                date(
                    2026,
                    12,
                    31,
                ),

            key=
                "comparison_end",
        )


    if selected_years:

        yoy_fig = go.Figure()

        start_month = (
            comparison_start.month
        )

        start_day = (
            comparison_start.day
        )

        end_month = (
            comparison_end.month
        )

        end_day = (
            comparison_end.day
        )

        for year in selected_years:

            x = sn_history[
                sn_history[
                    "datetime"
                ].dt.year
                == year
            ].copy()

            if x.empty:
                continue

            md = (
                x[
                    "datetime"
                ].dt.month
                * 100
                +
                x[
                    "datetime"
                ].dt.day
            )

            start_md = (
                start_month
                * 100
                +
                start_day
            )

            end_md = (
                end_month
                * 100
                +
                end_day
            )

            if start_md <= end_md:

                mask = (
                    (md >= start_md)
                    &
                    (md <= end_md)
                )

            else:

                mask = (
                    (md >= start_md)
                    |
                    (md <= end_md)
                )

            x = x[
                mask
            ].copy()

            if x.empty:
                continue

            x[
                "comparison_day"
            ] = np.arange(
                1,
                len(x) + 1,
            )

            yoy_fig.add_trace(
                go.Scatter(

                    x=
                        x[
                            "comparison_day"
                        ],

                    y=
                        x[
                            "nivel"
                        ],

                    mode=
                        "lines",

                    name=
                        str(
                            year
                        ),
                )
            )

        yoy_fig.update_layout(

            height=430,

            hovermode=
                "x unified",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),

            xaxis_title=
                "Días desde inicio de ventana",

            yaxis_title=
                "Nivel San Nicolás [m]",
        )

        st.plotly_chart(
            yoy_fig,
            use_container_width=True,
        )


# ============================================================
# CORRIENTES VS SAN NICOLÁS · PROPAGACIÓN AÑO POR AÑO
# ============================================================

st.subheader(
    "Corrientes vs San Nicolás · propagación año por año"
)

corrientes_yearly = hydrology.get(
    "corrientes_yearly",
    pd.DataFrame(),
)

corrientes_robust = hydrology.get(
    "corrientes_robust",
    {},
)

hydro_dataset = hydrology.get(
    "dataset",
    pd.DataFrame(),
)

if (
    isinstance(corrientes_yearly, pd.DataFrame)
    and not corrientes_yearly.empty
):
    annual_display = corrientes_yearly.copy()

    annual_display = annual_display.sort_values(
        "year",
        ascending=False,
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric(
            "Demora robusta",
            (
                f"{corrientes_robust.get('delay_days')} días"
                if corrientes_robust.get("delay_days") is not None
                else "—"
            ),
        )

    with a2:
        st.metric(
            "Rango histórico",
            (
                f"{corrientes_robust.get('delay_min')}–"
                f"{corrientes_robust.get('delay_max')} días"
                if (
                    corrientes_robust.get("delay_min") is not None
                    and corrientes_robust.get("delay_max") is not None
                )
                else "—"
            ),
        )

    with a3:
        robust_corr = safe_float(
            corrientes_robust.get(
                "correlation"
            )
        )

        st.metric(
            "Correlación mediana",
            (
                f"{robust_corr:.2f}"
                if np.isfinite(robust_corr)
                else "—"
            ),
        )

    with a4:
        robust_response = safe_float(
            corrientes_robust.get(
                "response_m_per_m"
            )
        )

        st.metric(
            "Respuesta SN / Corrientes",
            (
                f"{robust_response:.2f} m/m"
                if np.isfinite(robust_response)
                else "—"
            ),
        )

    years_corr = (
        annual_display[
            "year"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    selected_corr_year = st.selectbox(
        "Año para comparar Corrientes con San Nicolás",
        options=years_corr,
        index=0,
        key="corrientes_sn_year",
    )

    selected_row = annual_display[
        annual_display["year"]
        == selected_corr_year
    ]

    selected_lag = (
        int(
            selected_row.iloc[0][
                "lag_days"
            ]
        )
        if not selected_row.empty
        else int(
            corrientes_robust.get(
                "delay_days",
                20,
            )
        )
    )

    if (
        isinstance(hydro_dataset, pd.DataFrame)
        and not hydro_dataset.empty
        and "datetime" in hydro_dataset.columns
        and "nivel_corrientes" in hydro_dataset.columns
        and "nivel_san_nicolas" in hydro_dataset.columns
    ):
        annual = hydro_dataset[
            pd.to_datetime(
                hydro_dataset["datetime"],
                errors="coerce",
            ).dt.year
            == selected_corr_year
        ].copy()

        annual["datetime"] = pd.to_datetime(
            annual["datetime"],
            errors="coerce",
        )

        annual["corrientes_change"] = normalized_change(
            annual["nivel_corrientes"]
        )

        annual["sn_change"] = normalized_change(
            annual["nivel_san_nicolas"]
        )

        annual["corrientes_propagation_date"] = (
            annual["datetime"]
            + pd.to_timedelta(
                selected_lag,
                unit="D",
            )
        )

        corr_year_fig = go.Figure()

        corr_year_fig.add_trace(
            go.Scatter(
                x=annual[
                    "corrientes_propagation_date"
                ],
                y=annual[
                    "corrientes_change"
                ],
                mode="lines",
                name=(
                    f"Corrientes trasladado +{selected_lag} días"
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>Δ Corrientes: %{y:+.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        corr_year_fig.add_trace(
            go.Scatter(
                x=annual["datetime"],
                y=annual["sn_change"],
                mode="lines",
                name="San Nicolás",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>Δ San Nicolás: %{y:+.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        corr_year_fig.add_hline(
            y=0,
            line_width=1,
            line_dash="dot",
        )

        corr_year_fig.update_layout(
            height=430,
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
            xaxis_title="Fecha",
            yaxis_title="Cambio respecto del inicio del año [m]",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        corr_year_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            corr_year_fig,
            use_container_width=True,
        )

        st.caption(
            "Corrientes se desplaza hacia adelante por el retardo óptimo "
            "calculado para ese año. Las curvas representan cambios respecto "
            "del inicio del año, no alturas absolutas, porque cada estación "
            "tiene un cero hidrométrico diferente."
        )

        # ====================================================
        # ALTURAS ABSOLUTAS · MISMA FECHA · AÑO A AÑO
        # ====================================================

        st.markdown(
            "#### Alturas de río · Corrientes vs San Nicolás · misma fecha"
        )

        # Esta comparación NO aplica demora hidrológica.
        # Cada punto de Corrientes se enfrenta con San Nicolás
        # en la misma fecha calendario (lectura vs lectura).
        comparison_source = hydro_dataset[
            [
                "datetime",
                "nivel_corrientes",
                "nivel_san_nicolas",
            ]
        ].copy()

        comparison_source["datetime"] = pd.to_datetime(
            comparison_source["datetime"],
            errors="coerce",
        )
        comparison_source["nivel_corrientes"] = pd.to_numeric(
            comparison_source["nivel_corrientes"],
            errors="coerce",
        )
        comparison_source["nivel_san_nicolas"] = pd.to_numeric(
            comparison_source["nivel_san_nicolas"],
            errors="coerce",
        )

        comparison_source = comparison_source.dropna(
            subset=["datetime"]
        )

        years_same_day = sorted(
            comparison_source.loc[
                comparison_source["nivel_corrientes"].notna()
                & comparison_source["nivel_san_nicolas"].notna(),
                "datetime",
            ].dt.year.dropna().astype(int).unique().tolist(),
            reverse=True,
        )

        if years_same_day:
            default_same_day_index = (
                years_same_day.index(selected_corr_year)
                if selected_corr_year in years_same_day
                else 0
            )

            selected_same_day_year = st.selectbox(
                "Año para comparar alturas lectura contra lectura",
                options=years_same_day,
                index=default_same_day_index,
                key="corrientes_sn_same_day_year",
            )

            absolute = comparison_source[
                comparison_source["datetime"].dt.year
                == selected_same_day_year
            ].copy()

            # Conservamos únicamente las fechas en las que existe
            # lectura de ambas estaciones. No se desplaza ninguna serie.
            paired_abs = absolute.dropna(
                subset=[
                    "nivel_corrientes",
                    "nivel_san_nicolas",
                ]
            ).copy()

            absolute_fig = go.Figure()

            absolute_fig.add_trace(
                go.Scatter(
                    x=paired_abs["datetime"],
                    y=paired_abs["nivel_corrientes"],
                    mode="lines",
                    name="Corrientes · altura misma fecha",
                    yaxis="y",
                    connectgaps=False,
                    hovertemplate=(
                        "%{x|%d/%m/%Y}"
                        "<br>Corrientes: %{y:.2f} m"
                        "<extra></extra>"
                    ),
                )
            )

            absolute_fig.add_trace(
                go.Scatter(
                    x=paired_abs["datetime"],
                    y=paired_abs["nivel_san_nicolas"],
                    mode="lines",
                    name="San Nicolás · altura misma fecha",
                    yaxis="y2",
                    connectgaps=False,
                    hovertemplate=(
                        "%{x|%d/%m/%Y}"
                        "<br>San Nicolás: %{y:.2f} m"
                        "<extra></extra>"
                    ),
                )
            )

            absolute_fig.update_layout(
                height=460,
                hovermode="x unified",
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20,
                ),
                xaxis=dict(
                    title="Fecha",
                    tickformat="%d/%m",
                ),
                yaxis=dict(
                    title="Altura Corrientes [m]",
                    side="left",
                    rangemode="tozero",
                ),
                yaxis2=dict(
                    title="Altura San Nicolás [m]",
                    side="right",
                    overlaying="y",
                    rangemode="tozero",
                    showgrid=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(
                absolute_fig,
                use_container_width=True,
            )

            if not paired_abs.empty:
                h1, h2, h3, h4, h5 = st.columns(5)

                with h1:
                    st.metric(
                        f"Corrientes · inicio {selected_same_day_year}",
                        f"{float(paired_abs['nivel_corrientes'].iloc[0]):.2f} m",
                    )

                with h2:
                    st.metric(
                        f"San Nicolás · inicio {selected_same_day_year}",
                        f"{float(paired_abs['nivel_san_nicolas'].iloc[0]):.2f} m",
                    )

                with h3:
                    st.metric(
                        "Máximo Corrientes",
                        f"{float(paired_abs['nivel_corrientes'].max()):.2f} m",
                    )

                with h4:
                    st.metric(
                        "Máximo San Nicolás",
                        f"{float(paired_abs['nivel_san_nicolas'].max()):.2f} m",
                    )

                with h5:
                    st.metric(
                        "Lecturas coincidentes",
                        f"{len(paired_abs):,}".replace(",", "."),
                    )

                same_day_corr = paired_abs[
                    "nivel_corrientes"
                ].corr(
                    paired_abs["nivel_san_nicolas"]
                )

                if pd.notna(same_day_corr):
                    st.caption(
                        f"Correlación de alturas en la misma fecha para "
                        f"{selected_same_day_year}: {float(same_day_corr):.3f}. "
                        "Este valor es descriptivo y no incorpora demora de propagación."
                    )

            st.caption(
                "Comparación lectura contra lectura: Corrientes y San Nicolás "
                "se muestran exactamente en la misma fecha, sin trasladar Corrientes "
                "7 días ni aplicar ningún otro retardo. El selector permite revisar "
                "la relación año por año. Se mantienen dos ejes verticales porque "
                "cada estación posee un cero hidrométrico diferente."
            )
        else:
            st.info(
                "No hay años con suficientes lecturas coincidentes de Corrientes "
                "y San Nicolás para mostrar la comparación de alturas en la misma fecha."
            )

    table_display = annual_display.rename(
        columns={
            "year": "Año",
            "lag_days": "Demora [días]",
            "correlation": "Correlación",
            "response_m_per_m": "Respuesta SN/Corrientes [m/m]",
            "overlap": "Pares",
        }
    )

    for col in [
        "Correlación",
        "Respuesta SN/Corrientes [m/m]",
    ]:
        if col in table_display.columns:
            table_display[col] = (
                pd.to_numeric(
                    table_display[col],
                    errors="coerce",
                )
                .round(3)
            )

    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Todavía no hay suficientes años con datos coincidentes "
        "de Corrientes y San Nicolás para construir la comparación anual."
    )


# ============================================================
# COBERTURA DEL MODELO
# ============================================================

st.subheader(
    "Cobertura de variables del modelo"
)


coverage_rows = [

    {
        "Grupo":
            "Nivel San Nicolás",

        "Activo":
            True,

        "Variables":
            1,
    },

    {
        "Grupo":
            "Niveles aguas arriba",

        "Activo":
            bool(
                models.get(
                    "uses_upstream",
                    False,
                )
            ),

        "Variables":
            int(
                models.get(
                    "upstream_feature_count",
                    0,
                )
            ),
    },

    {
        "Grupo":
            "Caudales",

        "Activo":
            bool(
                models.get(
                    "uses_caudal",
                    False,
                )
            ),

        "Variables":
            int(
                models.get(
                    "flow_feature_count",
                    0,
                )
            ),
    },

    {
        "Grupo":
            "Lluvias",

        "Activo":
            bool(
                models.get(
                    "uses_rain",
                    False,
                )
            ),

        "Variables":
            int(
                models.get(
                    "rain_feature_count",
                    0,
                )
            ),
    },

    {
        "Grupo":
            "Hidrología / propagación",

        "Activo":
            bool(
                models.get(
                    "uses_hydrology",
                    False,
                )
            ),

        "Variables":
            int(
                models.get(
                    "hydrology_feature_count",
                    0,
                )
            ),
    },
]


coverage_table = pd.DataFrame(
    coverage_rows
)


coverage_table[
    "Estado"
] = coverage_table[
    "Activo"
].map(
    {
        True:
            "✅",

        False:
            "❌",
    }
)


st.dataframe(

    coverage_table[
        [
            "Estado",
            "Grupo",
            "Variables",
        ]
    ],

    use_container_width=True,

    hide_index=True,
)


# ============================================================
# MÉTRICAS DEL MODELO
# ============================================================

st.subheader(
    "Modelo y entrenamiento"
)


m1, m2, m3, m4 = st.columns(
    4
)


with m1:

    st.metric(
        "RMSE diario",
        (
            f"{safe_float(metrics.get('rmse')):.3f} m"
            if np.isfinite(
                safe_float(
                    metrics.get(
                        "rmse"
                    )
                )
            )
            else "—"
        ),
    )


with m2:

    st.metric(
        "MAE diario",
        (
            f"{safe_float(metrics.get('mae')):.3f} m"
            if np.isfinite(
                safe_float(
                    metrics.get(
                        "mae"
                    )
                )
            )
            else "—"
        ),
    )


with m3:

    st.metric(
        "Filas entrenamiento",
        str(
            models.get(
                "training_rows",
                0,
            )
        ),
    )


with m4:

    st.metric(
        "Variables",
        str(
            models.get(
                "feature_count",
                0,
            )
        ),
    )


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

importance = models.get(
    "importance"
)


if (
    isinstance(
        importance,
        pd.DataFrame,
    )
    and not importance.empty
):

    with st.expander(
        "Importancia de variables"
    ):

        importance_display = (
            importance
            .head(30)
            .copy()
        )

        importance_fig = go.Figure()

        importance_fig.add_trace(
            go.Bar(

                x=
                    importance_display[
                        "importance"
                    ],

                y=
                    importance_display[
                        "feature"
                    ],

                orientation=
                    "h",
            )
        )

        importance_fig.update_layout(

            height=650,

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),

            yaxis=dict(
                autorange=
                    "reversed"
            ),

            xaxis_title=
                "Importancia",

            yaxis_title=
                "",
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True,
        )


# ============================================================
# METODOLOGÍA
# ============================================================

with st.expander(
    "Metodología del pronóstico V11.12"
):

    st.markdown(
        """
        **1–15 días**

        Mayor peso del modelo estadístico, niveles aguas arriba,
        caudales y pronóstico meteorológico disponible.

        **16–30 días**

        Se incrementa el peso de la propagación hidrológica,
        tendencia de caudales y eventos históricos similares.

        **31–45 días**

        Se reduce el carácter determinístico. Los escenarios
        históricos tienen mayor influencia.

        **46–60 días**

        Se interpreta como tendencia hidrológica y escenario,
        no como pronóstico meteorológico exacto.

        **Caudales faltantes**

        Los valores reconstruidos se mantienen diferenciados de
        los observados mediante columnas de origen y calidad. El
        modelo reduce su influencia cuando la calidad es menor.

        **Propagación Corrientes → San Nicolás**

        Se busca el retardo histórico entre 1 y 60 días usando
        variaciones de nivel normalizadas y se resume año por año
        para evitar que un período aislado determine toda la demora.

        **Escenario extremo histórico**

        Se basa en eventos concurrentes históricos comparables.
        No suma arbitrariamente el máximo de lluvia de un año con
        el máximo de caudal de otro año.
        """
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

with st.expander(
    "Diagnóstico técnico"
):

    st.write(
        "Versión de app:",
        APP_VERSION,
    )

    st.write(
        "Última actualización:",
        st.session_state.get(
            "last_update"
        ),
    )

    st.write(
        "Estaciones aguas arriba disponibles:",
        upstream_meta.get(
            "available_stations",
            [],
        )
        if isinstance(
            upstream_meta,
            dict,
        )
        else [],
    )

    st.write(
        "Estaciones con caudal observado:",
        exog_meta.get(
            "flow_observed_stations",
            [],
        )
        if isinstance(
            exog_meta,
            dict,
        )
        else [],
    )

    st.write(
        "Estaciones con caudal disponible:",
        exog_meta.get(
            "flow_available_stations",
            [],
        )
        if isinstance(
            exog_meta,
            dict,
        )
        else [],
    )

    st.write(
        "Caudal principal:",
        flow_station,
    )

    st.write(
        "Retardos a San Nicolás:"
    )

    lag_to_sn = hydrology.get(
        "lag_to_sn"
    )

    if (
        isinstance(
            lag_to_sn,
            pd.DataFrame,
        )
        and not lag_to_sn.empty
    ):

        st.dataframe(
            lag_to_sn,
            use_container_width=True,
            hide_index=True,
        )

    st.write(
        "Retardos por tramo:"
    )

    corridor_lags = hydrology.get(
        "corridor_lags"
    )

    if (
        isinstance(
            corridor_lags,
            pd.DataFrame,
        )
        and not corridor_lags.empty
    ):

        st.dataframe(
            corridor_lags,
            use_container_width=True,
            hide_index=True,
        )

    st.write(
        "Presión hidrológica:"
    )

    st.json(
        hydrology.get(
            "pressure",
            {}
        )
    )

    st.write(
        "Modelo:"
    )

    st.json(
        {
            "version":
                models.get(
                    "version"
                ),

            "training_rows":
                models.get(
                    "training_rows"
                ),

            "feature_count":
                models.get(
                    "feature_count"
                ),

            "upstream_features":
                models.get(
                    "upstream_feature_count"
                ),

            "flow_features":
                models.get(
                    "flow_feature_count"
                ),

            "rain_features":
                models.get(
                    "rain_feature_count"
                ),

            "hydrology_features":
                models.get(
                    "hydrology_feature_count"
                ),
        }
    )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "PARANÁ · SAN NICOLÁS · "
    f"{APP_VERSION} · "
    "Modelo experimental de apoyo al análisis hidrológico. "
    "Los escenarios de 30–60 días representan proyecciones "
    "probabilísticas y no sustituyen avisos oficiales de INA."
)
