# ============================================================
# PARANÁ · SAN NICOLÁS
# app.py
#
# Monitoreo hidrológico y diagnóstico INA
# ============================================================

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ina import (
    STATIONS,
    diagnostic,
    forecast_meta,
    observed,
)


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1 {
        margin-bottom: 0rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -5px;
        margin-bottom: 20px;
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 10px;
        background-color: rgba(120,120,120,0.08);
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(120,120,120,0.05);
        border: 1px solid rgba(120,120,120,0.15);
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TÍTULO
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.markdown(
    """
    <div class="subtitle">
        Monitoreo hidrológico · datos INA · análisis experimental
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FECHAS POR DEFECTO
# ============================================================

today = date.today()

default_end = today

default_start = today - timedelta(days=30)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Consulta")

    start_date = st.date_input(
        "Desde",
        value=default_start,
        format="DD/MM/YYYY",
    )

    end_date = st.date_input(
        "Hasta",
        value=default_end,
        format="DD/MM/YYYY",
    )

    st.divider()

    update_button = st.button(
        "🔄 Actualizar INA",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.caption(
        "Fuente de datos observados: "
        "Instituto Nacional del Agua (INA)."
    )


# ============================================================
# VALIDACIÓN DE FECHAS
# ============================================================

if start_date > end_date:

    st.error(
        "La fecha Desde no puede ser posterior "
        "a la fecha Hasta."
    )

    st.stop()


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

if "ina_df" not in st.session_state:
    st.session_state.ina_df = pd.DataFrame()

if "ina_error" not in st.session_state:
    st.session_state.ina_error = None

if "diagnostic" not in st.session_state:
    st.session_state.diagnostic = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None


# ============================================================
# CONSULTAR DATOS
# ============================================================

def load_ina_data(start, end):

    df, error = observed(
        start=start,
        end=end,
    )

    diag = diagnostic(
        start=start,
        end=end,
    )

    return df, error, diag


# ============================================================
# CONSULTA AUTOMÁTICA INICIAL
# ============================================================

need_initial_load = (
    st.session_state.last_query is None
)


if update_button or need_initial_load:

    with st.spinner(
        "Consultando datos del INA..."
    ):

        df, error, diag = load_ina_data(
            start_date,
            end_date,
        )

    st.session_state.ina_df = df

    st.session_state.ina_error = error

    st.session_state.diagnostic = diag

    st.session_state.last_query = (
        str(start_date),
        str(end_date),
    )


# ============================================================
# RECUPERAR RESULTADOS
# ============================================================

df = st.session_state.ina_df

ina_error = st.session_state.ina_error

diag = st.session_state.diagnostic


# ============================================================
# ESTADO GENERAL
# ============================================================

st.subheader("Estado del sistema")

if ina_error is None and not df.empty:

    st.success(
        "INA conectado · datos observados disponibles"
    )

elif diag and diag.get("http_status") == 200:

    st.warning(
        "INA respondió, pero no se encontraron "
        "datos válidos para el período seleccionado."
    )

else:

    st.error(
        "No fue posible obtener datos válidos del INA."
    )


# ============================================================
# DIAGNÓSTICO INA
# ============================================================

with st.expander(
    "🔎 Diagnóstico INA",
    expanded=True,
):

    if diag is None:

        st.info(
            "Todavía no se realizó una consulta."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "HTTP",
            diag.get("http_status")
            if diag.get("http_status") is not None
            else "—",
        )

        col2.metric(
            "Serie",
            diag.get("series_id", "—"),
        )

        col3.metric(
            "Tipo",
            diag.get("tipo", "—"),
        )

        col4.metric(
            "Registros",
            diag.get("registros", 0),
        )

        col5, col6 = st.columns(2)

        col5.write(
            "**Desde:**",
            diag.get("desde", "—"),
        )

        col6.write(
            "**Hasta:**",
            diag.get("hasta", "—"),
        )

        st.write(
            "**Endpoint:**",
            diag.get("endpoint", "—"),
        )

        if diag.get("error"):

            st.error(
                f"Error informado: "
                f"{diag.get('error')}"
            )

        elif diag.get("http_status") == 200:

            if diag.get("registros", 0) > 0:

                st.success(
                    "El servicio INA está devolviendo registros."
                )

            else:

                st.warning(
                    "El INA responde HTTP 200, "
                    "pero la consulta devuelve 0 registros."
                )


# ============================================================
# ERROR DETALLADO
# ============================================================

if ina_error:

    st.warning(
        f"Detalle de la consulta: {ina_error}"
    )


# ============================================================
# SI NO HAY DATOS
# ============================================================

if df.empty:

    st.info(
        "Cuando el INA entregue observaciones válidas, "
        "en esta sección aparecerán el nivel actual, "
        "la variación y el gráfico histórico."
    )

    st.stop()


# ============================================================
# PREPARAR DATOS
# ============================================================

df = df.copy()

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce",
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "datetime",
        "value",
    ]
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


if df.empty:

    st.warning(
        "Se recibieron registros del INA, "
        "pero no quedaron valores válidos "
        "después de procesarlos."
    )

    st.stop()


# ============================================================
# ÚLTIMO NIVEL
# ============================================================

last_row = df.iloc[-1]

last_level = float(
    last_row["value"]
)

last_datetime = last_row[
    "datetime"
]


# ============================================================
# VARIACIÓN
# ============================================================

if len(df) >= 2:

    previous_level = float(
        df.iloc[-2]["value"]
    )

    variation = (
        last_level - previous_level
    )

else:

    previous_level = None

    variation = 0.0


# ============================================================
# TENDENCIA
# ============================================================

if variation > 0.01:

    trend_text = "▲ Creciendo"

elif variation < -0.01:

    trend_text = "▼ Bajando"

else:

    trend_text = "→ Estable"


# ============================================================
# MÉTRICAS
# ============================================================

st.subheader("📊 Nivel observado")

m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "San Nicolás",
    f"{last_level:.2f} m",
)


if previous_level is not None:

    m2.metric(
        "Variación",
        f"{variation:+.2f} m",
    )

else:

    m2.metric(
        "Variación",
        "—",
    )


m3.metric(
    "Tendencia",
    trend_text,
)


m4.metric(
    "Registros",
    f"{len(df)}",
)


st.caption(
    "Última medición disponible: "
    f"{last_datetime}"
)


# ============================================================
# GRÁFICO HISTÓRICO
# ============================================================

st.subheader(
    "Evolución del nivel · San Nicolás"
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["datetime"],
        y=df["value"],
        mode="lines+markers",
        name="Nivel observado",
        hovertemplate=(
            "%{x|%d/%m/%Y %H:%M}"
            "<br>"
            "Nivel: %{y:.2f} m"
            "<extra></extra>"
        ),
    )
)


fig.update_layout(
    height=480,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20,
    ),
    xaxis_title="Fecha",
    yaxis_title="Nivel (m)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)


# Escala fija solicitada para la plataforma
fig.update_yaxes(
    range=[0, 7],
    dtick=0.5,
)


fig.update_xaxes(
    tickformat="%d/%m/%Y",
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TABLA DE DATOS
# ============================================================

with st.expander(
    "📋 Ver datos observados"
):

    table_df = df[
        [
            "datetime",
            "value",
        ]
    ].copy()

    table_df.columns = [
        "Fecha / hora",
        "Nivel (m)",
    ]

    table_df["Nivel (m)"] = (
        table_df["Nivel (m)"]
        .round(2)
    )

    st.dataframe(
        table_df.sort_values(
            "Fecha / hora",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# INFORMACIÓN DEL SERVICIO
# ============================================================

meta = forecast_meta()


with st.expander(
    "ℹ️ Información de la fuente"
):

    st.write(
        "**Fuente:**",
        meta.get("fuente", "INA"),
    )

    st.write(
        "**Servicio:**",
        meta.get("servicio", "—"),
    )

    st.write(
        "**Estación:**",
        meta.get(
            "estacion",
            "San Nicolás",
        ),
    )

    st.write(
        "**Serie:**",
        meta.get("serie", 36),
    )

    st.write(
        "**Variable:**",
        meta.get(
            "variable",
            "Nivel hidrométrico",
        ),
    )

    st.write(
        "**Unidad:**",
        meta.get(
            "unidad",
            "metros",
        ),
    )

    if meta.get("observacion"):

        st.caption(
            meta["observacion"]
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás · "
    "Sistema experimental de monitoreo hidrológico. "
    "Los datos oficiales corresponden al INA."
)
