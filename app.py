from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.ina import observed
from src.monthly import (
    load_latest,
    prepare_actual,
    render_monthly,
)


APP_VERSION = "V11.22"
TZ = ZoneInfo("America/Argentina/Buenos_Aires")
BASE_DIR = Path(__file__).resolve().parent

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

SLUGS = [
    "corrientes",
    "goya",
    "la_paz",
    "parana",
    "diamante",
    "rosario",
    "villa_constitucion",
    "san_nicolas",
]

LEVEL_COLUMNS = {
    station: "nivel_" + slug
    for station, slug in zip(STATIONS, SLUGS)
}
FLOW_COLUMNS = {
    station: "q_" + slug
    for station, slug in zip(STATIONS, SLUGS)
}
RAIN_COLUMNS = {
    station: "rain_" + slug
    for station, slug in zip(STATIONS, SLUGS)
}

SEGMENTS = [
    (1, 15, "Pronóstico 1–15 días", "#2196f3"),
    (16, 30, "Proyección 16–30 días", "#f3b5b5"),
    (31, 45, "Escenario 31–45 días", "#ff7043"),
    (46, 60, "Tendencia 46–60 días", "#7ddc9a"),
]

try:
    PAGE_ICON = Image.open(BASE_DIR / "icon_rio_parana.png")
except Exception:
    PAGE_ICON = "🌊"

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1 { margin-bottom: 0.1rem; }
    h2 { margin-top: 1.4rem; }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e2e7;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.05);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #172b4d !important;
        font-size: 0.86rem;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #172b4d !important;
        font-size: 1.45rem;
    }

    .st-key-rio_navigation button {
        min-height: 58px;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .st-key-rio_navigation button[kind="primary"] {
        background: #164b83 !important;
        border-color: #164b83 !important;
    }
    .st-key-rio_navigation button[kind="primary"] p,
    .st-key-rio_navigation button[kind="primary"] span {
        color: white !important;
    }

    @media (max-width: 800px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        h1 { font-size: 1.7rem !important; }
        h2 { font-size: 1.3rem !important; }
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        .st-key-rio_navigation [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        .st-key-rio_navigation [data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError, OverflowError):
        return np.nan


def fmt(value, decimals=2, suffix=""):
    value = number(value)
    if not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def level(value):
    return fmt(value, 2, " m")


def flow(value):
    value = number(value)
    if not np.isfinite(value):
        return "—"
    return f"{value:,.0f}".replace(",", ".") + " m³/s"


def signed(value, suffix=" m"):
    value = number(value)
    if not np.isfinite(value):
        return None
    return f"{value:+.2f}{suffix}"


def mapping(value):
    return value if isinstance(value, dict) else {}


def frame(value):
    if not isinstance(value, pd.DataFrame):
        return pd.DataFrame(columns=["datetime"])

    result = value.copy()

    if "datetime" in result.columns:
        result["datetime"] = pd.to_datetime(
            result["datetime"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        result = (
            result.dropna(subset=["datetime"])
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
            .reset_index(drop=True)
        )

    return result


def valid_rows(data, column):
    if (
        not isinstance(data, pd.DataFrame)
        or "datetime" not in data.columns
        or column not in data.columns
    ):
        return pd.DataFrame(columns=["datetime", column])

    result = data.copy()
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["datetime", column]).sort_values(
        "datetime"
    )


def last_value(data, column):
    rows = valid_rows(data, column)
    return number(rows[column].iloc[-1]) if len(rows) else np.nan


def change(data, column, days):
    """Compara fechas separadas por días calendario."""
    rows = valid_rows(data, column)
    if rows.empty:
        return np.nan

    days_data = rows.copy()
    days_data["day"] = days_data["datetime"].dt.normalize()
    days_data = days_data.drop_duplicates("day", keep="last")

    last = days_data.iloc[-1]
    previous_day = last["day"] - pd.Timedelta(days=days)
    previous = days_data[days_data["day"] == previous_day]

    if previous.empty:
        return np.nan

    return float(last[column] - previous.iloc[-1][column])


def trend(value):
    value = number(value)
    if not np.isfinite(value):
        return "Sin datos"
    if value > 0.10:
        return "↑ Creciente"
    if value < -0.10:
        return "↓ Decreciente"
    return "→ Estable"


def timestamp_label(value):
    if value is None:
        return "No disponible"

    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return "No disponible"
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(TZ)
            return stamp.strftime("%d/%m/%Y %H:%M") + " (Argentina)"
        return stamp.strftime("%d/%m/%Y %H:%M") + " (hora sin zona)"
    except Exception:
        return str(value)


def table(data, formats=None):
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.info("Sin datos disponibles para esta sección.")
        return

    if formats:
        formats = {
            key: value
            for key, value in formats.items()
            if key in data.columns
        }
        data = data.style.format(formats, na_rep="—")

    st.dataframe(data, use_container_width=True, hide_index=True)


def add_line(fig, data, column, name, color=None, dash="solid"):
    if (
        data.empty
        or "datetime" not in data.columns
        or column not in data.columns
    ):
        return

    line_style = {"width": 2.5, "dash": dash}
    if color:
        line_style["color"] = color

    fig.add_trace(
        go.Scatter(
            x=data["datetime"],
            y=pd.to_numeric(data[column], errors="coerce"),
            name=name,
            mode="lines",
            connectgaps=False,
            line=line_style,
            hovertemplate=(
                "%{x|%d/%m/%Y}<br>%{y:.2f}<extra>%{fullData.name}</extra>"
            ),
        )
    )


def show_chart(fig, ylabel, height=420, fixed_level=False):
    fig.update_layout(
        height=height,
        hovermode="x unified",
        margin={"l": 15, "r": 15, "t": 30, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
        },
        xaxis_title="Fecha",
        yaxis_title=ylabel,
    )
    fig.update_xaxes(tickformat="%d/%m")
    if fixed_level:
        fig.update_yaxes(range=[0, 7])
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def query_ina(day_string):
    day = date.fromisoformat(day_string)
    raw, error = observed(
        (day - timedelta(days=10)).isoformat(),
        (day + timedelta(days=1)).isoformat(),
    )

    if error:
        return pd.DataFrame(), str(error), datetime.now(TZ)

    try:
        return prepare_actual(raw), None, datetime.now(TZ)
    except Exception as exc:
        return pd.DataFrame(), str(exc), datetime.now(TZ)


def choose_view(view):
    st.session_state["rio_view"] = view


st.title("🌊 PARANÁ · SAN NICOLÁS")
st.caption(
    f"{APP_VERSION} · Lectura oficial INA y pronóstico hidrológico"
)

# Una nueva sesión abre siempre el tablero.
# La versión nueva también migra las sesiones anteriores.
if st.session_state.get("rio_navigation_version") != APP_VERSION:
    st.session_state["rio_view"] = "full"
    st.session_state["rio_navigation_version"] = APP_VERSION

with st.container(key="rio_navigation"):
    full_button, monthly_button = st.columns(2)

    with full_button:
        st.button(
            "Pronóstico y tablero completo",
            type=(
                "primary"
                if st.session_state["rio_view"] == "full"
                else "secondary"
            ),
            use_container_width=True,
            on_click=choose_view,
            args=("full",),
        )

    with monthly_button:
        st.button(
            "Control mensual",
            type=(
                "primary"
                if st.session_state["rio_view"] == "monthly"
                else "secondary"
            ),
            use_container_width=True,
            on_click=choose_view,
            args=("monthly",),
        )

if st.session_state["rio_view"] == "monthly":
    render_monthly()
    st.stop()

today = datetime.now(TZ).date()

with st.sidebar:
    st.header("Visualización")
    visible_days = st.slider(
        "Historia visible",
        min_value=30,
        max_value=730,
        value=120,
        step=30,
    )
    if st.button("Consultar nuevamente INA", use_container_width=True):
        query_ina.clear()

    st.caption(
        "La lectura oficial se consulta al abrir el tablero "
        "y se conserva hasta cinco minutos en caché. "
        "El pronóstico se actualiza mediante la tarea programada."
    )
    st.divider()
    st.write("15 días · pronóstico")
    st.write("30 días · proyección")
    st.write("45 días · escenario extendido")
    st.write("60 días · tendencia hidrológica")

try:
    saved = load_latest()
except Exception as exc:
    saved = None
    st.error(f"No se pudo leer el resultado guardado: {exc}")

with st.spinner("Consultando la última lectura oficial de INA..."):
    try:
        live, live_error, checked_at = query_ina(today.isoformat())
    except Exception as exc:
        live = pd.DataFrame()
        live_error = str(exc)
        checked_at = datetime.now(TZ)

official_value = np.nan
official_timestamp = None

if not live.empty:
    official_value = number(live.iloc[-1]["value"])
    official_timestamp = pd.Timestamp(live.iloc[-1]["timestamp"])

st.subheader("Lectura oficial · San Nicolás")

if official_timestamp is not None:
    official_day = official_timestamp.tz_convert(TZ).date()

    a, b = st.columns(2)
    with a:
        st.metric(
            (
                "Lectura del día según INA"
                if official_day == today
                else "Última lectura disponible según INA"
            ),
            level(official_value),
        )
    with b:
        st.write("**Fecha y hora de la observación**")
        st.write(timestamp_label(official_timestamp))

    st.caption(
        "Este nivel corresponde a una observación publicada por INA; "
        "no es una previsión ni una interpolación."
    )

    if official_day < today:
        st.warning(
            "La última observación recibida corresponde al "
            f"{official_day:%d/%m/%Y}. No se presenta como lectura de hoy."
        )

else:
    st.warning(
        "No se pudo verificar la lectura oficial en esta consulta. "
        + (live_error or "INA no devolvió observaciones válidas.")
    )

st.caption("Consulta realizada: " + timestamp_label(checked_at))

if not isinstance(saved, dict) or not saved.get("data_loaded"):
    st.info(
        "Todavía no hay un pronóstico completo guardado. "
        "Ejecutá Actions → Actualizar Rio Parana → Run workflow."
    )
    st.stop()

# Se trabaja con el archivo recién leído, evitando mezclar
# resultados antiguos que pudieran quedar en session_state.
sn_history = frame(saved.get("sn_history"))
upstream_history = frame(saved.get("upstream_history"))
exog_history = frame(saved.get("exog_history"))
exog_future = frame(saved.get("exog_future"))
forecast = frame(saved.get("forecast"))

hydrology = mapping(saved.get("hydrology"))
models = mapping(saved.get("models"))
metrics = mapping(saved.get("metrics"))
upstream_meta = mapping(saved.get("upstream_meta"))
exog_meta = mapping(saved.get("exog_meta"))

if sn_history.empty or "nivel" not in sn_history.columns:
    st.error("El resultado guardado no contiene el historial local.")
    st.stop()

base_ts = pd.Timestamp(saved["base_date"])
if base_ts.tzinfo is not None:
    base_ts = base_ts.tz_convert(TZ).tz_localize(None)
base_ts = base_ts.normalize()

base_level = number(saved.get("observation_level"))
base_observed_at = saved.get("observation_timestamp")
has_official_metadata = (
    np.isfinite(base_level) and base_observed_at is not None
)

if not has_official_metadata:
    base_level = last_value(sn_history, "nivel")
    st.warning(
        "Este resultado no contiene los metadatos de lectura oficial "
        "del pipeline V11.21. Ejecutá la actualización programada "
        "para regenerarlo."
    )

st.subheader("Base del pronóstico guardado")
a, b, c = st.columns(3)
a.metric("Nivel utilizado para iniciar el cálculo", level(base_level))
b.metric("Fecha base", base_ts.strftime("%d/%m/%Y"))
with c:
    st.write("**Generado**")
    st.write(timestamp_label(saved.get("last_update")))

st.caption(
    "Hora de la observación base: "
    + timestamp_label(base_observed_at)
)

same_origin = False

if has_official_metadata and official_timestamp is not None:
    try:
        stored_timestamp = pd.Timestamp(base_observed_at)
        same_origin = (
            stored_timestamp.tzinfo is not None
            and stored_timestamp.tz_convert("UTC")
            == official_timestamp.tz_convert("UTC")
            and np.isclose(
                base_level,
                official_value,
                atol=1e-9,
                rtol=0,
            )
        )
    except Exception:
        same_origin = False

    if same_origin:
        st.success(
            "El pronóstico guardado parte de la última lectura INA "
            "recibida en esta consulta."
        )
    else:
        st.warning(
            "La lectura oficial consultada y la base del pronóstico "
            "guardado no coinciden. El gráfico conserva el cálculo "
            "emitido; falta actualizarlo desde la nueva observación."
        )
        st.caption(
            "Actualización: Actions → Actualizar Rio Parana "
            "→ Run workflow. Consultar INA en este tablero "
            "no ejecuta el entrenamiento."
        )

elif has_official_metadata:
    st.info(
        "Se muestra la base oficial registrada al generar el cálculo. "
        "No pudo comprobarse si hay una observación más reciente."
    )

st.caption(
    "El cálculo debe comenzar desde una lectura observada. "
    "Actualizar la tarjeta con una nueva lectura no cambia "
    "retroactivamente el pronóstico guardado."
)

# Las lecturas recientes se agregan solo a la visualización.
# No se guardan como pronósticos ni alteran el resultado del modelo.
display_history = sn_history[["datetime", "nivel"]].copy()

if not live.empty:
    recent = pd.DataFrame(
        {
            "datetime": pd.to_datetime(live["day"]),
            "nivel": pd.to_numeric(live["value"], errors="coerce"),
        }
    )
    display_history["datetime"] = (
        display_history["datetime"].dt.normalize()
    )
    display_history = (
        pd.concat([display_history, recent], ignore_index=True)
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

visible_start = pd.Timestamp(today) - pd.Timedelta(days=visible_days)
visible_observed = display_history[
    display_history["datetime"] >= visible_start
].copy()

delta_1 = change(display_history, "nivel", 1)
delta_7 = change(display_history, "nivel", 7)

hydro_estimate = mapping(hydrology.get("current_estimate"))
delay = hydro_estimate.get("delay_days")
delay_min = hydro_estimate.get("delay_min")
delay_max = hydro_estimate.get("delay_max")
correlation = number(hydro_estimate.get("correlation"))
flow_station = exog_meta.get("main_flow_station")

st.subheader("Estado del corredor")
a, b, c, d = st.columns(4)
a.metric("Variación observada · 1 día", signed(delta_1) or "—")
b.metric("Tendencia observada · 7 días", trend(delta_7), signed(delta_7))
c.metric(
    "Caudal de referencia guardado",
    flow(last_value(exog_history, "caudal_m3s")),
)
d.metric(
    "Demora Corrientes → SN",
    f"{delay} días" if delay is not None else "—",
)

if flow_station:
    st.caption(
        f"Caudal de referencia: {flow_station}. "
        "Su origen observado o reconstruido se detalla más abajo."
    )

a, b, c = st.columns(3)
a.metric(
    "Corrientes · variación 7 días",
    signed(hydro_estimate.get("corrientes_change_7d")) or "—",
)
b.metric(
    "Respuesta histórica SN / Corrientes",
    fmt(hydro_estimate.get("response_m_per_m"), 2, " m/m"),
)
c.metric(
    "Impacto estimado en San Nicolás",
    signed(hydro_estimate.get("expected_sn_change")) or "—",
)
st.caption(
    "La respuesta histórica se aplica sobre variaciones de nivel. "
    "Las estaciones tienen ceros hidrométricos diferentes."
)

if forecast.empty or not {"datetime", "prediction"}.issubset(
    forecast.columns
):
    st.error("El archivo guardado no contiene un pronóstico válido.")
    st.stop()

if "horizon_day" not in forecast.columns:
    forecast["horizon_day"] = (
        forecast["datetime"].dt.normalize() - base_ts
    ).dt.days

st.subheader("Nivel de San Nicolás · observado y proyección")
fig = go.Figure()

if {"lower", "upper"}.issubset(forecast.columns):
    fig.add_trace(
        go.Scatter(
            x=forecast["datetime"],
            y=forecast["upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["datetime"],
            y=forecast["lower"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.18)",
            name="Incertidumbre",
            hoverinfo="skip",
        )
    )

add_line(
    fig, visible_observed, "nivel", "Observado INA", "#80cbc4"
)

for start, end, label, color in SEGMENTS:
    segment = forecast[
        forecast["horizon_day"].between(start, end)
    ].copy()

    if segment.empty:
        continue

    if start == 1 and np.isfinite(base_level):
        anchor = pd.DataFrame(
            {"datetime": [base_ts], "prediction": [base_level]}
        )
    else:
        anchor = forecast[
            forecast["horizon_day"] == start - 1
        ][["datetime", "prediction"]]

    plotted = pd.concat(
        [anchor, segment[["datetime", "prediction"]]],
        ignore_index=True,
    )
    add_line(fig, plotted, "prediction", label, color)

add_line(
    fig,
    forecast,
    "scenario_adverse",
    "Escenario adverso",
    "#d4a72c",
    "dash",
)
add_line(
    fig,
    forecast,
    "scenario_extreme",
    "Escenario extremo histórico",
    "#ef4444",
    "dot",
)

if np.isfinite(base_level):
    fig.add_trace(
        go.Scatter(
            x=[base_ts],
            y=[base_level],
            mode="markers",
            name="Lectura base del cálculo",
            marker={"size": 11, "color": "#16a34a"},
        )
    )

show_chart(fig, "Nivel [m]", height=520, fixed_level=True)
st.caption(
    "Escala inicial de 0 a 7 m; podés ampliar con las herramientas "
    "del gráfico. Los escenarios históricos no son lecturas oficiales "
    "ni un pronóstico meteorológico exacto a 60 días."
)

st.subheader("Horizontes del pronóstico")
scenario_rows = []

for column, day in zip(st.columns(4), [15, 30, 45, 60]):
    match = forecast[forecast["horizon_day"] == day]

    with column:
        if match.empty:
            st.metric(f"{day} días", "—")
            continue

        row = match.iloc[0]
        prediction = number(row.get("prediction"))
        st.metric(
            f"{day} días",
            level(prediction),
            signed(prediction - base_level),
        )
        st.caption(
            pd.Timestamp(row["datetime"]).strftime("%d/%m/%Y")
            + " · variación respecto de la base"
        )
        st.caption("Adverso: " + level(row.get("scenario_adverse")))
        st.caption("Extremo: " + level(row.get("scenario_extreme")))

        scenario_rows.append(
            {
                "Horizonte": f"{day} días",
                "Fecha": pd.Timestamp(row["datetime"]).strftime(
                    "%d/%m/%Y"
                ),
                "Central [m]": prediction,
                "Probable histórico [m]": number(
                    row.get("scenario_probable")
                ),
                "Adverso [m]": number(row.get("scenario_adverse")),
                "Extremo [m]": number(row.get("scenario_extreme")),
            }
        )

st.subheader("Escenarios históricos")
scenario_table = pd.DataFrame(scenario_rows)
table(
    scenario_table,
    {
        name: "{:.2f}"
        for name in scenario_table.columns
        if name.endswith("[m]")
    },
)

with st.expander("Pronóstico completo · valores y descarga"):
    export = forecast.copy()
    export["datetime"] = export["datetime"].dt.strftime("%d/%m/%Y")
    table(export)
    st.download_button(
        "Descargar pronóstico CSV",
        export.to_csv(index=False, sep=";", decimal=",").encode(
            "utf-8-sig"
        ),
        "pronostico_san_nicolas.csv",
        "text/csv",
    )

st.subheader("Estado del corredor aguas arriba")
corridor_rows = []
quality_rows = []

for station in STATIONS:
    lc = LEVEL_COLUMNS[station]
    qc = FLOW_COLUMNS[station]
    rc = RAIN_COLUMNS[station]

    hist = display_history if station == "San Nicolás" else upstream_history
    column = "nivel" if station == "San Nicolás" else lc
    rows = valid_rows(hist, column)

    reading_date = (
        rows.iloc[-1]["datetime"].strftime("%d/%m/%Y")
        if not rows.empty
        else "—"
    )

    rain_rows = valid_rows(exog_history, rc)
    rain_total = np.nan

    if not rain_rows.empty:
        rain_end = rain_rows["datetime"].max().normalize()
        rain_window = rain_rows[
            rain_rows["datetime"].dt.normalize().between(
                rain_end - pd.Timedelta(days=6), rain_end
            )
        ]
        rain_total = rain_window[rc].sum(min_count=1)

    d7 = change(hist, column, 7)
    corridor_rows.append(
        {
            "Estación": station,
            "Fecha del nivel": reading_date,
            "Nivel [m]": last_value(hist, column),
            "Δ 1 día [m]": change(hist, column, 1),
            "Δ 7 días [m]": d7,
            "Estado": trend(d7),
            "Caudal [m³/s]": last_value(exog_history, qc),
            "Lluvia disponible 7d [mm]": rain_total,
        }
    )

    flow_rows = valid_rows(exog_history, qc)
    last = flow_rows.iloc[-1] if not flow_rows.empty else {}
    quality_rows.append(
        {
            "Estación": station,
            "Caudal [m³/s]": number(last.get(qc)),
            "Origen": last.get(qc + "_source", "No informado"),
            "Calidad": number(last.get(qc + "_quality")),
            "Fecha": (
                pd.Timestamp(last["datetime"]).strftime("%d/%m/%Y")
                if "datetime" in last
                else "—"
            ),
        }
    )

table(
    pd.DataFrame(corridor_rows),
    {
        "Nivel [m]": "{:.2f}",
        "Δ 1 día [m]": "{:+.2f}",
        "Δ 7 días [m]": "{:+.2f}",
        "Caudal [m³/s]": "{:,.0f}",
        "Lluvia disponible 7d [mm]": "{:.1f}",
    },
)
st.caption(
    "Los datos aguas arriba corresponden a la última ejecución "
    "guardada. Las fechas pueden diferir entre estaciones. "
    "La lluvia acumulada disponible puede estar incompleta."
)

st.subheader("Caudales · observado / reconstruido")
table(
    pd.DataFrame(quality_rows),
    {"Caudal [m³/s]": "{:,.0f}", "Calidad": "{:.0%}"},
)
st.caption(
    "Los caudales reconstruidos e interpolados son estimaciones. "
    "Se conserva el origen informado por el cálculo."
)

st.subheader("Caudal histórico y proyección")
available = [
    station for station in STATIONS
    if not valid_rows(exog_history, FLOW_COLUMNS[station]).empty
]

if available:
    selected = st.selectbox(
        "Estación de caudal",
        available,
        index=available.index(flow_station) if flow_station in available else 0,
    )
    qc = FLOW_COLUMNS[selected]
    qhist = exog_history[
        exog_history["datetime"] >= visible_start
    ].copy()

    future = exog_future.copy()
    if "datetime" in future:
        future["horizon"] = (
            future["datetime"].dt.normalize() - base_ts
        ).dt.days

    def future_flow(day):
        if qc not in future or "horizon" not in future:
            return np.nan
        match = future[future["horizon"] == day]
        return number(match.iloc[-1][qc]) if len(match) else np.nan

    maximum = np.nan
    if qc in future and "horizon" in future:
        values = pd.to_numeric(
            future.loc[future["horizon"].between(1, 15), qc],
            errors="coerce",
        )
        maximum = values.max()

    for box, label, value in zip(
        st.columns(5),
        ["Último guardado", "Máximo 15 días", "Día 15", "Día 30", "Día 60"],
        [
            last_value(exog_history, qc),
            maximum,
            future_flow(15),
            future_flow(30),
            future_flow(60),
        ],
    ):
        box.metric(label, flow(value))

    fig = go.Figure()
    source_col = qc + "_source"

    if source_col in qhist:
        mask = qhist[source_col].fillna("").astype(str).str.lower().eq(
            "observado"
        )
        direct = qhist.copy()
        estimated = qhist.copy()
        direct[qc] = pd.to_numeric(direct[qc], errors="coerce").where(mask)
        estimated[qc] = pd.to_numeric(
            estimated[qc], errors="coerce"
        ).where(~mask)
        add_line(fig, direct, qc, "Observado", "#2196f3")
        add_line(
            fig, estimated, qc, "Reconstruido / interpolado",
            "#ff9800", "dot"
        )
    else:
        add_line(fig, qhist, qc, "Histórico · origen no informado")

    if qc in future and "horizon" in future:
        for start, end, label, color in SEGMENTS:
            add_line(
                fig,
                future[future["horizon"].between(start, end)],
                qc,
                label,
                color,
            )
    show_chart(fig, "Caudal [m³/s]")
else:
    st.info("No hay series de caudal disponibles.")

st.subheader("Lluvia por punto del corredor")
available = [
    station for station in STATIONS
    if RAIN_COLUMNS[station] in exog_history.columns
]

if available:
    selected = st.selectbox("Estación de lluvia", available)
    rc = RAIN_COLUMNS[selected]
    hist = exog_history[
        exog_history["datetime"] >= visible_start
    ]
    fig = go.Figure()
    fig.add_bar(x=hist["datetime"], y=hist[rc], name="Histórico")

    if rc in exog_future and "datetime" in exog_future:
        future = exog_future.copy()
        if "rain_forecast_available" in future:
            available_mask = (
                future["rain_forecast_available"]
                .astype(str).str.lower().isin(["true", "1", "1.0"])
            )
            future = future[available_mask]
        else:
            future = future[
                future["datetime"].between(
                    base_ts + pd.Timedelta(days=1),
                    base_ts + pd.Timedelta(days=15),
                )
            ]

        fig.add_bar(
            x=future["datetime"],
            y=future[rc],
            name="Previsión de lluvia disponible",
        )

    show_chart(fig, "Precipitación [mm]", height=380)
else:
    st.info("No hay datos de lluvia disponibles.")

st.subheader("Propagación Corrientes → San Nicolás")
a, b, c, d = st.columns(4)
a.metric("Retardo estimado", f"{delay} días" if delay is not None else "—")
b.metric(
    "Rango histórico",
    f"{delay_min}–{delay_max} días"
    if delay_min is not None and delay_max is not None else "—",
)
c.metric("Correlación", fmt(correlation))
d.metric("Eventos similares", str(hydro_estimate.get("similar_event_count", 0)))

if "nivel_corrientes" in upstream_history:
    compare = upstream_history[["datetime", "nivel_corrientes"]].merge(
        sn_history[["datetime", "nivel"]], on="datetime", how="inner"
    )
    for col in ["nivel_corrientes", "nivel"]:
        compare[col] = pd.to_numeric(compare[col], errors="coerce")
    compare = compare.dropna()

    if len(compare) >= 30:
        for col in ["nivel_corrientes", "nivel"]:
            median = compare[col].rolling(30, min_periods=10).median()
            deviation = compare[col].rolling(60, min_periods=20).std()
            compare[col + "_anom"] = (
                compare[col] - median
            ) / deviation.replace(0, np.nan)

        compare = compare[
            compare["datetime"] >= base_ts - pd.Timedelta(days=730)
        ]
        fig = go.Figure()
        add_line(fig, compare, "nivel_corrientes_anom", "Corrientes")
        add_line(fig, compare, "nivel_anom", "San Nicolás")
        show_chart(fig, "Anomalía normalizada")
        st.caption(
            "Comparación normalizada: cada estación utiliza "
            "un cero hidrométrico diferente."
        )

st.subheader("Eventos históricos comparables")
events = hydrology.get("similar_events")
if isinstance(events, pd.DataFrame) and not events.empty:
    table(
        events.head(15).rename(
            columns={
                "start_date": "Inicio",
                "peak_date": "Pico",
                "start_level_sn": "Nivel inicial SN",
                "peak_level_sn": "Pico SN",
                "rise_sn": "Crecimiento",
                "rise_days": "Duración",
                "similarity_distance": "Distancia",
                "similarity_weight": "Peso",
            }
        )
    )
else:
    st.info("No hay suficientes eventos comparables disponibles.")

st.subheader("Comparativa año contra año")
years = sorted(
    sn_history["datetime"].dt.year.unique().tolist(), reverse=True
)
selected_years = st.multiselect("Años a comparar", years, default=years[:5])
a, b = st.columns(2)
start_window = a.date_input(
    "Inicio de ventana estacional", date(2024, 1, 1), format="DD/MM/YYYY"
)
end_window = b.date_input(
    "Fin de ventana estacional", date(2024, 12, 31), format="DD/MM/YYYY"
)

fig = go.Figure()
start_md = start_window.month * 100 + start_window.day
end_md = end_window.month * 100 + end_window.day
season_origin = pd.Timestamp(
    year=2024, month=start_window.month, day=start_window.day
)

for year in selected_years:
    annual = sn_history[sn_history["datetime"].dt.year == year].copy()
    md = annual["datetime"].dt.month * 100 + annual["datetime"].dt.day
    mask = (
        md.between(start_md, end_md)
        if start_md <= end_md
        else (md >= start_md) | (md <= end_md)
    )
    annual = annual[mask].copy()
    if annual.empty:
        continue

    # Alinea por calendario, sin comprimir días faltantes.
    season_dates = pd.to_datetime(
        annual["datetime"].dt.strftime("2024-%m-%d")
    )
    annual["season_day"] = (
        (season_dates - season_origin).dt.days % 366
    )
    annual = annual.sort_values("season_day")
    fig.add_scatter(
        x=annual["season_day"],
        y=annual["nivel"],
        mode="lines",
        name=str(year),
        connectgaps=False,
    )

fig.update_layout(
    height=430,
    hovermode="x unified",
    xaxis_title="Días calendario desde el inicio de la ventana",
    yaxis_title="Nivel San Nicolás [m]",
    yaxis={"range": [0, 7]},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Corrientes vs San Nicolás · comparación anual")
yearly = hydrology.get("corrientes_yearly")
robust = mapping(hydrology.get("corrientes_robust"))
dataset = frame(hydrology.get("dataset"))

if isinstance(yearly, pd.DataFrame) and not yearly.empty and "year" in yearly:
    annual_table = yearly.sort_values("year", ascending=False).copy()
    a, b, c, d = st.columns(4)
    a.metric("Demora robusta", fmt(robust.get("delay_days"), 0, " días"))
    b.metric(
        "Rango histórico",
        f"{robust.get('delay_min', '—')}–{robust.get('delay_max', '—')} días",
    )
    c.metric("Correlación mediana", fmt(robust.get("correlation")))
    d.metric("Respuesta SN / Corrientes", fmt(robust.get("response_m_per_m")))

    selected_year = st.selectbox(
        "Año para comparar la propagación",
        annual_table["year"].dropna().astype(int).tolist(),
    )
    row = annual_table[annual_table["year"] == selected_year].iloc[0]
    lag = number(row.get("lag_days"))

    needed = {"datetime", "nivel_corrientes", "nivel_san_nicolas"}
    if needed.issubset(dataset.columns):
        annual = dataset[
            dataset["datetime"].dt.year == selected_year
        ].copy()

        for col in ["nivel_corrientes", "nivel_san_nicolas"]:
            annual[col] = pd.to_numeric(annual[col], errors="coerce")

        if np.isfinite(lag):
            fig = go.Figure()
            for col, name, shift in [
                ("nivel_corrientes", "Corrientes trasladado", int(lag)),
                ("nivel_san_nicolas", "San Nicolás", 0),
            ]:
                values = annual[col].dropna()
                if values.empty:
                    continue
                fig.add_scatter(
                    x=annual["datetime"] + pd.Timedelta(days=shift),
                    y=annual[col] - values.iloc[0],
                    mode="lines",
                    name=name,
                    connectgaps=False,
                )
            show_chart(fig, "Cambio respecto del inicio del año [m]")
            st.caption(
                f"Corrientes se desplaza {int(lag)} días. "
                "Las curvas representan variaciones, no alturas equivalentes."
            )

        st.write("**Alturas en la misma fecha, sin aplicar retardo**")
        paired_source = dataset.dropna(
            subset=["nivel_corrientes", "nivel_san_nicolas"]
        )
        paired_years = sorted(
            paired_source["datetime"].dt.year.unique().tolist(),
            reverse=True,
        )

        if paired_years:
            same_year = st.selectbox(
                "Año para comparar lectura contra lectura",
                paired_years,
                index=(
                    paired_years.index(selected_year)
                    if selected_year in paired_years else 0
                ),
            )
            paired = paired_source[
                paired_source["datetime"].dt.year == same_year
            ].copy()
            for col in ["nivel_corrientes", "nivel_san_nicolas"]:
                paired[col] = pd.to_numeric(paired[col], errors="coerce")
            paired = paired.dropna(
                subset=["nivel_corrientes", "nivel_san_nicolas"]
            )

            fig = go.Figure()
            fig.add_scatter(
                x=paired["datetime"], y=paired["nivel_corrientes"],
                name="Corrientes", mode="lines",
            )
            fig.add_scatter(
                x=paired["datetime"], y=paired["nivel_san_nicolas"],
                name="San Nicolás", mode="lines", yaxis="y2",
            )
            fig.update_layout(
                height=440,
                hovermode="x unified",
                xaxis={"tickformat": "%d/%m"},
                yaxis={"title": "Corrientes [m]", "rangemode": "tozero"},
                yaxis2={
                    "title": "San Nicolás [m]",
                    "overlaying": "y",
                    "side": "right",
                    "range": [0, 7],
                    "showgrid": False,
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            if not paired.empty:
                a, b, c, d, e = st.columns(5)
                a.metric("Corrientes · inicio", level(paired["nivel_corrientes"].iloc[0]))
                b.metric("San Nicolás · inicio", level(paired["nivel_san_nicolas"].iloc[0]))
                c.metric("Máximo Corrientes", level(paired["nivel_corrientes"].max()))
                d.metric("Máximo San Nicolás", level(paired["nivel_san_nicolas"].max()))
                e.metric("Lecturas coincidentes", len(paired))
                correlation_same = paired["nivel_corrientes"].corr(
                    paired["nivel_san_nicolas"]
                )
                st.caption(
                    "Correlación descriptiva en la misma fecha: "
                    + fmt(correlation_same, 3)
                    + ". No incorpora propagación."
                )

    table(
        annual_table.rename(
            columns={
                "year": "Año",
                "lag_days": "Demora [días]",
                "correlation": "Correlación",
                "response_m_per_m": "Respuesta SN/Corrientes [m/m]",
                "overlap": "Pares",
            }
        )
    )
else:
    st.info("No hay suficientes datos para la comparación anual.")

st.subheader("Cobertura de variables del modelo")
coverage = [
    {"Grupo": "Nivel San Nicolás", "Activo": True, "Variables": 1}
]
for label, active_key, count_key in [
    ("Niveles aguas arriba", "uses_upstream", "upstream_feature_count"),
    ("Caudales", "uses_caudal", "flow_feature_count"),
    ("Lluvias", "uses_rain", "rain_feature_count"),
    ("Hidrología / propagación", "uses_hydrology", "hydrology_feature_count"),
]:
    coverage.append(
        {
            "Grupo": label,
            "Activo": bool(models.get(active_key, False)),
            "Variables": models.get(count_key, 0),
        }
    )
table(pd.DataFrame(coverage))

st.subheader("Modelo y entrenamiento")
a, b, c, d = st.columns(4)
a.metric("RMSE informado por el modelo", fmt(metrics.get("rmse"), 3, " m"))
b.metric("MAE informado por el modelo", fmt(metrics.get("mae"), 3, " m"))
c.metric("Filas de entrenamiento", str(models.get("training_rows", 0)))
d.metric("Variables", str(models.get("feature_count", 0)))
st.caption(
    "Estas métricas del modelo no reemplazan la comparación "
    "de pronósticos emitidos contra observaciones posteriores."
)

importance = models.get("importance")
if (
    isinstance(importance, pd.DataFrame)
    and {"importance", "feature"}.issubset(importance.columns)
    and not importance.empty
):
    with st.expander("Importancia de variables"):
        top = importance.head(30)
        fig = go.Figure(
            go.Bar(
                x=top["importance"],
                y=top["feature"],
                orientation="h",
            )
        )
        fig.update_layout(
            height=650,
            yaxis={"autorange": "reversed"},
            xaxis_title="Importancia",
        )
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Metodología y lectura del tablero"):
    st.markdown(
        """
        - **Lectura oficial:** última observación recibida de INA,
          identificada por fecha y hora.
        - **Base del cálculo:** observación utilizada al generar el
          pronóstico guardado. Si aparece una lectura nueva, se señala
          la diferencia hasta que se ejecute la actualización.
        - **1–15 días:** pronóstico del modelo.
        - **16–30 días:** proyección hidrológica.
        - **31–60 días:** escenarios y tendencia extendida.
        - **Caudales:** se distingue el origen observado del reconstruido.
        - **Control mensual:** mantiene la primera emisión del mes;
          recalcular hoy no modifica esa referencia.
        - **Error:** debe evaluarse con pronósticos guardados antes de
          conocer la observación. La nueva lectura no elimina el error
          histórico ni demuestra por sí sola una mejora del modelo.
        """
    )

with st.expander("Diagnóstico técnico"):
    st.write("Versión de interfaz:", APP_VERSION)
    st.write("Versión de pipeline guardado:", saved.get("pipeline_version"))
    st.write("Actualización del cálculo:", timestamp_label(saved.get("last_update")))
    st.write("Observación utilizada:", timestamp_label(base_observed_at))
    st.write("Nivel utilizado:", base_level)
    st.write("Última observación consultada:", timestamp_label(official_timestamp))
    st.write("Coincide con la base:", same_origin)
    st.write("Estaciones aguas arriba:", upstream_meta.get("available_stations", []))
    st.write("Caudales observados:", exog_meta.get("flow_observed_stations", []))
    st.write("Caudales disponibles:", exog_meta.get("flow_available_stations", []))
    st.write("Caudal principal:", flow_station)

    for key, label in [
        ("lag_to_sn", "Retardos a San Nicolás"),
        ("corridor_lags", "Retardos por tramo"),
    ]:
        value = hydrology.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty:
            st.write(label)
            table(value)

    st.write("Presión hidrológica:", hydrology.get("pressure", {}))
    st.write(
        "Modelo:",
        {
            key: models.get(key)
            for key in [
                "version",
                "training_rows",
                "feature_count",
                "upstream_feature_count",
                "flow_feature_count",
                "rain_feature_count",
                "hydrology_feature_count",
            ]
        },
    )

st.divider()
st.caption(
    f"PARANÁ · SAN NICOLÁS · {APP_VERSION}. "
    "Pronóstico experimental. Las proyecciones no sustituyen "
    "los avisos oficiales de INA."
)
