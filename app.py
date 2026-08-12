import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import date, timedelta

from src.ina import observed, forecast_meta, STATIONS
from src.model import train, predict, prob
from src.scenario import run


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná San Nicolás | Pronóstico del río",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# ESTILO / ENCABEZADO
# ============================================================

st.title("🌊 PARANÁ · SAN NICOLÁS")

st.caption(
    "V9 · Plataforma pública de monitoreo y predicción experimental"
)


st.markdown(
    """
    Esta plataforma utiliza datos hidrométricos observados del
    Instituto Nacional del Agua (INA) y un modelo experimental
    propio para estimar la evolución del nivel del río Paraná
    en San Nicolás.
    """
)


# ============================================================
# VARIABLES DE SESIÓN
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "errors" not in st.session_state:
    st.session_state.errors = []

if "pred" not in st.session_state:
    st.session_state.pred = {}

if "met" not in st.session_state:
    st.session_state.met = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Consulta online")

    start = st.date_input(
        "Desde",
        date.today() - timedelta(days=365 * 3)
    )

    end = st.date_input(
        "Hasta",
        date.today()
    )

    # --------------------------------------------------------
    # VALIDACIÓN DE FECHAS
    # --------------------------------------------------------

    if start > end:

        st.error(
            "La fecha inicial no puede ser posterior "
            "a la fecha final."
        )

    # --------------------------------------------------------
    # BOTÓN INA
    # --------------------------------------------------------

    if st.button(
        "🔄 Actualizar INA",
        type="primary",
        use_container_width=True
    ):

        if start > end:

            st.error(
                "Corrija el rango de fechas antes de consultar."
            )

        else:

            with st.spinner(
                "Consultando datos hidrométricos del INA..."
            ):

                try:

                    result = observed(
                        start.isoformat(),
                        end.isoformat()
                    )

                    # ------------------------------------------------
                    # observed() normalmente devuelve:
                    #
                    # dataframe, errors
                    #
                    # Pero mantenemos compatibilidad por seguridad.
                    # ------------------------------------------------

                    if (
                        isinstance(result, tuple)
                        and len(result) == 2
                    ):

                        df_result, errors_result = result

                    else:

                        df_result = result
                        errors_result = []

                    # ------------------------------------------------
                    # Validación del DataFrame
                    # ------------------------------------------------

                    if df_result is None:

                        raise RuntimeError(
                            "El INA no devolvió un DataFrame."
                        )

                    if not isinstance(
                        df_result,
                        pd.DataFrame
                    ):

                        df_result = pd.DataFrame(
                            df_result
                        )

                    if df_result.empty:

                        raise RuntimeError(
                            "La API del INA no devolvió "
                            "datos para el período seleccionado."
                        )

                    # ------------------------------------------------
                    # Guardar resultado
                    # ------------------------------------------------

                    st.session_state.df = df_result
                    st.session_state.errors = (
                        errors_result or []
                    )

                    # Limpiar resultados anteriores
                    st.session_state.pred = {}
                    st.session_state.met = {}

                    st.success(
                        "✅ Datos del INA actualizados correctamente."
                    )

                except Exception as exc:

                    st.session_state.df = None

                    st.error(
                        f"❌ No fue posible actualizar los datos: {exc}"
                    )


    # --------------------------------------------------------
    # INFORMACIÓN DE FUENTE
    # --------------------------------------------------------

    st.divider()

    st.write(
        "**Objetivo:** San Nicolás de los Arroyos"
    )

    st.caption(
        "Fuente hidrométrica: Instituto Nacional del Agua (INA)"
    )


# ============================================================
# SI TODAVÍA NO HAY DATOS
# ============================================================

if st.session_state.df is None:

    st.info(
        "Presione **🔄 Actualizar INA** para iniciar "
        "la consulta online."
    )

    st.stop()


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

df = st.session_state.df.copy()


# ------------------------------------------------------------
# Verificar datetime
# ------------------------------------------------------------

if "datetime" not in df.columns:

    st.error(
        "Los datos recibidos del INA no contienen "
        "la columna 'datetime'."
    )

    st.stop()


df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce",
    utc=True
)


df = df.dropna(
    subset=["datetime"]
)


df = df.sort_values(
    "datetime"
)


# ============================================================
# VERIFICAR SAN NICOLÁS
# ============================================================

if "San Nicolás" not in df.columns:

    st.error(
        "Los datos recibidos no contienen la estación "
        "'San Nicolás'."
    )

    st.write(
        "Columnas recibidas:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()


# ------------------------------------------------------------
# Convertir estaciones a valores numéricos
# ------------------------------------------------------------

for station in STATIONS:

    if station in df.columns:

        df[station] = pd.to_numeric(
            df[station],
            errors="coerce"
        )


# ============================================================
# DATOS DE SAN NICOLÁS
# ============================================================

san_nicolas = df[
    "San Nicolás"
].dropna()


if san_nicolas.empty:

    st.error(
        "La estación San Nicolás no contiene "
        "observaciones válidas."
    )

    st.stop()


last = san_nicolas.iloc[-1]


# Cambio aproximado de las últimas 7 observaciones

if len(san_nicolas) >= 7:

    previous = san_nicolas.iloc[-7]

else:

    previous = san_nicolas.iloc[0]


change_7 = float(
    last - previous
)


# ============================================================
# PESTAÑAS
# ============================================================

tabs = st.tabs(
    [
        "📍 Estado",
        "🔮 Pronóstico",
        "🌧️ Lluvia",
        "🚦 Riesgo",
        "ℹ️ Metodología",
    ]
)


# ============================================================
# TAB 1 — ESTADO
# ============================================================

with tabs[0]:

    st.subheader(
        "Estado hidrométrico"
    )


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Nivel San Nicolás",
        f"{float(last):.2f} m"
    )


    col2.metric(
        "Cambio últimas 7 obs.",
        f"{change_7:+.2f} m"
    )


    # Villa Constitución

    if "Villa Constitución" in df.columns:

        vc = df[
            "Villa Constitución"
        ].dropna()

        if not vc.empty:

            col3.metric(
                "Villa Constitución",
                f"{float(vc.iloc[-1]):.2f} m"
            )

        else:

            col3.metric(
                "Villa Constitución",
                "—"
            )

    else:

        col3.metric(
            "Villa Constitución",
            "—"
        )


    # Rosario

    if "Rosario" in df.columns:

        rosario = df[
            "Rosario"
        ].dropna()

        if not rosario.empty:

            col4.metric(
                "Rosario",
                f"{float(rosario.iloc[-1]):.2f} m"
            )

        else:

            col4.metric(
                "Rosario",
                "—"
            )

    else:

        col4.metric(
            "Rosario",
            "—"
        )


    # ========================================================
    # GRÁFICO
    # ========================================================

    st.subheader(
        "Evolución del nivel del río Paraná"
    )


    fig = go.Figure()


    for station in STATIONS:

        if station not in df.columns:
            continue

        station_data = df[
            ["datetime", station]
        ].dropna()


        if station_data.empty:
            continue


        fig.add_trace(
            go.Scatter(
                x=station_data["datetime"],
                y=station_data[station],
                name=station,
                mode="lines"
            )
        )


    fig.update_layout(
        height=520,
        hovermode="x unified",
        xaxis_title="Fecha",
        yaxis_title="Nivel del río (m)",
        legend_title="Estación",
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # TABLA DE ESTACIONES
    # ========================================================

    st.subheader(
        "Último nivel disponible por estación"
    )


    latest_rows = []


    for station in STATIONS:

        if station not in df.columns:
            continue

        values = df[
            station
        ].dropna()


        if values.empty:
            continue


        latest_rows.append(
            {
                "Estación": station,
                "Nivel (m)": round(
                    float(values.iloc[-1]),
                    2
                )
            }
        )


    if latest_rows:

        st.dataframe(
            pd.DataFrame(latest_rows),
            hide_index=True,
            use_container_width=True
        )


    # ========================================================
    # ERRORES PARCIALES DEL INA
    # ========================================================

    if st.session_state.errors:

        with st.expander(
            "⚠️ Estaciones con problemas de consulta"
        ):

            for error in st.session_state.errors:

                st.warning(error)


# ============================================================
# TAB 2 — PRONÓSTICO
# ============================================================

with tabs[1]:

    st.subheader(
        "Pronóstico experimental"
    )


    st.info(
        "La predicción es experimental y se basa en "
        "niveles observados y tendencias de las estaciones "
        "del sistema. No reemplaza un pronóstico oficial."
    )


    if st.button(
        "🧠 Ejecutar modelo +24/+48/+72",
        type="primary"
    ):

        with st.spinner(
            "Entrenando modelo estadístico..."
        ):

            try:

                models, metrics = train(df)

                predictions = predict(
                    df,
                    models
                )

                st.session_state.pred = predictions
                st.session_state.met = metrics

                st.success(
                    "✅ Modelo ejecutado correctamente."
                )

            except Exception as exc:

                st.error(
                    f"❌ No fue posible ejecutar el modelo: {exc}"
                )


    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    predictions = st.session_state.pred
    metrics = st.session_state.met


    c1, c2, c3 = st.columns(3)


    for col, horizon in zip(
        [c1, c2, c3],
        [24, 48, 72]
    ):

        if horizon in predictions:

            col.metric(
                f"+{horizon} h",
                f"{float(predictions[horizon]):.2f} m"
            )

        else:

            col.metric(
                f"+{horizon} h",
                "—"
            )


    # --------------------------------------------------------
    # MÉTRICAS DEL MODELO
    # --------------------------------------------------------

    if metrics:

        st.subheader(
            "Desempeño del modelo"
        )


        metrics_rows = []


        for horizon, values in metrics.items():

            row = {
                "Horizonte": f"+{horizon} h"
            }

            if isinstance(values, dict):

                row.update(values)

            metrics_rows.append(row)


        if metrics_rows:

            st.dataframe(
                pd.DataFrame(metrics_rows),
                hide_index=True,
                use_container_width=True
            )


    # --------------------------------------------------------
    # PRONÓSTICO INA
    # --------------------------------------------------------

    st.divider()


    if st.button(
        "Consultar información del INA"
    ):

        try:

            st.json(
                forecast_meta()
            )

        except Exception as exc:

            st.error(
                f"No fue posible consultar la información: {exc}"
            )


# ============================================================
# TAB 3 — LLUVIA
# ============================================================

with tabs[2]:

    st.subheader(
        "Escenario experimental de lluvia"
    )


    st.warning(
        "Este módulo representa sensibilidad experimental. "
        "Todavía no constituye un modelo físico calibrado "
        "lluvia → escorrentía."
    )


    rain = {}


    c1, c2, c3 = st.columns(3)


    rain["Corrientes"] = c1.number_input(
        "Corrientes · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )


    rain["Goya"] = c2.number_input(
        "Goya · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=150.0
    )


    rain["Reconquista"] = c3.number_input(
        "Reconquista · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )


    c4, c5, c6 = st.columns(3)


    rain["Esquina"] = c4.number_input(
        "Esquina · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )


    rain["La Paz"] = c5.number_input(
        "La Paz · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0
    )


    rain["Paraná"] = c6.number_input(
        "Paraná · 72 h (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0
    )


    if st.button(
        "🌧️ Calcular impacto"
    ):

        try:

            scenario_result = run(
                float(last),
                rain
            )


            c1, c2, c3 = st.columns(3)


            c1.metric(
                "Escenario bajo",
                f"{scenario_result['Bajo']:.2f} m"
            )


            c2.metric(
                "Escenario central",
                f"{scenario_result['Central']:.2f} m"
            )


            c3.metric(
                "Escenario alto",
                f"{scenario_result['Alto']:.2f} m"
            )


        except Exception as exc:

            st.error(
                f"No fue posible calcular el escenario: {exc}"
            )


# ============================================================
# TAB 4 — RIESGO
# ============================================================

with tabs[3]:

    st.subheader(
        "🚦 Semáforo experimental de riesgo"
    )


    st.caption(
        "Los umbrales deben configurarse con referencias "
        "oficiales/locales antes de utilizarse como alerta."
    )


    threshold = st.number_input(
        "Umbral de referencia (m)",
        min_value=0.0,
        max_value=20.0,
        value=float(last + 0.5),
        step=0.10
    )


    if predictions:

        risk_rows = []


        for horizon, prediction in predictions.items():

            model_info = metrics.get(
                horizon,
                {}
            )


            rmse = model_info.get(
                "RMSE",
                0.20
            )


            probability = prob(
                prediction,
                threshold,
                rmse
            )


            if probability >= 0.75:

                state = "ALTO"

            elif probability >= 0.40:

                state = "MEDIO"

            else:

                state = "BAJO"


            risk_rows.append(
                {
                    "Horizonte": f"+{horizon} h",
                    "Predicción (m)": round(
                        float(prediction),
                        2
                    ),
                    "Probabilidad": (
                        f"{probability * 100:.0f}%"
                    ),
                    "Estado": state,
                }
            )


        st.dataframe(
            pd.DataFrame(risk_rows),
            hide_index=True,
            use_container_width=True
        )


    else:

        st.info(
            "Ejecute primero el modelo en la pestaña "
            "**Pronóstico**."
        )


# ============================================================
# TAB 5 — METODOLOGÍA
# ============================================================

with tabs[4]:

    st.subheader(
        "Metodología V9"
    )


    st.markdown(
        """
        ### Fuente de datos

        Los niveles observados utilizados por la aplicación
        provienen de la API pública del Instituto Nacional
        del Agua (INA).

        ### Modelo

        El modelo estadístico utiliza información temporal
        de las estaciones disponibles y variables derivadas
        de niveles, rezagos y cambios.

        ### Pronóstico

        Se generan estimaciones experimentales para:

        - +24 horas
        - +48 horas
        - +72 horas

        ### Riesgo

        El semáforo transforma la predicción y el error
        estimado del modelo en una probabilidad experimental
        respecto del umbral seleccionado.

        ### Lluvia

        El módulo de lluvia representa actualmente un
        escenario de sensibilidad. No debe interpretarse
        como un modelo hidrológico físico calibrado.

        ### Importante

        Esta aplicación NO constituye una alerta oficial.

        Para una versión operacional deberán incorporarse,
        entre otros elementos:

        - precipitación espacial;
        - pronóstico meteorológico;
        - caudales;
        - erogaciones;
        - tiempos de propagación;
        - características de la cuenca;
        - calibración con eventos históricos;
        - validación independiente.
        """
    )


# ============================================================
# DESCARGA DE DATOS
# ============================================================

st.divider()


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "⬇️ Descargar datos CSV",
    csv_data,
    "parana_san_nicolas_v9.csv",
    "text/csv"
)


st.caption(
    "PARANÁ · SAN NICOLÁS V9 · Consulta online"
)
