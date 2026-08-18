import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta, datetime


from src.ina import observed

from src.model import (
    train,
    predict,
)

from src.exogenous import (
    get_exogenous_data,
)

from src.upstream import (
    get_upstream_history,
)

from src.stress_ui import (
    render_stress_scenario,
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# APP V11.7.1
# ============================================================

APP_VERSION = "V11.7.1"

FORECAST_DAYS = 15
TREND_DAYS = 30
STRESS_DAYS = 60

HISTORY_START = "1900-01-01"


# ============================================================
# ESCALA HIDROMÉTRICA FIJA
# ============================================================

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        padding: 0.35rem 0.45rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.80rem;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 0.84rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · "
    "Plataforma pública experimental de monitoreo, "
    "pronóstico y análisis hidrométrico"
)

st.markdown(
    """
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**.

    **nivel real INA · lluvia prevista · caudal · estaciones aguas arriba ·
    pronóstico diario recursivo · extremos históricos · escenarios severos**
    """
)


# ============================================================
# NORMALIZACIÓN DE FECHAS
# ============================================================

def normalizar_fechas(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
    ):

        return pd.DataFrame()

    if df.empty:

        return df.copy()

    x = df.copy()

    if "datetime" not in x.columns:

        return x

    # --------------------------------------------------------
    # CONVERTIR A DATETIME REAL
    # --------------------------------------------------------

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    # --------------------------------------------------------
    # CONVERTIR A HORA ARGENTINA
    # Y QUITAR TIMEZONE PARA PLOTLY
    # --------------------------------------------------------

    try:

        x["datetime"] = (
            x["datetime"]
            .dt
            .tz_convert(
                "America/Argentina/Buenos_Aires"
            )
            .dt
            .tz_localize(None)
        )

    except Exception:

        try:

            x["datetime"] = (
                x["datetime"]
                .dt
                .tz_localize(None)
            )

        except Exception:

            pass

    x = (
        x
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# PREPARAR DATOS INA
# ============================================================

def preparar_datos(
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

    x = df.copy()

    if "datetime" not in x.columns:

        return pd.DataFrame()

    if "value" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["value"],
            errors="coerce",
        )

    elif "nivel" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["nivel"],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

    x = (
        x
        .dropna(
            subset=[
                "datetime",
                "nivel",
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

    return x


# ============================================================
# FORMATO NUMÉRICO
# ============================================================

def formato_numero(
    value,
    decimals=0,
):

    try:

        if pd.isna(value):

            return "--"

        text = (
            f"{float(value):,.{decimals}f}"
        )

        return (
            text
            .replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    except Exception:

        return "--"


# ============================================================
# ESCALA DE NIVEL
# ============================================================

def aplicar_escala_nivel(
    fig,
):

    fig.update_yaxes(
        title_text="Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        tick0=0,
        dtick=Y_STEP,
        autorange=False,
    )

    return fig


# ============================================================
# CONFIGURAR EJE DE FECHAS
# ============================================================

def aplicar_eje_fecha(
    fig,
    intervalo_dias=1,
    formato="%d/%m",
):

    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat=formato,
        dtick=(
            intervalo_dias
            * 24
            * 60
            * 60
            * 1000
        ),
        tickangle=0,
    )

    return fig


# ============================================================
# TENDENCIA CAUDAL
# ============================================================

def calcular_tendencia_caudal(
    df_caudal,
):

    resultado = {
        "actual": None,
        "delta_3": None,
        "delta_7": None,
        "pct_7": None,
        "pendiente": None,
        "estado": "Sin datos",
    }

    if (
        df_caudal is None
        or not isinstance(
            df_caudal,
            pd.DataFrame,
        )
        or df_caudal.empty
        or "caudal_m3s"
        not in df_caudal.columns
    ):

        return resultado

    q = df_caudal.copy()

    q["caudal_m3s"] = pd.to_numeric(
        q["caudal_m3s"],
        errors="coerce",
    )

    q = q.dropna(
        subset=[
            "caudal_m3s"
        ]
    )

    if q.empty:

        return resultado

    valores = q[
        "caudal_m3s"
    ].to_numpy(
        dtype=float
    )

    actual = float(
        valores[-1]
    )

    resultado["actual"] = actual


    if len(valores) >= 4:

        resultado["delta_3"] = (
            actual
            - float(
                valores[-4]
            )
        )


    if len(valores) >= 8:

        q7 = float(
            valores[-8]
        )

        resultado["delta_7"] = (
            actual
            - q7
        )

        if q7 != 0:

            resultado["pct_7"] = (
                resultado["delta_7"]
                / q7
                * 100.0
            )


    ultimos = valores[
        -min(
            7,
            len(valores),
        ):
    ]

    if len(ultimos) >= 3:

        try:

            pendiente = float(
                np.polyfit(
                    np.arange(
                        len(ultimos)
                    ),
                    ultimos,
                    1,
                )[0]
            )

        except Exception:

            pendiente = 0.0

    else:

        pendiente = 0.0


    resultado["pendiente"] = pendiente


    umbral = max(
        abs(actual)
        * 0.002,
        1.0,
    )

    if pendiente > umbral:

        resultado["estado"] = (
            "Creciente"
        )

    elif pendiente < -umbral:

        resultado["estado"] = (
            "Bajante"
        )

    else:

        resultado["estado"] = (
            "Estable"
        )

    return resultado


# ============================================================
# TENDENCIA 30 DÍAS
# ============================================================

def calcular_tendencia_30_dias(
    df,
    forecast30,
):

    resultado = {
        "estado": "Sin datos",
        "nivel_actual": None,
        "nivel_dia_15": None,
        "nivel_dia_30": None,
        "cambio_30": None,
        "cambio_pct": None,
        "pendiente": None,
    }

    if (
        df is None
        or df.empty
        or "nivel"
        not in df.columns
    ):

        return resultado

    niveles = (
        pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )
        .dropna()
    )

    if niveles.empty:

        return resultado

    nivel_actual = float(
        niveles.iloc[-1]
    )

    resultado[
        "nivel_actual"
    ] = nivel_actual


    if (
        forecast30 is None
        or not isinstance(
            forecast30,
            pd.DataFrame,
        )
        or forecast30.empty
        or "prediction"
        not in forecast30.columns
    ):

        return resultado


    serie = forecast30.copy()

    serie["prediction"] = pd.to_numeric(
        serie["prediction"],
        errors="coerce",
    )

    serie = serie.dropna(
        subset=[
            "prediction"
        ]
    )

    if serie.empty:

        return resultado


    if len(serie) >= 15:

        nivel15 = float(
            serie["prediction"]
            .iloc[14]
        )

    else:

        nivel15 = float(
            serie["prediction"]
            .iloc[-1]
        )


    nivel30 = float(
        serie["prediction"]
        .iloc[-1]
    )

    cambio30 = (
        nivel30
        - nivel_actual
    )


    resultado[
        "nivel_dia_15"
    ] = nivel15

    resultado[
        "nivel_dia_30"
    ] = nivel30

    resultado[
        "cambio_30"
    ] = cambio30


    if nivel_actual != 0:

        resultado[
            "cambio_pct"
        ] = (
            cambio30
            / nivel_actual
            * 100.0
        )


    if len(serie) >= 3:

        try:

            resultado[
                "pendiente"
            ] = float(
                np.polyfit(
                    np.arange(
                        len(serie)
                    ),
                    serie[
                        "prediction"
                    ].to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        except Exception:

            resultado[
                "pendiente"
            ] = 0.0

    else:

        resultado[
            "pendiente"
        ] = 0.0


    if cambio30 >= 0.30:

        resultado[
            "estado"
        ] = "Creciente"

    elif cambio30 <= -0.30:

        resultado[
            "estado"
        ] = "Bajante"

    else:

        resultado[
            "estado"
        ] = "Estable"

    return resultado


# ============================================================
# ENVOLVENTE HISTÓRICA
# ============================================================

def construir_envolvente_historica(
    df_historico,
    fechas_objetivo,
):

    if (
        df_historico is None
        or not isinstance(
            df_historico,
            pd.DataFrame,
        )
        or df_historico.empty
        or "datetime"
        not in df_historico.columns
        or "nivel"
        not in df_historico.columns
    ):

        return pd.DataFrame()


    hist = df_historico.copy()

    hist["datetime"] = pd.to_datetime(
        hist["datetime"],
        errors="coerce",
        utc=True,
    )

    hist["nivel"] = pd.to_numeric(
        hist["nivel"],
        errors="coerce",
    )

    hist = hist.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )


    if hist.empty:

        return pd.DataFrame()


    hist["mes_dia"] = (
        hist["datetime"]
        .dt
        .strftime(
            "%m-%d"
        )
    )


    resumen = (
        hist
        .groupby(
            "mes_dia"
        )["nivel"]
        .agg(
            nivel_min_historico="min",
            nivel_max_historico="max",
            nivel_promedio_historico="mean",
            registros="count",
        )
        .reset_index()
    )


    fechas = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    fechas_objetivo,
                    errors="coerce",
                    utc=True,
                )
        }
    )


    fechas["mes_dia"] = (
        fechas["datetime"]
        .dt
        .strftime(
            "%m-%d"
        )
    )


    resultado = fechas.merge(
        resumen,
        on="mes_dia",
        how="left",
    )


    resultado = normalizar_fechas(
        resultado
    )

    return resultado


# ============================================================
# ESTACIONES AGUAS ARRIBA
# ============================================================

def resumen_estaciones_upstream(
    upstream_meta,
    upstream_history,
):

    estaciones = []

    if not isinstance(
        upstream_meta,
        dict,
    ):

        return estaciones


    for estacion, info in upstream_meta.items():

        series_id = None
        proc_name = None
        disponible = False


        if isinstance(
            info,
            dict,
        ):

            series_id = info.get(
                "series_id"
            )

            proc_name = info.get(
                "proc_name"
            )


        nombre = (
            estacion
            .lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace(" ", "_")
        )


        col = (
            "nivel_"
            + nombre
        )


        if (
            isinstance(
                upstream_history,
                pd.DataFrame,
            )
            and col
            in upstream_history.columns
        ):

            disponible = bool(
                upstream_history[col]
                .notna()
                .any()
            )


        estaciones.append(
            {
                "Estación":
                    estacion,

                "Disponible":
                    disponible,

                "seriesId":
                    series_id,

                "Procedimiento":
                    proc_name,
            }
        )

    return estaciones


# ============================================================
# INCERTIDUMBRE
# ============================================================

def obtener_margen_incertidumbre(
    forecast,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return pd.Series(
            dtype=float
        )


    if (
        "uncertainty_margin"
        in forecast.columns
    ):

        return pd.to_numeric(
            forecast[
                "uncertainty_margin"
            ],
            errors="coerce",
        )


    if (
        "upper"
        in forecast.columns
        and "lower"
        in forecast.columns
    ):

        upper = pd.to_numeric(
            forecast["upper"],
            errors="coerce",
        )

        lower = pd.to_numeric(
            forecast["lower"],
            errors="coerce",
        )

        return (
            upper
            - lower
        ) / 2.0


    return pd.Series(
        np.nan,
        index=forecast.index,
        dtype=float,
    )


# ============================================================
# BANDA DE INCERTIDUMBRE
# ============================================================

def agregar_banda_incertidumbre(
    fig,
    forecast,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
        or "upper"
        not in forecast.columns
        or "lower"
        not in forecast.columns
    ):

        return


    f = normalizar_fechas(
        forecast
    )


    f["upper"] = pd.to_numeric(
        f["upper"],
        errors="coerce",
    )

    f["lower"] = pd.to_numeric(
        f["lower"],
        errors="coerce",
    )


    fig.add_trace(
        go.Scatter(
            x=f["datetime"],
            y=f["upper"],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.32)"
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=f["datetime"],
            y=f["lower"],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.32)"
                ),
            ),
            fill="tonexty",
            fillcolor=(
                "rgba(255,165,0,0.12)"
            ),
            name=(
                "Banda experimental 80%"
            ),
            hoverinfo="skip",
        )
    )


# ============================================================
# TRAZA DEL PRONÓSTICO
# ============================================================

def agregar_pronostico(
    fig,
    forecast,
    nombre,
    dash=None,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return


    f = normalizar_fechas(
        forecast
    )


    for col in [
        "prediction",
        "lower",
        "upper",
        "nivel_base",
        "variacion_dia",
        "precip_mm",
        "caudal_m3s",
    ]:

        if col not in f.columns:

            f[col] = np.nan

        f[col] = pd.to_numeric(
            f[col],
            errors="coerce",
        )


    f[
        "uncertainty_margin"
    ] = obtener_margen_incertidumbre(
        f
    )


    customdata = np.column_stack(
        [
            f["lower"],
            f["upper"],
            f["uncertainty_margin"],
            f["nivel_base"],
            f["variacion_dia"],
            f["precip_mm"],
            f["caudal_m3s"],
        ]
    )


    line_config = {
        "width":
            3,
    }


    if dash:

        line_config[
            "dash"
        ] = dash


    fig.add_trace(
        go.Scatter(
            x=f["datetime"],
            y=f["prediction"],
            customdata=customdata,
            mode="lines+markers",
            name=nombre,
            line=line_config,
            marker=dict(
                size=6,
            ),
            hovertemplate=(
                "<b>%{x|%d/%m/%Y}</b>"
                "<br>Nivel estimado: %{y:.2f} m"
                "<br>Inferior: %{customdata[0]:.2f} m"
                "<br>Superior: %{customdata[1]:.2f} m"
                "<br>Incertidumbre: ±%{customdata[2]:.2f} m"
                "<br>Nivel base: %{customdata[3]:.2f} m"
                "<br>Variación día: %{customdata[4]:+.3f} m"
                "<br>Lluvia: %{customdata[5]:.1f} mm"
                "<br>Caudal: %{customdata[6]:,.0f} m³/s"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Consulta"
)


fecha_hasta = date.today()

fecha_desde = (
    fecha_hasta
    - timedelta(
        days=120
    )
)


desde = st.sidebar.date_input(
    "Desde",
    value=fecha_desde,
    format="DD/MM/YYYY",
)


hasta = st.sidebar.date_input(
    "Hasta",
    value=fecha_hasta,
    format="DD/MM/YYYY",
)


actualizar = st.sidebar.button(
    "🔄 Actualizar modelo",
    use_container_width=True,
    type="primary",
)


st.sidebar.divider()


st.sidebar.subheader(
    "Ubicación"
)

st.sidebar.write(
    "San Nicolás de los Arroyos"
)


st.sidebar.subheader(
    "Horizontes"
)

st.sidebar.write(
    "Pronóstico: **15 días**"
)

st.sidebar.write(
    "Extensión: **30 días**"
)

st.sidebar.write(
    "Escenarios severos: **60 días**"
)


st.sidebar.subheader(
    "Escala"
)

st.sidebar.write(
    "**0–7 m** en todos los gráficos de nivel"
)


st.sidebar.divider()

st.sidebar.caption(
    "Datos hidrométricos: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)

st.sidebar.caption(
    "Modelo: V11.7"
)


# ============================================================
# VALIDACIÓN
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser "
        "posterior a Hasta."
    )


# ============================================================
# ACTUALIZAR DATOS
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "Período no válido."
        )

    else:

        inicio = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )


        # ====================================================
        # NIVEL INA
        # ====================================================

        with st.spinner(
            "Consultando nivel del INA..."
        ):

            try:

                (
                    df_ina,
                    error_ina,
                ) = observed(
                    inicio,
                    fin,
                )

            except Exception as exc:

                df_ina = pd.DataFrame()

                error_ina = str(exc)


        if error_ina:

            st.error(
                error_ina
            )

        else:

            df = preparar_datos(
                df_ina
            )


            if df.empty:

                st.error(
                    "No se obtuvieron "
                    "observaciones válidas."
                )

                st.stop()


            # =================================================
            # HISTÓRICO COMPLETO
            # =================================================

            with st.spinner(
                "Consultando historial completo..."
            ):

                try:

                    (
                        hist_raw,
                        hist_error,
                    ) = observed(
                        HISTORY_START,
                        fin,
                    )

                    df_historico_total = (
                        preparar_datos(
                            hist_raw
                        )
                    )


                    if (
                        hist_error
                        or df_historico_total.empty
                    ):

                        df_historico_total = (
                            df.copy()
                        )

                        st.warning(
                            "No se recuperó todo el historial. "
                            "Se utilizará el período disponible."
                        )

                except Exception:

                    df_historico_total = (
                        df.copy()
                    )

                    st.warning(
                        "No fue posible recuperar "
                        "el historial completo."
                    )


            # =================================================
            # LLUVIA + CAUDAL
            # =================================================

            with st.spinner(
                "Consultando lluvia y caudal..."
            ):

                try:

                    (
                        exog_history,
                        exog_future,
                        exog_meta,
                    ) = get_exogenous_data(
                        inicio,
                        fin,
                        TREND_DAYS,
                    )

                except Exception as exc:

                    exog_history = (
                        pd.DataFrame()
                    )

                    exog_future = (
                        pd.DataFrame()
                    )

                    exog_meta = {}

                    st.warning(
                        "Variables externas incompletas: "
                        f"{exc}"
                    )


            # =================================================
            # AGUAS ARRIBA
            # =================================================

            with st.spinner(
                "Consultando estaciones aguas arriba..."
            ):

                try:

                    (
                        upstream_history,
                        upstream_meta,
                    ) = get_upstream_history(
                        inicio,
                        fin,
                    )

                except Exception as exc:

                    upstream_history = (
                        pd.DataFrame()
                    )

                    upstream_meta = {}

                    st.warning(
                        "Estaciones aguas arriba incompletas: "
                        f"{exc}"
                    )


            # =================================================
            # ENTRENAMIENTO Y PRONÓSTICO
            # =================================================

            with st.spinner(
                "Entrenando modelo V11.7 "
                "y calculando 30 días..."
            ):

                try:

                    (
                        models,
                        metrics,
                    ) = train(
                        df,
                        exog_history=
                            exog_history,
                        upstream_history=
                            upstream_history,
                    )


                    forecast30 = predict(
                        df=df,
                        models=models,
                        days=TREND_DAYS,
                        exog_future=
                            exog_future,
                        upstream_future=
                            None,
                    )


                    forecast30 = (
                        normalizar_fechas(
                            forecast30
                        )
                    )


                    # =========================================
                    # VALIDAR FECHAS DEL PRONÓSTICO
                    # =========================================

                    if (
                        not forecast30.empty
                        and "datetime"
                        in forecast30.columns
                    ):

                        forecast30 = (
                            forecast30
                            .dropna(
                                subset=[
                                    "datetime"
                                ]
                            )
                            .sort_values(
                                "datetime"
                            )
                            .reset_index(
                                drop=True
                            )
                        )


                    forecast = (
                        forecast30
                        .head(
                            FORECAST_DAYS
                        )
                        .copy()
                    )


                except Exception as exc:

                    models = {}

                    metrics = {}

                    forecast = (
                        pd.DataFrame()
                    )

                    forecast30 = (
                        pd.DataFrame()
                    )

                    st.error(
                        "No fue posible generar "
                        f"el pronóstico: {exc}"
                    )


            # =================================================
            # SESIÓN
            # =================================================

            st.session_state[
                "datos"
            ] = df

            st.session_state[
                "forecast"
            ] = forecast

            st.session_state[
                "forecast30"
            ] = forecast30

            st.session_state[
                "df_historico_total"
            ] = df_historico_total

            st.session_state[
                "models"
            ] = models

            st.session_state[
                "metrics"
            ] = metrics

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
                "upstream_history"
            ] = upstream_history

            st.session_state[
                "upstream_meta"
            ] = upstream_meta

            st.session_state[
                "actualizado"
            ] = datetime.now()


            st.success(
                "✅ Datos y modelo actualizados correctamente."
            )


# ============================================================
# SIN DATOS EN SESIÓN
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione el período y presione "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state[
        "datos"
    ]

    forecast = st.session_state.get(
        "forecast",
        pd.DataFrame(),
    )

    forecast30 = st.session_state.get(
        "forecast30",
        pd.DataFrame(),
    )

    df_historico_total = (
        st.session_state.get(
            "df_historico_total",
            df,
        )
    )

    models = st.session_state.get(
        "models",
        {},
    )

    metrics = st.session_state.get(
        "metrics",
        {},
    )

    exog_history = (
        st.session_state.get(
            "exog_history",
            pd.DataFrame(),
        )
    )

    exog_future = (
        st.session_state.get(
            "exog_future",
            pd.DataFrame(),
        )
    )

    exog_meta = (
        st.session_state.get(
            "exog_meta",
            {},
        )
    )

    upstream_history = (
        st.session_state.get(
            "upstream_history",
            pd.DataFrame(),
        )
    )

    upstream_meta = (
        st.session_state.get(
            "upstream_meta",
            {},
        )
    )

    actualizado = (
        st.session_state.get(
            "actualizado"
        )
    )


    # ========================================================
    # NORMALIZAR TODAS LAS FECHAS PARA PLOTLY
    # ========================================================

    df_plot = normalizar_fechas(
        df
    )

    forecast = normalizar_fechas(
        forecast
    )

    forecast30 = normalizar_fechas(
        forecast30
    )

    exog_history_plot = (
        normalizar_fechas(
            exog_history
        )
    )

    exog_future_plot = (
        normalizar_fechas(
            exog_future
        )
    )

    upstream_history_plot = (
        normalizar_fechas(
            upstream_history
        )
    )


    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    nivel_actual = float(
        pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
        .iloc[-1]
    )


    ultima_fecha = (
        df_plot[
            "datetime"
        ]
        .iloc[-1]
    )


    # ========================================================
    # SITUACIÓN OBSERVADA
    # ========================================================

    st.subheader(
        "📊 Situación observada"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
    )


    c2.metric(
        "Mínimo período",
        f"{df['nivel'].min():.2f} m",
    )


    c3.metric(
        "Máximo período",
        f"{df['nivel'].max():.2f} m",
    )


    c4.metric(
        "Promedio período",
        f"{df['nivel'].mean():.2f} m",
    )


    st.caption(
        "Última observación: "
        f"**{ultima_fecha.strftime('%d/%m/%Y')}** · "
        f"Registros: **{len(df)}**"
    )


    # ========================================================
    # ESTADO SISTEMA
    # ========================================================

    estado_lluvia = (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    )


    estado_caudal = (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
        and "caudal_m3s"
        in exog_history.columns
        and exog_history[
            "caudal_m3s"
        ]
        .notna()
        .any()
    )


    estaciones_disponibles = 0


    if isinstance(
        upstream_history,
        pd.DataFrame,
    ):

        estaciones_disponibles = len(
            [
                c
                for c
                in upstream_history.columns
                if (
                    c.startswith(
                        "nivel_"
                    )
                    and upstream_history[
                        c
                    ]
                    .notna()
                    .any()
                )
            ]
        )


    st.caption(
        "🟢 **Estado del sistema**"
    )


    s1, s2, s3, s4 = st.columns(
        4
    )


    s1.caption(
        "**INA** · ✅ Operativo"
    )

    s2.caption(
        (
            "**Lluvia** · ✅ Disponible"
            if estado_lluvia
            else
            "**Lluvia** · ⚠️ Sin datos"
        )
    )

    s3.caption(
        (
            "**Caudal** · ✅ Disponible"
            if estado_caudal
            else
            "**Caudal** · ⚠️ Sin datos"
        )
    )

    s4.caption(
        "**Aguas arriba** · "
        f"✅ {estaciones_disponibles} estaciones"
    )


    st.divider()


    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Pronóstico experimental · 15 días"
    )


    fig15 = go.Figure()


    obs = df_plot.tail(
        45
    )


    fig15.add_trace(
        go.Scatter(
            x=obs["datetime"],
            y=obs["nivel"],
            mode="lines",
            name="Observado",
            line=dict(
                width=2,
                color=(
                    "rgba(120,120,120,0.55)"
                ),
            ),
        )
    )


    fig15.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        line_color="black",
        annotation_text=(
            f"Actual {nivel_actual:.2f} m"
        ),
    )


    if not forecast.empty:

        agregar_banda_incertidumbre(
            fig15,
            forecast,
        )

        agregar_pronostico(
            fig15,
            forecast,
            "Pronóstico 1–15 días",
        )


    fig15.update_layout(
        height=560,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.06,
        ),
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )


    aplicar_escala_nivel(
        fig15
    )


    # ========================================================
    # FECHAS CADA 2 DÍAS
    # ========================================================

    aplicar_eje_fecha(
        fig15,
        intervalo_dias=2,
        formato="%d/%m",
    )


    st.plotly_chart(
        fig15,
        use_container_width=True,
    )


    # ========================================================
    # INCERTIDUMBRE
    # ========================================================

    if not forecast.empty:

        margen = (
            obtener_margen_incertidumbre(
                forecast
            )
            .dropna()
        )

        if not margen.empty:

            st.caption(
                "Banda experimental 80% · "
                f"Día 1: **±{margen.iloc[0]:.2f} m** · "
                f"Día 15: **±{margen.iloc[-1]:.2f} m** · "
                "máximo: **±0,35 m**."
            )


    # ========================================================
    # VALIDACIÓN DE FECHAS
    # ========================================================

    with st.expander(
        "📅 Validación de fechas del pronóstico"
    ):

        if not forecast.empty:

            validacion = forecast.copy()

            validacion[
                "Día"
            ] = np.arange(
                1,
                len(
                    validacion
                )
                + 1,
            )

            validacion[
                "Fecha"
            ] = (
                validacion[
                    "datetime"
                ]
                .dt
                .strftime(
                    "%d/%m/%Y"
                )
            )

            validacion[
                "Nivel"
            ] = (
                pd.to_numeric(
                    validacion[
                        "prediction"
                    ],
                    errors="coerce",
                )
                .round(2)
            )


            st.dataframe(
                validacion[
                    [
                        "Día",
                        "Fecha",
                        "Nivel",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "Sin pronóstico disponible."
            )


    # ========================================================
    # TABLA 15 DÍAS
    # ========================================================

    if not forecast.empty:

        with st.expander(
            "🔎 Pronóstico diario detallado"
        ):

            tabla = forecast.copy()


            tabla["Fecha"] = (
                tabla["datetime"]
                .dt
                .strftime(
                    "%d/%m/%Y"
                )
            )


            tabla[
                "Nivel base"
            ] = (
                pd.to_numeric(
                    tabla[
                        "nivel_base"
                    ],
                    errors="coerce",
                )
                .round(2)
            )


            tabla[
                "Lluvia"
            ] = (
                pd.to_numeric(
                    tabla[
                        "precip_mm"
                    ],
                    errors="coerce",
                )
                .round(1)
            )


            tabla[
                "Caudal"
            ] = (
                pd.to_numeric(
                    tabla[
                        "caudal_m3s"
                    ],
                    errors="coerce",
                )
                .round(0)
            )


            tabla[
                "Variación"
            ] = (
                pd.to_numeric(
                    tabla[
                        "variacion_dia"
                    ],
                    errors="coerce",
                )
                .round(3)
            )


            tabla[
                "Nivel previsto"
            ] = (
                pd.to_numeric(
                    tabla[
                        "prediction"
                    ],
                    errors="coerce",
                )
                .round(2)
            )


            tabla[
                "Inferior"
            ] = (
                pd.to_numeric(
                    tabla[
                        "lower"
                    ],
                    errors="coerce",
                )
                .round(2)
            )


            tabla[
                "Superior"
            ] = (
                pd.to_numeric(
                    tabla[
                        "upper"
                    ],
                    errors="coerce",
                )
                .round(2)
            )


            tabla[
                "Incertidumbre ±"
            ] = (
                obtener_margen_incertidumbre(
                    tabla
                )
                .round(2)
            )


            st.dataframe(
                tabla[
                    [
                        "Fecha",
                        "Nivel base",
                        "Lluvia",
                        "Caudal",
                        "Variación",
                        "Nivel previsto",
                        "Inferior",
                        "Superior",
                        "Incertidumbre ±",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Tendencia extendida · 30 días"
    )


    tendencia30 = (
        calcular_tendencia_30_dias(
            df,
            forecast30,
        )
    )


    nivel15 = tendencia30.get(
        "nivel_dia_15"
    )

    nivel30 = tendencia30.get(
        "nivel_dia_30"
    )

    cambio30 = tendencia30.get(
        "cambio_30"
    )

    pct30 = tendencia30.get(
        "cambio_pct"
    )


    t1, t2, t3, t4 = st.columns(
        4
    )


    t1.metric(
        "Tendencia",
        tendencia30.get(
            "estado",
            "Sin datos",
        ),
    )


    t2.metric(
        "Día 15",
        (
            f"{nivel15:.2f} m"
            if nivel15
            is not None
            else "--"
        ),
    )


    t3.metric(
        "Día 30",
        (
            f"{nivel30:.2f} m"
            if nivel30
            is not None
            else "--"
        ),
    )


    if cambio30 is not None:

        texto_cambio = (
            f"{cambio30:+.2f} m"
        )

        if pct30 is not None:

            texto_cambio += (
                f" ({pct30:+.1f}%)"
            )

    else:

        texto_cambio = "--"


    t4.metric(
        "Cambio vs. actual",
        texto_cambio,
    )


    # ========================================================
    # GRÁFICO 30 DÍAS
    # ========================================================

    if not forecast30.empty:

        fig30 = go.Figure()


        obs30 = df_plot.tail(
            30
        )


        fig30.add_trace(
            go.Scatter(
                x=obs30[
                    "datetime"
                ],
                y=obs30[
                    "nivel"
                ],
                mode="lines",
                name="Observado",
                line=dict(
                    width=2,
                    color=(
                        "rgba(120,120,120,0.50)"
                    ),
                ),
            )
        )


        fig30.add_hline(
            y=nivel_actual,
            line_dash="dash",
            line_width=2,
            line_color="black",
            annotation_text=(
                f"Actual {nivel_actual:.2f} m"
            ),
        )


        agregar_banda_incertidumbre(
            fig30,
            forecast30,
        )


        primeros15 = (
            forecast30
            .head(15)
            .copy()
        )


        agregar_pronostico(
            fig30,
            primeros15,
            "Pronóstico 1–15 días",
        )


        if len(
            forecast30
        ) > 15:

            extension = (
                forecast30
                .iloc[
                    14:
                ]
                .copy()
            )


            agregar_pronostico(
                fig30,
                extension,
                "Extensión 16–30 días",
                dash="dot",
            )


        fig30.update_layout(
            height=470,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.06,
            ),
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )


        aplicar_escala_nivel(
            fig30
        )


        # ====================================================
        # FECHAS CADA 3 DÍAS
        # ====================================================

        aplicar_eje_fecha(
            fig30,
            intervalo_dias=3,
            formato="%d/%m",
        )


        st.plotly_chart(
            fig30,
            use_container_width=True,
        )


    st.caption(
        "Los 30 días se calculan en una única simulación "
        "recursiva desde la última medición real."
    )


    # ========================================================
    # EXTREMOS HISTÓRICOS
    # ========================================================

    st.subheader(
        "📏 Nivel diario vs. extremos históricos"
    )


    fechas_env = [
        ultima_fecha
    ]

    niveles_env = [
        nivel_actual
    ]


    if not forecast30.empty:

        fechas_env.extend(
            forecast30[
                "datetime"
            ]
            .tolist()
        )

        niveles_env.extend(
            pd.to_numeric(
                forecast30[
                    "prediction"
                ],
                errors="coerce",
            )
            .tolist()
        )


    envolvente = (
        construir_envolvente_historica(
            df_historico_total,
            fechas_env,
        )
    )


    if not envolvente.empty:

        cantidad = min(
            len(envolvente),
            len(niveles_env),
        )


        envolvente = (
            envolvente
            .head(
                cantidad
            )
            .copy()
        )


        envolvente[
            "nivel_dia"
        ] = niveles_env[
            :cantidad
        ]


        fig_hist = go.Figure()


        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_max_historico"
                ],
                mode="lines",
                name=(
                    "Máximo histórico"
                ),
                line=dict(
                    color="crimson",
                    width=2,
                ),
            )
        )


        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_dia"
                ],
                mode="lines+markers",
                name=(
                    "Nivel actual / proyectado"
                ),
                line=dict(
                    color="royalblue",
                    width=3,
                ),
            )
        )


        fig_hist.add_trace(
            go.Scatter(
                x=envolvente[
                    "datetime"
                ],
                y=envolvente[
                    "nivel_min_historico"
                ],
                mode="lines",
                name=(
                    "Mínimo histórico"
                ),
                line=dict(
                    color="seagreen",
                    width=2,
                ),
            )
        )


        fig_hist.update_layout(
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.06,
            ),
        )


        aplicar_escala_nivel(
            fig_hist
        )


        aplicar_eje_fecha(
            fig_hist,
            intervalo_dias=3,
            formato="%d/%m",
        )


        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )


    # ========================================================
    # 60 DÍAS
    # ========================================================

    render_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
    )


    # ========================================================
    # LLUVIA
    # ========================================================

    st.subheader(
        "🌧️ Precipitación prevista · 15 días"
    )


    if (
        not exog_future_plot.empty
        and "precip_mm"
        in exog_future_plot.columns
    ):

        rain = (
            exog_future_plot
            .head(15)
            .copy()
        )


        rain[
            "precip_mm"
        ] = (
            pd.to_numeric(
                rain[
                    "precip_mm"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )


        r1, r2, r3 = st.columns(
            3
        )


        r1.metric(
            "Acumulado",
            f"{rain['precip_mm'].sum():.1f} mm",
        )


        r2.metric(
            "Máximo diario",
            f"{rain['precip_mm'].max():.1f} mm",
        )


        r3.metric(
            "Días ≥ 1 mm",
            int(
                (
                    rain[
                        "precip_mm"
                    ]
                    >= 1
                )
                .sum()
            ),
        )


        rain_fig = go.Figure()


        rain_fig.add_trace(
            go.Bar(
                x=rain[
                    "datetime"
                ],
                y=rain[
                    "precip_mm"
                ],
                name="Lluvia",
            )
        )


        rain_fig.update_layout(
            height=300,
            yaxis_title=(
                "Precipitación (mm/día)"
            ),
        )


        aplicar_eje_fecha(
            rain_fig,
            intervalo_dias=2,
            formato="%d/%m",
        )


        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


    else:

        st.info(
            "Sin precipitación prevista."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal utilizado por el modelo"
    )


    if (
        not exog_history_plot.empty
        and "caudal_m3s"
        in exog_history_plot.columns
        and exog_history_plot[
            "caudal_m3s"
        ]
        .notna()
        .any()
    ):

        q_hist = (
            exog_history_plot
            .dropna(
                subset=[
                    "caudal_m3s"
                ]
            )
            .copy()
        )


        tq = calcular_tendencia_caudal(
            q_hist
        )


        q1, q2, q3, q4 = st.columns(
            4
        )


        q1.metric(
            "Actual",
            (
                f"{formato_numero(tq['actual'],0)} m³/s"
                if tq[
                    "actual"
                ]
                is not None
                else "--"
            ),
        )


        q2.metric(
            "Δ 3 días",
            (
                f"{tq['delta_3']:+,.0f} m³/s"
                if tq[
                    "delta_3"
                ]
                is not None
                else "--"
            ),
        )


        if tq[
            "delta_7"
        ] is not None:

            texto7 = (
                f"{tq['delta_7']:+,.0f} m³/s"
            )

            if tq[
                "pct_7"
            ] is not None:

                texto7 += (
                    f" ({tq['pct_7']:+.1f}%)"
                )

        else:

            texto7 = "--"


        q3.metric(
            "Δ 7 días",
            texto7,
        )


        q4.metric(
            "Tendencia",
            tq[
                "estado"
            ],
        )


        q_fig = go.Figure()


        q_fig.add_trace(
            go.Scatter(
                x=q_hist[
                    "datetime"
                ],
                y=q_hist[
                    "caudal_m3s"
                ],
                mode="lines",
                name="Caudal histórico",
            )
        )


        if (
            not exog_future_plot.empty
            and "caudal_m3s"
            in exog_future_plot.columns
            and exog_future_plot[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            q_fig.add_trace(
                go.Scatter(
                    x=exog_future_plot[
                        "datetime"
                    ],
                    y=exog_future_plot[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    line=dict(
                        dash="dash",
                    ),
                    name=(
                        "Proyección"
                    ),
                )
            )


        q_fig.update_layout(
            height=370,
            hovermode="x unified",
            yaxis_title="Caudal (m³/s)",
            legend=dict(
                orientation="h",
                y=1.05,
            ),
        )


        aplicar_eje_fecha(
            q_fig,
            intervalo_dias=7,
            formato="%d/%m",
        )


        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


    else:

        st.info(
            "Sin caudal utilizable."
        )


    # ========================================================
    # VARIABLES
    # ========================================================

    with st.expander(
        "🌊 Variables utilizadas"
    ):

        estaciones = (
            resumen_estaciones_upstream(
                upstream_meta,
                upstream_history,
            )
        )


        rows = [
            {
                "Variable":
                    "San Nicolás",

                "Estado":
                    "✓ Disponible",
            },

            {
                "Variable":
                    "Precipitación",

                "Estado":
                    (
                        "✓ Utilizada"
                        if models.get(
                            "uses_rain",
                            False,
                        )
                        else
                        "✗ No utilizada"
                    ),
            },

            {
                "Variable":
                    "Caudal",

                "Estado":
                    (
                        "✓ Utilizado"
                        if models.get(
                            "uses_caudal",
                            False,
                        )
                        else
                        "✗ No utilizado"
                    ),
            },
        ]


        for item in estaciones:

            rows.append(
                {
                    "Variable":
                        item[
                            "Estación"
                        ],

                    "Estado":
                        (
                            "✓ Disponible"
                            if item[
                                "Disponible"
                            ]
                            else
                            "✗ Sin datos"
                        ),
                }
            )


        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # IMPORTANCIA DEL MODELO
    # ========================================================

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
            "🧠 Importancia de variables"
        ):

            top_imp = (
                importance
                .head(20)
                .copy()
            )


            imp_fig = go.Figure()


            imp_fig.add_trace(
                go.Bar(
                    x=top_imp[
                        "importance"
                    ],
                    y=top_imp[
                        "feature"
                    ],
                    orientation="h",
                )
            )


            imp_fig.update_layout(
                height=550,
                xaxis_title=(
                    "Importancia relativa"
                ),
            )


            imp_fig.update_yaxes(
                autorange="reversed"
            )


            st.plotly_chart(
                imp_fig,
                use_container_width=True,
            )


    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🧪 Diagnóstico del modelo"
    ):

        rmse = metrics.get(
            "RMSE"
        )


        limite = models.get(
            "daily_change_limit"
        )


        diagnostico = pd.DataFrame(
            [
                {
                    "Parámetro":
                        "Modelo",

                    "Valor":
                        models.get(
                            "version",
                            "V11.7",
                        ),
                },

                {
                    "Parámetro":
                        "Observaciones",

                    "Valor":
                        models.get(
                            "observations",
                            "--",
                        ),
                },

                {
                    "Parámetro":
                        "RMSE",

                    "Valor":
                        (
                            f"{float(rmse):.3f} m"
                            if rmse
                            is not None
                            else "--"
                        ),
                },

                {
                    "Parámetro":
                        "Máximo Δ diario",

                    "Valor":
                        (
                            f"±{float(limite):.3f} m/día"
                            if limite
                            is not None
                            else "--"
                        ),
                },

                {
                    "Parámetro":
                        "Incertidumbre",

                    "Valor":
                        "Banda experimental 80%",
                },

                {
                    "Parámetro":
                        "Máximo banda",

                    "Valor":
                        "±0,35 m",
                },

                {
                    "Parámetro":
                        "Escala nivel",

                    "Valor":
                        "0–7 m",
                },
            ]
        )


        st.dataframe(
            diagnostico,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología"
    ):

        st.markdown(
            """
            **15 días**

            El pronóstico parte del último nivel real disponible
            de San Nicolás.

            Cada jornada incorpora el nivel del día anterior,
            precipitaciones, caudal y señales de estaciones
            aguas arriba.

            **30 días**

            Es la continuación de la misma simulación recursiva.
            El día 16 comienza exactamente desde el resultado
            del día 15.

            **Incertidumbre**

            Se utiliza una banda experimental del 80% que aumenta
            progresivamente con el horizonte y queda limitada
            a un máximo de ±0,35 m.

            **60 días**

            Corresponde a escenarios históricos severos y no a
            un pronóstico meteorológico convencional.

            **Escalas**

            Todos los gráficos de nivel utilizan una escala fija
            de 0 a 7 metros.
            """
        )


        st.warning(
            "La plataforma es experimental y no reemplaza "
            "pronósticos, avisos ni alertas oficiales."
        )


    if actualizado:

        st.caption(
            "Última actualización: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# FUENTES
# ============================================================

st.divider()


st.markdown(
    """
    **Fuentes**

    Nivel hidrométrico y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Predicción y escenarios: **modelo experimental propio**
    """
)


st.warning(
    "Los resultados tienen carácter experimental e informativo. "
    "Ante situaciones de riesgo deben consultarse las "
    "comunicaciones oficiales."
)


st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "15 días + 30 días + escenario 60 días | "
    "Escala hidrométrica 0–7 m"
)
