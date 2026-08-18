import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta, datetime


from src.ina import (
    observed,
)

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
# APP V11.7
# ============================================================

APP_VERSION = "V11.7"


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

FORECAST_DAYS = 15

TREND_DAYS = 30

STRESS_DAYS = 60

HISTORY_START = "1900-01-01"


# ============================================================
# ESCALA HIDROMÉTRICA
# SIEMPRE 0–7 m
# ============================================================

Y_MIN = 0.0

Y_MAX = 7.0

Y_STEP = 0.5


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        padding: 0.45rem 0.55rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 0.86rem;
    }

    h1 {
        margin-bottom: 0.15rem;
    }

    h2 {
        margin-top: 1.1rem;
    }

    h3 {
        margin-top: 0.9rem;
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

    El sistema integra:

    **nivel real INA · lluvia prevista · caudal · estaciones aguas arriba ·
    pronóstico recursivo diario · extremos históricos · escenarios severos**
    """
)


# ============================================================
# FUNCIONES AUXILIARES
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

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    if "value" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "value"
            ],
            errors="coerce",
        )

    elif "nivel" in x.columns:

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "nivel"
            ],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

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
# FORMATO
# ============================================================

def formato_numero(
    value,
    decimals=0,
):

    try:

        if pd.isna(
            value
        ):

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
# TENDENCIA CAUDAL
# ============================================================

def calcular_tendencia_caudal(
    df_caudal,
):

    resultado = {

        "actual":
            None,

        "delta_3":
            None,

        "delta_7":
            None,

        "pct_7":
            None,

        "pendiente":
            None,

        "estado":
            "Sin datos",
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

    q = (
        df_caudal
        .copy()
    )

    q[
        "caudal_m3s"
    ] = pd.to_numeric(
        q[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    q = q.dropna(
        subset=[
            "caudal_m3s"
        ]
    )

    if q.empty:

        return resultado

    valores = (
        q[
            "caudal_m3s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    actual = float(
        valores[
            -1
        ]
    )

    resultado[
        "actual"
    ] = actual


    # --------------------------------------------------------
    # 3 DÍAS
    # --------------------------------------------------------

    if len(
        valores
    ) >= 4:

        resultado[
            "delta_3"
        ] = (
            actual
            - float(
                valores[
                    -4
                ]
            )
        )


    # --------------------------------------------------------
    # 7 DÍAS
    # --------------------------------------------------------

    if len(
        valores
    ) >= 8:

        q7 = float(
            valores[
                -8
            ]
        )

        delta7 = (
            actual
            - q7
        )

        resultado[
            "delta_7"
        ] = delta7

        if q7 != 0:

            resultado[
                "pct_7"
            ] = (
                delta7
                / q7
                * 100.0
            )


    # --------------------------------------------------------
    # PENDIENTE
    # --------------------------------------------------------

    ultimos = valores[
        -min(
            7,
            len(
                valores
            ),
        ):
    ]

    if len(
        ultimos
    ) >= 3:

        x = np.arange(
            len(
                ultimos
            ),
            dtype=float,
        )

        try:

            pendiente = float(
                np.polyfit(
                    x,
                    ultimos,
                    1,
                )[0]
            )

        except Exception:

            pendiente = 0.0

    else:

        pendiente = 0.0

    resultado[
        "pendiente"
    ] = pendiente


    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

    umbral = max(
        abs(
            actual
        )
        * 0.002,
        1.0,
    )

    if pendiente > umbral:

        resultado[
            "estado"
        ] = "Creciente"

    elif pendiente < -umbral:

        resultado[
            "estado"
        ] = "Bajante"

    else:

        resultado[
            "estado"
        ] = "Estable"

    return resultado


# ============================================================
# TENDENCIA 30 DÍAS
# ============================================================

def calcular_tendencia_30_dias(
    df,
    forecast30,
):

    resultado = {

        "estado":
            "Sin datos",

        "nivel_actual":
            None,

        "nivel_dia_15":
            None,

        "nivel_dia_30":
            None,

        "cambio_30":
            None,

        "cambio_pct":
            None,

        "pendiente":
            None,
    }

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "nivel"
        not in df.columns
    ):

        return resultado

    niveles = (
        pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if niveles.empty:

        return resultado

    nivel_actual = float(
        niveles.iloc[
            -1
        ]
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

    serie = (
        forecast30
        .copy()
    )

    serie[
        "prediction"
    ] = pd.to_numeric(
        serie[
            "prediction"
        ],
        errors="coerce",
    )

    serie = serie.dropna(
        subset=[
            "prediction"
        ]
    )

    if serie.empty:

        return resultado

    if len(
        serie
    ) >= 15:

        nivel15 = float(
            serie[
                "prediction"
            ].iloc[
                14
            ]
        )

    else:

        nivel15 = float(
            serie[
                "prediction"
            ].iloc[
                -1
            ]
        )

    nivel30 = float(
        serie[
            "prediction"
        ].iloc[
            -1
        ]
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

    if len(
        serie
    ) >= 3:

        try:

            pendiente = float(
                np.polyfit(
                    np.arange(
                        len(
                            serie
                        )
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

            pendiente = 0.0

    else:

        pendiente = 0.0

    resultado[
        "pendiente"
    ] = pendiente

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

    hist = (
        df_historico
        .copy()
    )

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
        "nivel"
    ] = pd.to_numeric(
        hist[
            "nivel"
        ],
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

    hist[
        "mes_dia"
    ] = (
        hist[
            "datetime"
        ]
        .dt
        .strftime(
            "%m-%d"
        )
    )

    resumen = (
        hist
        .groupby(
            "mes_dia"
        )[
            "nivel"
        ]
        .agg(
            nivel_min_historico="min",
            nivel_max_historico="max",
            nivel_promedio_historico="mean",
            cantidad="count",
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

    fechas[
        "mes_dia"
    ] = (
        fechas[
            "datetime"
        ]
        .dt
        .strftime(
            "%m-%d"
        )
    )

    return fechas.merge(
        resumen,
        on="mes_dia",
        how="left",
    )


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
            .replace(
                "á",
                "a",
            )
            .replace(
                "é",
                "e",
            )
            .replace(
                "í",
                "i",
            )
            .replace(
                "ó",
                "o",
            )
            .replace(
                "ú",
                "u",
            )
            .replace(
                " ",
                "_",
            )
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
                upstream_history[
                    col
                ]
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
# ESCALA FIJA NIVEL
# ============================================================

def aplicar_escala_nivel(
    fig,
):

    fig.update_yaxes(
        title_text=(
            "Nivel hidrométrico (m)"
        ),
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
            forecast[
                "upper"
            ],
            errors="coerce",
        )

        lower = pd.to_numeric(
            forecast[
                "lower"
            ],
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
# AGREGAR BANDA DE INCERTIDUMBRE
# ============================================================

def agregar_banda_incertidumbre(
    fig,
    forecast,
    nombre=(
        "Banda experimental 80%"
    ),
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

    f = forecast.copy()

    f[
        "upper"
    ] = pd.to_numeric(
        f[
            "upper"
        ],
        errors="coerce",
    )

    f[
        "lower"
    ] = pd.to_numeric(
        f[
            "lower"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # LÍMITE SUPERIOR
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "upper"
            ],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.28)"
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # LÍMITE INFERIOR + RELLENO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "lower"
            ],
            mode="lines",
            line=dict(
                width=1,
                color=(
                    "rgba(255,165,0,0.28)"
                ),
            ),
            fill="tonexty",
            fillcolor=(
                "rgba(255,165,0,0.11)"
            ),
            name=nombre,
            hoverinfo="skip",
        )
    )


# ============================================================
# PRONÓSTICO CON HOVER COMPLETO
# ============================================================

def agregar_pronostico(
    fig,
    forecast,
    nombre,
    dash=None,
    width=3,
    marker_size=6,
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

    f = (
        forecast
        .copy()
    )

    margen = (
        obtener_margen_incertidumbre(
            f
        )
    )

    if len(
        margen
    ) == len(
        f
    ):

        f[
            "uncertainty_margin"
        ] = margen

    else:

        f[
            "uncertainty_margin"
        ] = np.nan

    for col in [
        "prediction",
        "lower",
        "upper",
        "nivel_base",
        "variacion_dia",
        "uncertainty_margin",
        "precip_mm",
        "caudal_m3s",
    ]:

        if col not in f.columns:

            f[
                col
            ] = np.nan

        f[
            col
        ] = pd.to_numeric(
            f[
                col
            ],
            errors="coerce",
        )

    customdata = np.column_stack(
        [
            f[
                "lower"
            ].to_numpy(),

            f[
                "upper"
            ].to_numpy(),

            f[
                "uncertainty_margin"
            ].to_numpy(),

            f[
                "nivel_base"
            ].to_numpy(),

            f[
                "variacion_dia"
            ].to_numpy(),

            f[
                "precip_mm"
            ].to_numpy(),

            f[
                "caudal_m3s"
            ].to_numpy(),
        ]
    )

    line_dict = {
        "width":
            width,
    }

    if dash:

        line_dict[
            "dash"
        ] = dash

    fig.add_trace(
        go.Scatter(
            x=f[
                "datetime"
            ],
            y=f[
                "prediction"
            ],
            customdata=
                customdata,
            mode=(
                "lines+markers"
            ),
            name=nombre,
            line=line_dict,
            marker=dict(
                size=
                    marker_size,
            ),
            hovertemplate=(
                "<b>%{x|%d/%m/%Y}</b>"
                "<br>Nivel estimado: "
                "%{y:.2f} m"
                "<br>Banda inferior: "
                "%{customdata[0]:.2f} m"
                "<br>Banda superior: "
                "%{customdata[1]:.2f} m"
                "<br>Incertidumbre: "
                "±%{customdata[2]:.2f} m"
                "<br>Nivel base: "
                "%{customdata[3]:.2f} m"
                "<br>Variación diaria: "
                "%{customdata[4]:+.3f} m"
                "<br>Lluvia: "
                "%{customdata[5]:.1f} mm"
                "<br>Caudal: "
                "%{customdata[6]:,.0f} m³/s"
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
    "Pronóstico principal: **15 días**"
)

st.sidebar.write(
    "Extensión recursiva: **30 días**"
)

st.sidebar.write(
    "Escenario histórico: **60 días**"
)

st.sidebar.subheader(
    "Escala"
)

st.sidebar.write(
    "Nivel hidrométrico: **0–7 m**"
)

st.sidebar.divider()

st.sidebar.caption(
    "Nivel: Instituto Nacional del Agua"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)

st.sidebar.caption(
    "Modelo: Random Forest recursivo"
)


# ============================================================
# VALIDAR FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser "
        "posterior a Hasta."
    )


# ============================================================
# ACTUALIZACIÓN
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado "
            "no es válido."
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

                df_ina, error_ina = (
                    observed(
                        inicio,
                        fin,
                    )
                )

            except Exception as exc:

                df_ina = (
                    pd.DataFrame()
                )

                error_ina = str(
                    exc
                )


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

            else:

                # ============================================
                # HISTORIAL COMPLETO
                # ============================================

                with st.spinner(
                    "Consultando historial completo "
                    "para máximos y mínimos..."
                ):

                    try:

                        (
                            df_hist_raw,
                            error_hist,
                        ) = observed(
                            HISTORY_START,
                            fin,
                        )

                        df_historico_total = (
                            preparar_datos(
                                df_hist_raw
                            )
                        )

                        if (
                            error_hist
                            or df_historico_total.empty
                        ):

                            df_historico_total = (
                                df.copy()
                            )

                            st.warning(
                                "No fue posible recuperar "
                                "todo el historial del INA. "
                                "Se utilizará el período "
                                "disponible."
                            )

                    except Exception:

                        df_historico_total = (
                            df.copy()
                        )

                        st.warning(
                            "No fue posible recuperar "
                            "todo el historial del INA."
                        )


                # ============================================
                # LLUVIA + CAUDAL
                # ============================================

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
                            "No fue posible obtener "
                            "todas las variables externas: "
                            f"{exc}"
                        )


                # ============================================
                # ESTACIONES AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando estaciones "
                    "aguas arriba..."
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
                            "No fue posible obtener "
                            "todas las estaciones "
                            f"aguas arriba: {exc}"
                        )


                # ============================================
                # ENTRENAR
                # ============================================

                with st.spinner(
                    "Entrenando modelo V11.7..."
                ):

                    try:

                        models, metrics = train(
                            df,
                            exog_history=
                                exog_history,
                            upstream_history=
                                upstream_history,
                        )


                        # ====================================
                        # IMPORTANTE
                        #
                        # GENERAMOS 30 DÍAS EN UNA SOLA
                        # EJECUCIÓN RECURSIVA.
                        # ====================================

                        forecast30 = predict(
                            df=df,
                            models=models,
                            days=
                                TREND_DAYS,
                            exog_future=
                                exog_future,
                            upstream_future=
                                None,
                        )


                        if (
                            forecast30 is None
                            or not isinstance(
                                forecast30,
                                pd.DataFrame,
                            )
                        ):

                            forecast30 = (
                                pd.DataFrame()
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
                            "el pronóstico: "
                            f"{exc}"
                        )


                # ============================================
                # GUARDAR SESIÓN
                # ============================================

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
                    "✅ Datos y modelo actualizados."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if (
    "datos"
    not in st.session_state
):

    st.info(
        "Seleccione un período y presione "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state[
        "datos"
    ]

    forecast = (
        st.session_state
        .get(
            "forecast",
            pd.DataFrame(),
        )
    )

    forecast30 = (
        st.session_state
        .get(
            "forecast30",
            pd.DataFrame(),
        )
    )

    df_historico_total = (
        st.session_state
        .get(
            "df_historico_total",
            df,
        )
    )

    models = (
        st.session_state
        .get(
            "models",
            {},
        )
    )

    metrics = (
        st.session_state
        .get(
            "metrics",
            {},
        )
    )

    exog_history = (
        st.session_state
        .get(
            "exog_history",
            pd.DataFrame(),
        )
    )

    exog_future = (
        st.session_state
        .get(
            "exog_future",
            pd.DataFrame(),
        )
    )

    exog_meta = (
        st.session_state
        .get(
            "exog_meta",
            {},
        )
    )

    upstream_history = (
        st.session_state
        .get(
            "upstream_history",
            pd.DataFrame(),
        )
    )

    upstream_meta = (
        st.session_state
        .get(
            "upstream_meta",
            {},
        )
    )

    actualizado = (
        st.session_state
        .get(
            "actualizado"
        )
    )


    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    ultima_fecha = (
        df[
            "datetime"
        ]
        .iloc[
            -1
        ]
    )

    nivel_actual = float(
        df[
            "nivel"
        ]
        .iloc[
            -1
        ]
    )


    # ========================================================
    # SITUACIÓN OBSERVADA
    # ========================================================

    st.subheader(
        "📊 Situación observada"
    )

    c1, c2, c3, c4 = (
        st.columns(
            4
        )
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


    try:

        if getattr(
            ultima_fecha,
            "tzinfo",
            None,
        ):

            fecha_obs = (
                ultima_fecha
                .tz_convert(
                    "America/Argentina/Buenos_Aires"
                )
                .strftime(
                    "%d/%m/%Y %H:%M"
                )
            )

        else:

            fecha_obs = (
                ultima_fecha
                .strftime(
                    "%d/%m/%Y"
                )
            )

    except Exception:

        fecha_obs = str(
            ultima_fecha
        )


    st.caption(
        "Última observación INA: "
        f"**{fecha_obs}** · "
        f"Registros utilizados: **{len(df)}**"
    )


    # ========================================================
    # ESTADO DEL SISTEMA
    # ========================================================

    estado_ina_ok = (
        isinstance(
            df,
            pd.DataFrame,
        )
        and not df.empty
    )

    estado_lluvia_ok = (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    )

    estado_caudal_ok = (
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

    s1, s2, s3, s4 = (
        st.columns(
            [
                1,
                1.2,
                1,
                1.5,
            ]
        )
    )

    with s1:

        st.caption(
            (
                "**INA** · ✅ Operativo"
                if estado_ina_ok
                else
                "**INA** · ⚠️ Sin datos"
            )
        )

    with s2:

        st.caption(
            (
                "**Lluvia** · ✅ Disponible"
                if estado_lluvia_ok
                else
                "**Lluvia** · ⚠️ Sin datos"
            )
        )

    with s3:

        st.caption(
            (
                "**Caudal** · ✅ Disponible"
                if estado_caudal_ok
                else
                "**Caudal** · ⚠️ Sin datos"
            )
        )

    with s4:

        st.caption(
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

    st.caption(
        "Cada jornada se calcula desde el nivel del día anterior "
        "incorporando lluvia, caudal y señales aguas arriba."
    )


    fig15 = go.Figure()


    # --------------------------------------------------------
    # OBSERVADO RECIENTE
    # --------------------------------------------------------

    obs = df.tail(
        45
    )

    fig15.add_trace(
        go.Scatter(
            x=obs[
                "datetime"
            ],
            y=obs[
                "nivel"
            ],
            mode="lines",
            name="Observado",
            line=dict(
                width=2,
                color=(
                    "rgba(120,120,120,0.55)"
                ),
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Nivel observado: "
                "%{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )


    # --------------------------------------------------------
    # NIVEL ACTUAL
    # --------------------------------------------------------

    fig15.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        line_color="black",
        annotation_text=(
            f"Actual {nivel_actual:.2f} m"
        ),
    )


    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        f15 = forecast.copy()

        f15[
            "datetime"
        ] = pd.to_datetime(
            f15[
                "datetime"
            ],
            errors="coerce",
        )


        # ----------------------------------------------------
        # BANDA 80 %
        # ----------------------------------------------------

        agregar_banda_incertidumbre(
            fig15,
            f15,
            nombre=(
                "Banda experimental 80%"
            ),
        )


        # ----------------------------------------------------
        # PRONÓSTICO
        # ----------------------------------------------------

        agregar_pronostico(
            fig15,
            f15,
            nombre=(
                "Pronóstico 1–15 días"
            ),
            width=3,
            marker_size=6,
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
            t=25,
            b=10,
        ),
    )

    fig15.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat="%d/%m/%Y",
    )

    aplicar_escala_nivel(
        fig15
    )

    st.plotly_chart(
        fig15,
        use_container_width=True,
    )


    # ========================================================
    # INFORMACIÓN DE INCERTIDUMBRE
    # ========================================================

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        margen = (
            obtener_margen_incertidumbre(
                forecast
            )
        )

        margen = (
            pd.to_numeric(
                margen,
                errors="coerce",
            )
            .dropna()
        )

        if not margen.empty:

            st.caption(
                "Banda experimental 80% · "
                f"Día 1: **±{margen.iloc[0]:.2f} m** · "
                f"Día {len(margen)}: "
                f"**±{margen.iloc[-1]:.2f} m** · "
                "margen máximo configurado: "
                "**±0,35 m**."
            )


    # ========================================================
    # TABLA 15 DÍAS
    # ========================================================

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        with st.expander(
            "🔎 Pronóstico diario detallado · 15 días"
        ):

            tabla15 = (
                forecast
                .copy()
            )

            tabla15[
                "Fecha"
            ] = (
                pd.to_datetime(
                    tabla15[
                        "datetime"
                    ]
                )
                .dt
                .strftime(
                    "%d/%m/%Y"
                )
            )

            tabla15[
                "Nivel base (m)"
            ] = pd.to_numeric(
                tabla15[
                    "nivel_base"
                ],
                errors="coerce",
            ).round(2)

            tabla15[
                "Variación diaria (m)"
            ] = pd.to_numeric(
                tabla15[
                    "variacion_dia"
                ],
                errors="coerce",
            ).round(3)

            tabla15[
                "Nivel previsto (m)"
            ] = pd.to_numeric(
                tabla15[
                    "prediction"
                ],
                errors="coerce",
            ).round(2)

            tabla15[
                "Inferior (m)"
            ] = pd.to_numeric(
                tabla15[
                    "lower"
                ],
                errors="coerce",
            ).round(2)

            tabla15[
                "Superior (m)"
            ] = pd.to_numeric(
                tabla15[
                    "upper"
                ],
                errors="coerce",
            ).round(2)

            tabla15[
                "Incertidumbre ±m"
            ] = (
                obtener_margen_incertidumbre(
                    tabla15
                )
                .round(2)
            )

            tabla15[
                "Lluvia (mm)"
            ] = pd.to_numeric(
                tabla15[
                    "precip_mm"
                ],
                errors="coerce",
            ).round(1)

            tabla15[
                "Caudal (m³/s)"
            ] = pd.to_numeric(
                tabla15[
                    "caudal_m3s"
                ],
                errors="coerce",
            ).round(0)


            st.dataframe(
                tabla15[
                    [
                        "Fecha",
                        "Nivel base (m)",
                        "Lluvia (mm)",
                        "Caudal (m³/s)",
                        "Variación diaria (m)",
                        "Nivel previsto (m)",
                        "Inferior (m)",
                        "Superior (m)",
                        "Incertidumbre ±m",
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

    estado30 = tendencia30.get(
        "estado",
        "Sin datos",
    )


    t1, t2, t3, t4 = (
        st.columns(
            4
        )
    )

    t1.metric(
        "Tendencia 30 días",
        estado30,
    )

    t2.metric(
        "Nivel día 15",
        (
            f"{nivel15:.2f} m"
            if nivel15
            is not None
            else "--"
        ),
    )

    t3.metric(
        "Nivel día 30",
        (
            f"{nivel30:.2f} m"
            if nivel30
            is not None
            else "--"
        ),
    )


    if cambio30 is not None:

        cambio_texto = (
            f"{cambio30:+.2f} m"
        )

        if pct30 is not None:

            cambio_texto += (
                f" ({pct30:+.1f}%)"
            )

    else:

        cambio_texto = "--"


    t4.metric(
        "Cambio vs. actual",
        cambio_texto,
    )


    # ========================================================
    # GRÁFICO 30 DÍAS
    # ========================================================

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        fig30 = go.Figure()

        obs30 = df.tail(
            30
        )


        # ----------------------------------------------------
        # OBSERVADO
        # ----------------------------------------------------

        fig30.add_trace(
            go.Scatter(
                x=obs30[
                    "datetime"
                ],
                y=obs30[
                    "nivel"
                ],
                mode="lines",
                name="Observado reciente",
                line=dict(
                    width=2,
                    color=(
                        "rgba(120,120,120,0.50)"
                    ),
                ),
            )
        )


        # ----------------------------------------------------
        # ACTUAL
        # ----------------------------------------------------

        fig30.add_hline(
            y=nivel_actual,
            line_dash="dash",
            line_width=2,
            line_color="black",
            annotation_text=(
                f"Actual {nivel_actual:.2f} m"
            ),
        )


        # ----------------------------------------------------
        # BANDA 30 DÍAS
        # ----------------------------------------------------

        agregar_banda_incertidumbre(
            fig30,
            forecast30,
            nombre=(
                "Banda experimental 80%"
            ),
        )


        # ----------------------------------------------------
        # 1–15
        # ----------------------------------------------------

        primeros15 = (
            forecast30
            .head(
                15
            )
            .copy()
        )

        agregar_pronostico(
            fig30,
            primeros15,
            nombre=(
                "Pronóstico 1–15 días"
            ),
            width=3,
            marker_size=5,
        )


        # ----------------------------------------------------
        # 16–30
        # ----------------------------------------------------

        extension = (
            forecast30
            .iloc[
                15:
            ]
            .copy()
        )

        if not extension.empty:

            # Añadir último punto del día 15
            # para continuidad visual.

            union = (
                forecast30
                .iloc[
                    14:
                ]
                .copy()
            )

            agregar_pronostico(
                fig30,
                union,
                nombre=(
                    "Extensión 16–30 días"
                ),
                dash="dot",
                width=3,
                marker_size=5,
            )


        fig30.update_layout(
            height=460,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.06,
            ),
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10,
            ),
        )

        fig30.update_xaxes(
            title_text="Fecha",
            tickformat="%d/%m",
        )

        aplicar_escala_nivel(
            fig30
        )

        st.plotly_chart(
            fig30,
            use_container_width=True,
        )


    st.caption(
        "Los 30 días se calculan en una única simulación recursiva. "
        "Cada nivel estimado se convierte en la base del día siguiente."
    )


    # ========================================================
    # NIVEL VS HISTÓRICOS
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


    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        fechas_env.extend(
            forecast30[
                "datetime"
            ].tolist()
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
            len(
                envolvente
            ),
            len(
                niveles_env
            ),
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


        # ----------------------------------------------------
        # MÁXIMO
        # ----------------------------------------------------

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
                    "Máximo histórico del día"
                ),
                line=dict(
                    color="crimson",
                    width=2,
                ),
            )
        )


        # ----------------------------------------------------
        # NIVEL ACTUAL / PROYECTADO
        # ----------------------------------------------------

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
                marker=dict(
                    size=5,
                ),
            )
        )


        # ----------------------------------------------------
        # MÍNIMO
        # ----------------------------------------------------

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
                    "Mínimo histórico del día"
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
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10,
            ),
        )

        fig_hist.update_xaxes(
            title_text="Fecha",
            tickformat="%d/%m",
        )

        aplicar_escala_nivel(
            fig_hist
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )

        st.caption(
            "Rojo: máximo histórico para la misma fecha. "
            "Azul: nivel real y trayectoria calculada. "
            "Verde: mínimo histórico para esa fecha."
        )

    else:

        st.info(
            "No hay historial suficiente "
            "para calcular la envolvente."
        )


    # ========================================================
    # ESCENARIOS 60 DÍAS
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
    # PRECIPITACIÓN
    # ========================================================

    st.subheader(
        "🌧️ Precipitación prevista · 15 días"
    )


    if (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    ):

        rain = (
            exog_future
            .head(
                FORECAST_DAYS
            )
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
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
        )


        r1, r2, r3 = (
            st.columns(
                3
            )
        )

        r1.metric(
            "Acumulado previsto",
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
                    >= 1.0
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
                name=(
                    "Precipitación prevista"
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>Lluvia: "
                    "%{y:.1f} mm"
                    "<extra></extra>"
                ),
            )
        )

        rain_fig.update_layout(
            height=310,
            yaxis_title=(
                "Precipitación (mm/día)"
            ),
            margin=dict(
                l=10,
                r=10,
                t=15,
                b=10,
            ),
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m"
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No se encuentra disponible "
            "la precipitación prevista."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal utilizado por el modelo"
    )


    if (
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
    ):

        q_hist = (
            exog_history
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


        q1, q2, q3, q4 = (
            st.columns(
                4
            )
        )


        q1.metric(
            "Caudal actual",
            (
                f"{formato_numero(tq['actual'], 0)} m³/s"
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


        if (
            tq[
                "delta_7"
            ]
            is not None
        ):

            texto7 = (
                f"{tq['delta_7']:+,.0f} m³/s"
            )

            if (
                tq[
                    "pct_7"
                ]
                is not None
            ):

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
                name=(
                    "Caudal observado"
                ),
            )
        )


        if (
            isinstance(
                exog_future,
                pd.DataFrame,
            )
            and not exog_future.empty
            and "caudal_m3s"
            in exog_future.columns
            and exog_future[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            q_fig.add_trace(
                go.Scatter(
                    x=exog_future[
                        "datetime"
                    ],
                    y=exog_future[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    line=dict(
                        dash="dash",
                    ),
                    name=(
                        "Proyección experimental"
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
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
        )

        q_fig.update_xaxes(
            tickformat="%d/%m"
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


        st.caption(
            "La tendencia del caudal representa únicamente "
            "la variable de entrada utilizada por el modelo."
        )

    else:

        st.info(
            "No se encuentra disponible "
            "una serie de caudal utilizable."
        )


    # ========================================================
    # VARIABLES UTILIZADAS
    # ========================================================

    with st.expander(
        "🌊 Variables utilizadas por el modelo"
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
    # IMPORTANCIA VARIABLES
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
            "🧠 Importancia de variables del modelo"
        ):

            top_imp = (
                importance
                .head(
                    20
                )
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
                    name="Importancia",
                )
            )

            imp_fig.update_layout(
                height=550,
                xaxis_title=(
                    "Importancia relativa"
                ),
                yaxis_title="Variable",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
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
    # DIAGNÓSTICO DEL MODELO
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

        model_version = models.get(
            "version",
            "Sin versión",
        )

        diagnostico = [

            {
                "Parámetro":
                    "Modelo",

                "Valor":
                    model_version,
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
                    "Filas entrenamiento",

                "Valor":
                    models.get(
                        "training_rows",
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
                    "Límite histórico Δ diario",

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
                    "Banda experimental",

                "Valor":
                    "80%",
            },

            {
                "Parámetro":
                    "Margen máximo banda",

                "Valor":
                    "±0,35 m",
            },

            {
                "Parámetro":
                    "Horizonte recursivo",

                "Valor":
                    "30 días",
            },
        ]


        st.dataframe(
            pd.DataFrame(
                diagnostico
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología, calidad y alcance"
    ):

        st.markdown(
            """
            **Pronóstico principal: 15 días**

            Parte de la última altura real disponible en San Nicolás.
            El modelo calcula la variación del nivel para el día
            siguiente utilizando el nivel actual, lluvia, caudal,
            comportamiento reciente y estaciones aguas arriba.

            Cada nivel pronosticado pasa a ser la base del día
            siguiente.

            **Extensión: 30 días**

            Utiliza la misma simulación recursiva, manteniendo
            continuidad entre los días 1 y 30.

            **Banda experimental 80%**

            La incertidumbre comienza cerca del error histórico del
            modelo y aumenta progresivamente con el horizonte. Está
            limitada a un máximo de ±0,35 m para evitar un abanico
            visual artificialmente amplio.

            **Escenario 60 días**

            Es una simulación histórica de estrés y no un pronóstico
            meteorológico convencional.

            **Escala**

            Todos los gráficos que representan nivel hidrométrico
            utilizan una escala fija de **0 a 7 m**.
            """
        )

        st.warning(
            "La plataforma es experimental. "
            "No reemplaza información, pronósticos, avisos "
            "ni alertas de organismos oficiales."
        )


    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        st.caption(
            "Última actualización del modelo: "
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
    "Los resultados de esta plataforma tienen carácter "
    "experimental e informativo. Ante situaciones de riesgo "
    "deben consultarse las comunicaciones oficiales."
)

st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Pronóstico: 15 días | "
    "Extensión: 30 días | "
    "Escenarios históricos: 60 días | "
    "Escala hidrométrica: 0–7 m"
)
