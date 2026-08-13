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
# VERSIÓN
# ============================================================

APP_VERSION = "V10.2"

FORECAST_DAYS = 15
TREND_DAYS = 30

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
# ESTILO GENERAL
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.90rem;
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
    f"{APP_VERSION} · Plataforma pública experimental "
    "de monitoreo y análisis hidrométrico"
)

st.markdown(
    """
    Seguimiento del nivel del río Paraná en
    **San Nicolás de los Arroyos**, utilizando observaciones
    del **Instituto Nacional del Agua (INA)** y un modelo
    experimental multivariable.

    El sistema combina, cuando están disponibles:

    **nivel observado · estaciones aguas arriba · caudal · precipitación prevista**
    """
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
    "Pronóstico experimental: **15 días**"
)

st.sidebar.write(
    "Tendencia extendida: **30 días**"
)

st.sidebar.write(
    "Escenario hipotético: **60 días**"
)


st.sidebar.subheader(
    "Escala"
)

st.sidebar.write(
    "Nivel hidrométrico: **0–7 m**"
)


st.sidebar.divider()


st.sidebar.caption(
    "Datos hidrométricos: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)

st.sidebar.caption(
    "Modelo: experimental"
)


# ============================================================
# PREPARAR DATOS
# ============================================================

def preparar_datos(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return pd.DataFrame()

    df = df.copy()

    if "datetime" not in df.columns:

        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True,
    )

    if "value" in df.columns:

        df["nivel"] = pd.to_numeric(
            df["value"],
            errors="coerce",
        )

    elif "nivel" in df.columns:

        df["nivel"] = pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )

    else:

        return pd.DataFrame()

    df = df.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    df = (
        df
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

    return df


# ============================================================
# TENDENCIA DEL CAUDAL
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

    resultado[
        "actual"
    ] = actual

    # --------------------------------------------------------
    # VARIACIÓN 3 DÍAS
    # --------------------------------------------------------

    if len(
        valores
    ) >= 4:

        resultado[
            "delta_3"
        ] = (
            actual
            - float(
                valores[-4]
            )
        )

    # --------------------------------------------------------
    # VARIACIÓN 7 DÍAS
    # --------------------------------------------------------

    if len(
        valores
    ) >= 8:

        q7 = float(
            valores[-8]
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
                * 100
            )

    # --------------------------------------------------------
    # PENDIENTE RECIENTE
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
            )
        )

        pendiente = float(
            np.polyfit(
                x,
                ultimos,
                1,
            )[0]
        )

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
# TENDENCIA EXTENDIDA 30 DÍAS
# ============================================================

def calcular_tendencia_30_dias(
    df,
    forecast,
):

    resultado = {
        "estado": "Sin datos",
        "nivel_actual": None,
        "nivel_dia_15": None,
        "nivel_dia_30": None,
        "cambio_30": None,
        "cambio_pct": None,
        "pendiente": None,
        "serie": pd.DataFrame(),
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

    if len(
        niveles
    ) < 10:

        return resultado

    nivel_actual = float(
        niveles.iloc[-1]
    )

    resultado[
        "nivel_actual"
    ] = nivel_actual

    # --------------------------------------------------------
    # PRONÓSTICO 15 DÍAS
    # --------------------------------------------------------

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
        or "prediction"
        not in forecast.columns
    ):

        return resultado

    pred = (
        pd.to_numeric(
            forecast[
                "prediction"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if pred.empty:

        return resultado

    nivel15 = float(
        pred.iloc[-1]
    )

    resultado[
        "nivel_dia_15"
    ] = nivel15

    # --------------------------------------------------------
    # PENDIENTE OBSERVADA
    # --------------------------------------------------------

    obs = (
        niveles
        .tail(
            30
        )
        .to_numpy(
            dtype=float
        )
    )

    if len(
        obs
    ) >= 5:

        pendiente_obs = float(
            np.polyfit(
                np.arange(
                    len(
                        obs
                    )
                ),
                obs,
                1,
            )[0]
        )

    else:

        pendiente_obs = 0.0

    # --------------------------------------------------------
    # PENDIENTE DEL PRONÓSTICO
    # --------------------------------------------------------

    pred_values = pred.to_numpy(
        dtype=float
    )

    if len(
        pred_values
    ) >= 5:

        pendiente_pred = float(
            np.polyfit(
                np.arange(
                    len(
                        pred_values
                    )
                ),
                pred_values,
                1,
            )[0]
        )

    else:

        pendiente_pred = 0.0

    # --------------------------------------------------------
    # COMBINAR
    # --------------------------------------------------------

    pendiente = (
        0.35
        * pendiente_obs
        + 0.65
        * pendiente_pred
    )

    pendiente = float(
        np.clip(
            pendiente,
            -0.08,
            0.08,
        )
    )

    resultado[
        "pendiente"
    ] = pendiente

    # --------------------------------------------------------
    # EXTENSIÓN 16–30
    # --------------------------------------------------------

    forecast_dates = pd.to_datetime(
        forecast[
            "datetime"
        ],
        errors="coerce",
    )

    ultima_fecha = forecast_dates.iloc[
        -1
    ]

    nivel = nivel15

    fechas = []
    valores = []

    for paso in range(
        1,
        16,
    ):

        amortiguacion = np.exp(
            -paso
            / 12.0
        )

        incremento = (
            pendiente
            * amortiguacion
        )

        nivel = float(
            np.clip(
                nivel
                + incremento,
                0.0,
                7.0,
            )
        )

        fechas.append(
            ultima_fecha
            + pd.Timedelta(
                days=paso
            )
        )

        valores.append(
            nivel
        )

    nivel30 = float(
        valores[-1]
    )

    cambio30 = (
        nivel30
        - nivel_actual
    )

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
            * 100
        )

    # --------------------------------------------------------
    # ESTADO
    # --------------------------------------------------------

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

    resultado[
        "serie"
    ] = pd.DataFrame(
        {
            "datetime": fechas,
            "prediction": valores,
        }
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
                "Estación": estacion,
                "Disponible": disponible,
                "seriesId": series_id,
                "Procedimiento": proc_name,
            }
        )

    return estaciones


# ============================================================
# VALIDACIÓN DE FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a Hasta."
    )


# ============================================================
# ACTUALIZAR MODELO
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado no es válido."
        )

    else:

        inicio = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )

        # ====================================================
        # DATOS INA
        # ====================================================

        with st.spinner(
            "Consultando observaciones del INA..."
        ):

            df_ina, error_ina = observed(
                inicio,
                fin,
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
                    "No se obtuvieron observaciones "
                    "hidrométricas válidas."
                )

            else:

                # ============================================
                # LLUVIA Y CAUDAL
                # ============================================

                with st.spinner(
                    "Consultando precipitación y caudal..."
                ):

                    try:

                        (
                            exog_history,
                            exog_future,
                            exog_meta,
                        ) = get_exogenous_data(
                            inicio,
                            fin,
                            FORECAST_DAYS,
                        )

                    except Exception:

                        exog_history = (
                            pd.DataFrame()
                        )

                        exog_future = (
                            pd.DataFrame()
                        )

                        exog_meta = {}

                        st.warning(
                            "No fue posible obtener todas "
                            "las variables externas."
                        )

                # ============================================
                # ESTACIONES AGUAS ARRIBA
                # ============================================

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

                    except Exception:

                        upstream_history = (
                            pd.DataFrame()
                        )

                        upstream_meta = {}

                        st.warning(
                            "No fue posible obtener todas "
                            "las estaciones aguas arriba."
                        )

                # ============================================
                # MODELO
                # ============================================

                with st.spinner(
                    "Entrenando modelo y generando pronóstico..."
                ):

                    try:

                        models, metrics = train(
                            df,
                            exog_history=exog_history,
                            upstream_history=upstream_history,
                        )

                        forecast = predict(
                            df,
                            models,
                            days=FORECAST_DAYS,
                            exog_future=exog_future,
                        )

                    except Exception as exc:

                        models = {}
                        metrics = {}

                        forecast = (
                            pd.DataFrame()
                        )

                        st.error(
                            "No fue posible generar "
                            f"el pronóstico: {exc}"
                        )

                # ============================================
                # GUARDAR EN SESIÓN
                # ============================================

                st.session_state[
                    "datos"
                ] = df

                st.session_state[
                    "forecast"
                ] = forecast

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
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione el período de análisis y presione "
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

    models = st.session_state.get(
        "models",
        {},
    )

    metrics = st.session_state.get(
        "metrics",
        {},
    )

    exog_history = st.session_state.get(
        "exog_history",
        pd.DataFrame(),
    )

    exog_future = st.session_state.get(
        "exog_future",
        pd.DataFrame(),
    )

    exog_meta = st.session_state.get(
        "exog_meta",
        {},
    )

    upstream_history = st.session_state.get(
        "upstream_history",
        pd.DataFrame(),
    )

    upstream_meta = st.session_state.get(
        "upstream_meta",
        {},
    )

    actualizado = st.session_state.get(
        "actualizado"
    )


    # ========================================================
    # SITUACIÓN OBSERVADA
    # ========================================================

    ultima_fecha = df[
        "datetime"
    ].iloc[-1]

    nivel_actual = float(
        df[
            "nivel"
        ].iloc[-1]
    )


    st.subheader(
        "📊 Situación observada"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Último nivel",
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
        "Promedio",
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

            fecha_obs = ultima_fecha.strftime(
                "%d/%m/%Y"
            )

    except Exception:

        fecha_obs = str(
            ultima_fecha
        )


    st.caption(
        f"Última observación INA: **{fecha_obs}** · "
        f"Registros utilizados: **{len(df)}**"
    )


    # ========================================================
    # ESTADO DEL SISTEMA - COMPACTO Y SIN HTML
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
        ].notna().any()
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
                    ].notna().any()
                )
            ]
        )


    st.caption(
        "🟢 **Estado del sistema**"
    )


    s1, s2, s3, s4 = st.columns(
        [
            1,
            1.2,
            1,
            1.5,
        ]
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
            f"✅ {estaciones_disponibles}/6 estaciones"
        )


    st.divider()


    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Pronóstico experimental · 15 días"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=df[
                "datetime"
            ],
            y=df[
                "nivel"
            ],
            mode="lines",
            name="Nivel observado",
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )


    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        f = forecast.copy()

        f[
            "datetime"
        ] = pd.to_datetime(
            f[
                "datetime"
            ],
            errors="coerce",
        )


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
                    width=0,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )


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
                    width=0,
                ),
                fill="tonexty",
                name="Intervalo experimental",
                hoverinfo="skip",
            )
        )


        fig.add_trace(
            go.Scatter(
                x=f[
                    "datetime"
                ],
                y=f[
                    "prediction"
                ],
                mode="lines+markers",
                line=dict(
                    dash="dash",
                    width=3,
                ),
                marker=dict(
                    size=6,
                ),
                name="Pronóstico multivariable",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Pronóstico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )


    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.03,
        ),
    )


    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat="%d/%m/%Y",
    )


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
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Tendencia extendida · 30 días"
    )


    tendencia30 = calcular_tendencia_30_dias(
        df,
        forecast,
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


    t1, t2, t3, t4 = st.columns(
        4
    )


    t1.metric(
        "Tendencia",
        estado30,
    )


    t2.metric(
        "Día 15",
        (
            f"{nivel15:.2f} m"
            if nivel15
            is not None
            else "Sin datos"
        ),
    )


    t3.metric(
        "Referencia día 30",
        (
            f"{nivel30:.2f} m"
            if nivel30
            is not None
            else "Sin datos"
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

        cambio_texto = (
            "Sin datos"
        )


    t4.metric(
        "Cambio vs. actual",
        cambio_texto,
    )


    serie30 = tendencia30.get(
        "serie",
        pd.DataFrame(),
    )


    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
        and isinstance(
            serie30,
            pd.DataFrame,
        )
        and not serie30.empty
    ):

        fig30 = go.Figure()


        obs30 = df.tail(
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
                name="Observado reciente",
            )
        )


        fig30.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico días 1–15",
            )
        )


        fig30.add_trace(
            go.Scatter(
                x=serie30[
                    "datetime"
                ],
                y=serie30[
                    "prediction"
                ],
                mode="lines+markers",
                line=dict(
                    dash="dot",
                ),
                name="Tendencia días 16–30",
            )
        )


        fig30.update_layout(
            height=390,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.05,
            ),
        )


        fig30.update_xaxes(
            title_text="Fecha",
            tickformat="%d/%m",
        )


        fig30.update_yaxes(
            title_text=(
                "Nivel hidrométrico (m)"
            ),
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )


        st.plotly_chart(
            fig30,
            use_container_width=True,
        )


    st.caption(
        "Del día 1 al 15 se utiliza el modelo experimental "
        "multivariable. Entre los días 16 y 30 se muestra "
        "una extrapolación amortiguada de tendencia. "
        "No constituye un pronóstico diario equivalente "
        "al horizonte de 15 días."
    )


    # ========================================================
    # ESCENARIO HIPOTÉTICO 60 DÍAS
    # ========================================================

    render_stress_scenario(
        df=df,
        models=models,
        exog_history=exog_history,
        upstream_history=upstream_history,
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

        rain = exog_future.copy()

        rain[
            "precip_mm"
        ] = pd.to_numeric(
            rain[
                "precip_mm"
            ],
            errors="coerce",
        ).fillna(
            0
        )


        r1, r2, r3 = st.columns(
            3
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
                    >= 1
                ).sum()
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
                name="Precipitación prevista",
            )
        )


        rain_fig.update_layout(
            height=320,
            yaxis_title="Precipitación (mm/día)",
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
        ].notna().any()
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


        q1, q2, q3, q4 = st.columns(
            4
        )


        q1.metric(
            "Caudal actual",
            (
                f"{tq['actual']:,.0f} m³/s"
                if tq[
                    "actual"
                ]
                is not None
                else "Sin datos"
            ),
        )


        q2.metric(
            "Variación 3 días",
            (
                f"{tq['delta_3']:+,.0f} m³/s"
                if tq[
                    "delta_3"
                ]
                is not None
                else "Sin datos"
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

            texto7 = (
                "Sin datos"
            )


        q3.metric(
            "Variación 7 días",
            texto7,
        )


        q4.metric(
            "Tendencia del caudal",
            tq[
                "estado"
            ],
        )


        st.caption(
            "La tendencia creciente, estable o bajante se refiere "
            "exclusivamente al caudal de entrada utilizado por "
            "el modelo. No implica por sí sola igual comportamiento "
            "del nivel hidrométrico en San Nicolás."
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
                name="Caudal observado",
            )
        )


        if (
            isinstance(
                exog_future,
                pd.DataFrame,
            )
            and "caudal_m3s"
            in exog_future.columns
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
            height=380,
            hovermode="x unified",
            yaxis_title="Caudal (m³/s)",
        )


        q_fig.update_xaxes(
            tickformat="%d/%m"
        )


        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


    else:

        st.info(
            "No se encuentra disponible una serie "
            "de caudal utilizable."
        )


    # ========================================================
    # VARIABLES UTILIZADAS
    # ========================================================

    with st.expander(
        "🌊 Variables utilizadas por el modelo"
    ):

        estaciones = resumen_estaciones_upstream(
            upstream_meta,
            upstream_history,
        )


        rows = [
            {
                "Variable": "San Nicolás",
                "Estado": "✓ Disponible",
            },
            {
                "Variable": "Precipitación",
                "Estado": (
                    "✓ Utilizada"
                    if models.get(
                        "uses_rain",
                        False,
                    )
                    else "✗ No utilizada"
                ),
            },
            {
                "Variable": "Caudal",
                "Estado": (
                    "✓ Utilizado"
                    if models.get(
                        "uses_caudal",
                        False,
                    )
                    else "✗ No utilizado"
                ),
            },
        ]


        for item in estaciones:

            rows.append(
                {
                    "Variable": item[
                        "Estación"
                    ],
                    "Estado": (
                        "✓ Disponible"
                        if item[
                            "Disponible"
                        ]
                        else "✗ Sin datos"
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
    # IMPORTANCIA DE VARIABLES
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

            top_imp = importance.head(
                20
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
                height=600,
                xaxis_title="Importancia relativa",
                yaxis_title="Variable",
            )


            imp_fig.update_yaxes(
                autorange="reversed"
            )


            st.plotly_chart(
                imp_fig,
                use_container_width=True,
            )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología, calidad y alcance"
    ):

        st.write(
            "**Pronóstico principal:** 15 días"
        )

        st.write(
            "**Tendencia extendida:** 30 días"
        )

        st.write(
            "**Escenario hipotético:** 60 días"
        )

        st.write(
            "**Modelo:** Random Forest Regressor"
        )


        rmse = metrics.get(
            "RMSE"
        )


        if rmse is not None:

            st.write(
                "**RMSE de validación histórica:** "
                f"{float(rmse):.3f} m"
            )


        st.markdown(
            """
            El horizonte de **15 días** utiliza información
            hidrométrica y meteorológica disponible.

            La sección de **30 días** es un indicador de tendencia
            de mediano plazo y no un pronóstico meteorológico completo.

            El escenario de **60 días** es una simulación de tipo
            *qué pasa si*, construida con condiciones históricamente
            elevadas de precipitación, caudal y niveles aguas arriba.

            El RMSE corresponde a validación histórica y no representa
            una garantía de error máximo para un pronóstico futuro.
            """
        )


        st.warning(
            "Esta plataforma es experimental. "
            "No reemplaza información, avisos, alertas "
            "ni pronósticos emitidos por organismos oficiales."
        )


    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        st.caption(
            "Última actualización de la plataforma: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# FUENTES
# ============================================================

st.divider()


st.markdown(
    """
    **Fuentes**

    Datos hidrométricos y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Predicción y escenarios: **modelo experimental propio**
    """
)


st.warning(
    "Los resultados de esta plataforma tienen carácter "
    "experimental e informativo. Ante situaciones de riesgo "
    "deben consultarse las comunicaciones oficiales de las "
    "autoridades y organismos competentes."
)


st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Pronóstico experimental: 15 días | "
    "Tendencia extendida: 30 días | "
    "Escenario hipotético: 60 días"
)
