import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta

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


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# CONSTANTES
# ============================================================

FORECAST_DAYS = 15

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    "V9 · Monitoreo hidrométrico y pronóstico experimental multivariable"
)

st.markdown(
    """
    Esta plataforma utiliza datos hidrométricos observados del
    **Instituto Nacional del Agua (INA)** y un modelo experimental
    para estimar la evolución del nivel del río Paraná en
    **San Nicolás de los Arroyos**.

    El modelo incorpora, cuando están disponibles, el nivel local,
    niveles aguas arriba, precipitación prevista y caudal.
    """
)


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header(
    "Consulta online"
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
)


st.sidebar.divider()


st.sidebar.subheader(
    "Objetivo"
)

st.sidebar.write(
    "San Nicolás de los Arroyos"
)


st.sidebar.subheader(
    "Pronóstico"
)

st.sidebar.write(
    "Pronóstico experimental: 15 días"
)

st.sidebar.write(
    "Tendencia extendida: 30 días"
)


st.sidebar.subheader(
    "Escala hidrométrica"
)

st.sidebar.write(
    "0–7 m"
)


# ============================================================
# PREPARAR DATOS INA
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
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TENDENCIA DEL CAUDAL
# ============================================================

def calcular_tendencia_caudal(df_caudal):

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

    # --------------------------------------------------------
    # CAMBIO 3 DÍAS
    # --------------------------------------------------------

    if len(valores) >= 4:

        q3 = float(
            valores[-4]
        )

        resultado["delta_3"] = (
            actual
            - q3
        )

    # --------------------------------------------------------
    # CAMBIO 7 DÍAS
    # --------------------------------------------------------

    if len(valores) >= 8:

        q7 = float(
            valores[-8]
        )

        delta7 = (
            actual
            - q7
        )

        resultado["delta_7"] = delta7

        if q7 != 0:

            resultado["pct_7"] = (
                delta7
                / q7
                * 100
            )

    # --------------------------------------------------------
    # PENDIENTE ÚLTIMOS 7 REGISTROS
    # --------------------------------------------------------

    ultimos = valores[
        -min(
            7,
            len(valores),
        ):
    ]

    if len(ultimos) >= 3:

        x = np.arange(
            len(ultimos)
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

    umbral = max(
        abs(actual) * 0.002,
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
    """
    Indicador extendido de tendencia.

    Usa:
    - evolución observada reciente;
    - pronóstico experimental de los primeros 15 días.

    Los días 16-30 son una extrapolación amortiguada de tendencia.
    NO son un pronóstico meteorológico/hidrológico equivalente
    al horizonte de 15 días.
    """

    resultado = {
        "estado": "Sin datos",
        "nivel_actual": None,
        "nivel_dia_15": None,
        "nivel_dia_30": None,
        "cambio_30": None,
        "cambio_pct": None,
        "pendiente": None,
        "confianza": "Baja",
        "serie": pd.DataFrame(),
    }

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "nivel" not in df.columns
    ):

        return resultado

    niveles_obs = (
        pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )
        .dropna()
    )

    if len(niveles_obs) < 10:

        return resultado

    nivel_actual = float(
        niveles_obs.iloc[-1]
    )

    resultado[
        "nivel_actual"
    ] = nivel_actual

    # ========================================================
    # TOMAR PRONÓSTICO 15 DÍAS
    # ========================================================

    if (
        forecast is not None
        and isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
        and "prediction"
        in forecast.columns
    ):

        pred = pd.to_numeric(
            forecast["prediction"],
            errors="coerce",
        ).dropna()

    else:

        pred = pd.Series(
            dtype=float
        )

    if len(pred) == 0:

        return resultado

    nivel_dia_15 = float(
        pred.iloc[-1]
    )

    resultado[
        "nivel_dia_15"
    ] = nivel_dia_15

    # ========================================================
    # TENDENCIA OBSERVADA RECIENTE
    # ========================================================

    observados_recientes = (
        niveles_obs
        .tail(30)
        .to_numpy(
            dtype=float
        )
    )

    if len(
        observados_recientes
    ) >= 5:

        x_obs = np.arange(
            len(
                observados_recientes
            )
        )

        pendiente_obs = float(
            np.polyfit(
                x_obs,
                observados_recientes,
                1,
            )[0]
        )

    else:

        pendiente_obs = 0.0

    # ========================================================
    # TENDENCIA DEL PRONÓSTICO 15 DÍAS
    # ========================================================

    pred_values = pred.to_numpy(
        dtype=float
    )

    if len(pred_values) >= 5:

        x_pred = np.arange(
            len(pred_values)
        )

        pendiente_pred = float(
            np.polyfit(
                x_pred,
                pred_values,
                1,
            )[0]
        )

    else:

        pendiente_pred = 0.0

    # ========================================================
    # COMBINACIÓN DE TENDENCIAS
    # ========================================================

    # Mayor peso al pronóstico reciente
    pendiente_combinada = (
        0.35
        * pendiente_obs
        + 0.65
        * pendiente_pred
    )

    # Evitar extrapolaciones exageradas
    pendiente_maxima = 0.08

    pendiente_combinada = float(
        np.clip(
            pendiente_combinada,
            -pendiente_maxima,
            pendiente_maxima,
        )
    )

    resultado[
        "pendiente"
    ] = pendiente_combinada

    # ========================================================
    # EXTRAPOLACIÓN DÍAS 16 A 30
    # ========================================================

    fechas_extra = []

    valores_extra = []

    if (
        "datetime"
        in forecast.columns
    ):

        ultima_fecha_15 = pd.to_datetime(
            forecast[
                "datetime"
            ].iloc[-1],
            errors="coerce",
        )

    else:

        ultima_fecha_15 = pd.Timestamp.today()

    nivel = nivel_dia_15

    for dia in range(
        16,
        31,
    ):

        paso = (
            dia
            - 15
        )

        # La tendencia pierde fuerza progresivamente
        amortiguacion = np.exp(
            -paso / 12.0
        )

        incremento = (
            pendiente_combinada
            * amortiguacion
        )

        nivel = (
            nivel
            + incremento
        )

        nivel = float(
            np.clip(
                nivel,
                0.0,
                7.0,
            )
        )

        fecha = (
            ultima_fecha_15
            + pd.Timedelta(
                days=paso
            )
        )

        fechas_extra.append(
            fecha
        )

        valores_extra.append(
            nivel
        )

    if valores_extra:

        nivel_dia_30 = float(
            valores_extra[-1]
        )

    else:

        nivel_dia_30 = nivel_dia_15

    resultado[
        "nivel_dia_30"
    ] = nivel_dia_30

    cambio_30 = (
        nivel_dia_30
        - nivel_actual
    )

    resultado[
        "cambio_30"
    ] = cambio_30

    if nivel_actual != 0:

        resultado[
            "cambio_pct"
        ] = (
            cambio_30
            / nivel_actual
            * 100
        )

    # ========================================================
    # CLASIFICACIÓN
    # ========================================================

    if cambio_30 >= 0.30:

        estado = "Tendencia creciente"

    elif cambio_30 <= -0.30:

        estado = "Tendencia bajante"

    else:

        estado = "Tendencia estable"

    resultado[
        "estado"
    ] = estado

    # ========================================================
    # SERIE EXTENDIDA
    # ========================================================

    resultado[
        "serie"
    ] = pd.DataFrame(
        {
            "datetime": fechas_extra,
            "prediction": valores_extra,
        }
    )

    return resultado


# ============================================================
# RESUMEN ESTACIONES AGUAS ARRIBA
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

        disponible = False
        series_id = None
        proc_name = None

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

        nombre_col = (
            "nivel_"
            + estacion
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

        if (
            isinstance(
                upstream_history,
                pd.DataFrame,
            )
            and nombre_col
            in upstream_history.columns
        ):

            disponible = bool(
                upstream_history[
                    nombre_col
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
# ACTUALIZACIÓN DEL MODELO
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
        # SAN NICOLÁS
        # ====================================================

        with st.spinner(
            "Consultando nivel de San Nicolás..."
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
                    "No se obtuvieron datos hidrométricos válidos."
                )

            else:

                # ============================================
                # LLUVIA + CAUDAL
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

                    except Exception as exc:

                        exog_history = pd.DataFrame()
                        exog_future = pd.DataFrame()
                        exog_meta = {}

                        st.warning(
                            "No fue posible obtener todas "
                            "las variables externas: "
                            f"{exc}"
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

                    except Exception as exc:

                        upstream_history = pd.DataFrame()
                        upstream_meta = {}

                        st.warning(
                            "No fue posible obtener las "
                            "estaciones aguas arriba: "
                            f"{exc}"
                        )

                # ============================================
                # MODELO
                # ============================================

                with st.spinner(
                    "Entrenando modelo hidrológico multivariable..."
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
                        forecast = pd.DataFrame()

                        st.error(
                            f"Error del modelo: {exc}"
                        )

                # ============================================
                # GUARDAR
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

                st.success(
                    "✅ Datos y modelo actualizados correctamente."
                )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Presione **Actualizar modelo** para consultar "
        "los datos y generar el pronóstico."
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


    # ========================================================
    # SITUACIÓN OBSERVADA
    # ========================================================

    st.subheader(
        "📊 Situación observada"
    )

    nivel_actual = float(
        df[
            "nivel"
        ].iloc[-1]
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


    # ========================================================
    # GRÁFICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Nivel observado y pronóstico a 15 días"
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
            mode="lines+markers",
            name="Nivel observado",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=4,
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

        # Banda superior
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

        # Banda inferior
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
                name=(
                    "Pronóstico · modelo "
                    "hidrológico multivariable"
                ),
                line=dict(
                    dash="dash",
                    width=3,
                ),
                marker=dict(
                    size=7,
                ),
            )
        )

    fig.update_layout(
        height=620,
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
    # NUEVO CUADRO: TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Tendencia extendida · 30 días"
    )

    tendencia30 = calcular_tendencia_30_dias(
        df,
        forecast,
    )

    estado30 = tendencia30.get(
        "estado",
        "Sin datos",
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

    # --------------------------------------------------------
    # CUADROS DE RESUMEN
    # --------------------------------------------------------

    t1, t2, t3, t4 = st.columns(
        4
    )

    t1.metric(
        "Tendencia 30 días",
        estado30,
    )

    if nivel15 is not None:

        t2.metric(
            "Nivel estimado día 15",
            f"{nivel15:.2f} m",
        )

    else:

        t2.metric(
            "Nivel estimado día 15",
            "Sin datos",
        )

    if nivel30 is not None:

        t3.metric(
            "Referencia día 30",
            f"{nivel30:.2f} m",
        )

    else:

        t3.metric(
            "Referencia día 30",
            "Sin datos",
        )

    if cambio30 is not None:

        texto_cambio = (
            f"{cambio30:+.2f} m"
        )

        if (
            pct30 is not None
            and not pd.isna(
                pct30
            )
        ):

            texto_cambio += (
                f" ({pct30:+.1f}%)"
            )

        t4.metric(
            "Cambio vs. actual",
            texto_cambio,
        )

    else:

        t4.metric(
            "Cambio vs. actual",
            "Sin datos",
        )

    # --------------------------------------------------------
    # MENSAJE DE INTERPRETACIÓN
    # --------------------------------------------------------

    if estado30 == "Tendencia creciente":

        st.warning(
            "↗️ La proyección extendida mantiene una "
            "tendencia general creciente hacia el horizonte "
            "de 30 días."
        )

    elif estado30 == "Tendencia bajante":

        st.info(
            "↘️ La proyección extendida mantiene una "
            "tendencia general bajante hacia el horizonte "
            "de 30 días."
        )

    elif estado30 == "Tendencia estable":

        st.success(
            "➡️ La proyección extendida indica una "
            "tendencia relativamente estable hacia "
            "el horizonte de 30 días."
        )

    else:

        st.info(
            "No hay información suficiente para "
            "calcular la tendencia extendida."
        )

    # --------------------------------------------------------
    # MINI GRÁFICO 30 DÍAS
    # --------------------------------------------------------

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

        trend_fig = go.Figure()

        # Últimos 30 días observados
        obs30 = df.tail(
            30
        )

        trend_fig.add_trace(
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

        # Pronóstico 1-15
        trend_fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(
                    forecast[
                        "datetime"
                    ]
                ),
                y=forecast[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico días 1–15",
            )
        )

        # Tendencia 16-30
        trend_fig.add_trace(
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

        trend_fig.update_layout(
            height=380,
            hovermode="x unified",
            xaxis_title="Fecha",
            yaxis_title=(
                "Nivel hidrométrico (m)"
            ),
            legend=dict(
                orientation="h",
                y=1.05,
            ),
        )

        trend_fig.update_xaxes(
            tickformat="%d/%m"
        )

        trend_fig.update_yaxes(
            range=[
                0,
                7,
            ],
            dtick=0.5,
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True,
        )

    st.caption(
        "Los primeros 15 días corresponden al pronóstico "
        "experimental multivariable. Del día 16 al 30 se muestra "
        "una tendencia extendida mediante extrapolación amortiguada. "
        "No debe interpretarse como un pronóstico meteorológico "
        "o hidrológico diario de 30 días."
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
            0.0
        )

        total_rain = float(
            rain[
                "precip_mm"
            ].sum()
        )

        max_rain = float(
            rain[
                "precip_mm"
            ].max()
        )

        wet_days = int(
            (
                rain[
                    "precip_mm"
                ]
                >= 1
            ).sum()
        )

        r1, r2, r3 = st.columns(
            3
        )

        r1.metric(
            "Acumulado previsto",
            f"{total_rain:.1f} mm",
        )

        r2.metric(
            "Máximo diario",
            f"{max_rain:.1f} mm",
        )

        r3.metric(
            "Días ≥ 1 mm",
            wet_days,
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
            height=350,
            xaxis_title="Fecha",
            yaxis_title=(
                "Precipitación (mm/día)"
            ),
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m"
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal del Paraná y tendencia"
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

        tendencia_q = calcular_tendencia_caudal(
            q_hist
        )

        q_actual = tendencia_q.get(
            "actual"
        )

        delta3 = tendencia_q.get(
            "delta_3"
        )

        delta7 = tendencia_q.get(
            "delta_7"
        )

        pct7 = tendencia_q.get(
            "pct_7"
        )

        estado_q = tendencia_q.get(
            "estado",
            "Sin datos",
        )

        q1, q2, q3, q4 = st.columns(
            4
        )

        if q_actual is not None:

            q1.metric(
                "Caudal actual",
                f"{q_actual:,.0f} m³/s",
            )

        else:

            q1.metric(
                "Caudal actual",
                "Sin datos",
            )

        if delta3 is not None:

            q2.metric(
                "Variación 3 días",
                f"{delta3:+,.0f} m³/s",
            )

        else:

            q2.metric(
                "Variación 3 días",
                "Sin datos",
            )

        if delta7 is not None:

            texto7 = (
                f"{delta7:+,.0f} m³/s"
            )

            if (
                pct7 is not None
                and not pd.isna(
                    pct7
                )
            ):

                texto7 += (
                    f" ({pct7:+.1f}%)"
                )

            q3.metric(
                "Variación 7 días",
                texto7,
            )

        else:

            q3.metric(
                "Variación 7 días",
                "Sin datos",
            )

        q4.metric(
            "Tendencia",
            estado_q,
        )

        # ----------------------------------------------------
        # GRÁFICO CAUDAL
        # ----------------------------------------------------

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
            height=400,
            xaxis_title="Fecha",
            yaxis_title="Caudal (m³/s)",
            hovermode="x unified",
        )

        q_fig.update_xaxes(
            tickformat="%d/%m"
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


    # ========================================================
    # VARIABLES DEL MODELO
    # ========================================================

    st.subheader(
        "🌊 Variables hidrológicas del modelo"
    )

    estaciones = resumen_estaciones_upstream(
        upstream_meta,
        upstream_history,
    )

    estado_variables = [
        {
            "Variable": "San Nicolás",
            "Tipo": "Nivel objetivo",
            "Estado": "✓ Disponible",
        },
        {
            "Variable": "Precipitación",
            "Tipo": "Meteorológica",
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
            "Tipo": "Hidrológica",
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

        estado_variables.append(
            {
                "Variable": item[
                    "Estación"
                ],
                "Tipo": "Nivel aguas arriba",
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
            estado_variables
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # IMPORTANCIA
    # ========================================================

    with st.expander(
        "🧠 Importancia de variables"
    ):

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

        else:

            st.info(
                "No se dispone de información "
                "de importancia de variables."
            )


    # ========================================================
    # INFORMACIÓN DEL MODELO
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y alcance"
    ):

        st.write(
            "**Pronóstico principal:** 15 días"
        )

        st.write(
            "**Tendencia extendida:** 30 días"
        )

        st.write(
            "**Modelo principal:** Random Forest Regressor"
        )

        rmse = metrics.get(
            "RMSE"
        )

        if rmse is not None:

            st.write(
                "**RMSE de validación:** "
                f"{float(rmse):.3f} m"
            )

        st.markdown(
            """
            El horizonte de **15 días** utiliza el modelo
            hidrológico multivariable.

            El indicador de **30 días** no constituye un segundo
            pronóstico meteorológico completo. Utiliza la dirección
            del pronóstico de 15 días y la tendencia hidrométrica
            reciente para construir una extrapolación amortiguada
            entre los días 16 y 30.

            Por este motivo debe interpretarse como un
            **indicador de tendencia de mediano plazo**.
            """
        )

        st.warning(
            "Todo el sistema es experimental y no reemplaza "
            "pronósticos, avisos ni alertas oficiales."
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | "
    "Nivel y caudal: INA | "
    "Precipitación: Open-Meteo | "
    "Pronóstico experimental: 15 días | "
    "Tendencia extendida: 30 días"
)
