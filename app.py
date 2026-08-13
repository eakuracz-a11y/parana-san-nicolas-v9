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

    El modelo puede incorporar el nivel local, niveles aguas arriba,
    precipitación prevista y caudal disponible.
    """
)


# ============================================================
# BARRA LATERAL
# ============================================================

st.sidebar.header(
    "Consulta online"
)

fecha_hasta = date.today()

# Mayor histórico para entrenamiento multivariable
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
    "Horizonte"
)

st.sidebar.write(
    "15 días"
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

    df[
        "datetime"
    ] = pd.to_datetime(
        df[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    if "value" in df.columns:

        df[
            "nivel"
        ] = pd.to_numeric(
            df[
                "value"
            ],
            errors="coerce",
        )

    elif "nivel" in df.columns:

        df[
            "nivel"
        ] = pd.to_numeric(
            df[
                "nivel"
            ],
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

def calcular_tendencia_caudal(df_caudal):

    resultado = {
        "actual": None,
        "hace_3_dias": None,
        "hace_7_dias": None,
        "delta_3": None,
        "delta_7": None,
        "pct_3": None,
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
    # 3 DÍAS
    # --------------------------------------------------------

    if len(
        valores
    ) >= 4:

        q3 = float(
            valores[-4]
        )

        delta3 = (
            actual
            - q3
        )

        pct3 = (
            delta3
            / q3
            * 100
            if q3 != 0
            else np.nan
        )

        resultado[
            "hace_3_dias"
        ] = q3

        resultado[
            "delta_3"
        ] = delta3

        resultado[
            "pct_3"
        ] = pct3

    # --------------------------------------------------------
    # 7 DÍAS
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

        pct7 = (
            delta7
            / q7
            * 100
            if q7 != 0
            else np.nan
        )

        resultado[
            "hace_7_dias"
        ] = q7

        resultado[
            "delta_7"
        ] = delta7

        resultado[
            "pct_7"
        ] = pct7

    # --------------------------------------------------------
    # PENDIENTE RECIENTE
    # --------------------------------------------------------

    ultimos = valores[
        -min(
            len(
                valores
            ),
            7,
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
    # CLASIFICACIÓN
    # --------------------------------------------------------

    if actual > 0:

        umbral = max(
            actual
            * 0.002,
            1.0,
        )

    else:

        umbral = 1.0

    if pendiente > umbral:

        estado = "Creciente"

    elif pendiente < -umbral:

        estado = "Bajante"

    else:

        estado = "Estable"

    resultado[
        "estado"
    ] = estado

    return resultado


# ============================================================
# ESTACIONES DISPONIBLES
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
        ultima_fecha = None

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

            ultima_fecha = info.get(
                "to_date"
            )

        # Buscar columna correspondiente
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

            disponible = (
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
                "Último dato catálogo": ultima_fecha,
            }
        )

    return estaciones


# ============================================================
# VALIDAR FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a Hasta."
    )


# ============================================================
# ACTUALIZACIÓN GENERAL
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
        # NIVEL SAN NICOLÁS
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
                    "No se obtuvieron datos "
                    "hidrométricos válidos."
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
                            "No fue posible obtener "
                            "todas las estaciones aguas arriba: "
                            f"{exc}"
                        )

                # ============================================
                # ENTRENAMIENTO
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
                # GUARDAR ESTADO
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
        "nivel, caudal, precipitación y estaciones aguas arriba."
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
    # GRÁFICO PRINCIPAL
    # ========================================================

    st.subheader(
        "📈 Nivel observado y pronóstico a 15 días"
    )

    fig = go.Figure()

    # Observado
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
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Observado: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    # ========================================================
    # FORECAST
    # ========================================================

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

        # Pronóstico
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
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Pronóstico: %{y:.2f} m"
                    "<extra></extra>"
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
    # LLUVIA PREVISTA
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
                >= 1.0
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

    else:

        st.info(
            "No se obtuvo pronóstico "
            "de precipitación."
        )


    # ========================================================
    # CAUDAL Y TENDENCIA
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

        tendencia = calcular_tendencia_caudal(
            q_hist
        )

        q_actual = tendencia.get(
            "actual"
        )

        q_delta3 = tendencia.get(
            "delta_3"
        )

        q_delta7 = tendencia.get(
            "delta_7"
        )

        q_pct7 = tendencia.get(
            "pct_7"
        )

        estado_q = tendencia.get(
            "estado",
            "Sin datos",
        )

        pendiente_q = tendencia.get(
            "pendiente"
        )

        q_future = pd.Series(
            dtype=float
        )

        if (
            isinstance(
                exog_future,
                pd.DataFrame,
            )
            and "caudal_m3s"
            in exog_future.columns
        ):

            q_future = (
                pd.to_numeric(
                    exog_future[
                        "caudal_m3s"
                    ],
                    errors="coerce",
                )
                .dropna()
            )

        if len(
            q_future
        ):

            q15 = float(
                q_future.iloc[-1]
            )

        else:

            q15 = None

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

        if q_delta3 is not None:

            q2.metric(
                "Variación 3 días",
                f"{q_delta3:+,.0f} m³/s",
            )

        else:

            q2.metric(
                "Variación 3 días",
                "Sin datos",
            )

        if q_delta7 is not None:

            texto_delta7 = (
                f"{q_delta7:+,.0f} m³/s"
            )

            if (
                q_pct7 is not None
                and not pd.isna(
                    q_pct7
                )
            ):

                texto_delta7 += (
                    f" ({q_pct7:+.1f}%)"
                )

            q3.metric(
                "Variación 7 días",
                texto_delta7,
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
        # SEGUNDA FILA
        # ----------------------------------------------------

        q5, q6, q7 = st.columns(
            3
        )

        if pendiente_q is not None:

            q5.metric(
                "Pendiente reciente",
                f"{pendiente_q:+,.1f} m³/s por día",
            )

        else:

            q5.metric(
                "Pendiente reciente",
                "Sin datos",
            )

        if q15 is not None:

            q6.metric(
                "Proyección día 15",
                f"{q15:,.0f} m³/s",
            )

        else:

            q6.metric(
                "Proyección día 15",
                "Sin datos",
            )

        if (
            q_actual is not None
            and q15 is not None
        ):

            cambio15 = (
                q15
                - q_actual
            )

            q7.metric(
                "Cambio proyectado 15 días",
                f"{cambio15:+,.0f} m³/s",
            )

        else:

            q7.metric(
                "Cambio proyectado 15 días",
                "Sin datos",
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

        info_q = exog_meta.get(
            "caudal_series"
        )

        if isinstance(
            info_q,
            dict,
        ):

            st.caption(
                "Serie de caudal utilizada: "
                f"**{info_q.get('station', '-')}** · "
                f"seriesId **{info_q.get('series_id', '-')}** · "
                f"{info_q.get('proc_name', '')}."
            )

        st.caption(
            "La tendencia del caudal se calcula sobre "
            "los últimos datos observados disponibles."
        )

        st.warning(
            "La proyección futura del caudal es experimental "
            "y extrapola una tendencia reciente amortiguada. "
            "No constituye un pronóstico oficial del INA."
        )

    else:

        st.info(
            "No se encontró una serie reciente "
            "de caudal adecuada."
        )


    # ========================================================
    # VARIABLES HIDROLÓGICAS UTILIZADAS
    # ========================================================

    st.subheader(
        "🌊 Variables hidrológicas del modelo"
    )

    estaciones = resumen_estaciones_upstream(
        upstream_meta,
        upstream_history,
    )

    estado_variables = []

    estado_variables.append(
        {
            "Variable": "San Nicolás",
            "Tipo": "Nivel objetivo",
            "Estado": "✓ Disponible",
        }
    )

    uses_rain = bool(
        models.get(
            "uses_rain",
            False,
        )
    )

    estado_variables.append(
        {
            "Variable": "Precipitación",
            "Tipo": "Variable meteorológica",
            "Estado": (
                "✓ Utilizada"
                if uses_rain
                else "✗ No utilizada"
            ),
        }
    )

    uses_caudal = bool(
        models.get(
            "uses_caudal",
            False,
        )
    )

    estado_variables.append(
        {
            "Variable": "Caudal",
            "Tipo": "Variable hidrológica",
            "Estado": (
                "✓ Utilizado"
                if uses_caudal
                else "✗ No utilizado"
            ),
        }
    )

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

    df_estado = pd.DataFrame(
        estado_variables
    )

    st.dataframe(
        df_estado,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DETALLE ESTACIONES AGUAS ARRIBA
    # ========================================================

    with st.expander(
        "📍 Detalle de estaciones aguas arriba"
    ):

        if estaciones:

            df_estaciones = pd.DataFrame(
                estaciones
            )

            st.dataframe(
                df_estaciones,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No se obtuvo metadata "
                "de estaciones aguas arriba."
            )


    # ========================================================
    # IMPORTANCIA DE VARIABLES
    # ========================================================

    st.subheader(
        "🧠 Importancia de variables del modelo"
    )

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
            height=650,
            xaxis_title=(
                "Importancia relativa"
            ),
            yaxis_title="Variable",
        )

        imp_fig.update_yaxes(
            autorange="reversed"
        )

        st.plotly_chart(
            imp_fig,
            use_container_width=True,
        )

        with st.expander(
            "📋 Ver importancia completa"
        ):

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No se dispone de información "
            "de importancia de variables."
        )


    # ========================================================
    # DETALLE PRONÓSTICO 15 DÍAS
    # ========================================================

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        st.subheader(
            "🔮 Detalle del pronóstico · 15 días"
        )

        table = forecast.copy()

        table[
            "Fecha"
        ] = pd.to_datetime(
            table[
                "datetime"
            ],
            errors="coerce",
        ).dt.strftime(
            "%d/%m/%Y"
        )

        table[
            "Nivel previsto (m)"
        ] = pd.to_numeric(
            table[
                "prediction"
            ],
            errors="coerce",
        ).round(
            2
        )

        table[
            "Inferior (m)"
        ] = pd.to_numeric(
            table[
                "lower"
            ],
            errors="coerce",
        ).round(
            2
        )

        table[
            "Superior (m)"
        ] = pd.to_numeric(
            table[
                "upper"
            ],
            errors="coerce",
        ).round(
            2
        )

        if "precip_mm" in table.columns:

            table[
                "Lluvia (mm)"
            ] = pd.to_numeric(
                table[
                    "precip_mm"
                ],
                errors="coerce",
            ).round(
                1
            )

        else:

            table[
                "Lluvia (mm)"
            ] = np.nan

        if "caudal_m3s" in table.columns:

            table[
                "Caudal proyectado (m³/s)"
            ] = pd.to_numeric(
                table[
                    "caudal_m3s"
                ],
                errors="coerce",
            ).round(
                0
            )

        else:

            table[
                "Caudal proyectado (m³/s)"
            ] = np.nan

        st.dataframe(
            table[
                [
                    "Fecha",
                    "Nivel previsto (m)",
                    "Inferior (m)",
                    "Superior (m)",
                    "Lluvia (mm)",
                    "Caudal proyectado (m³/s)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # INFORMACIÓN DEL MODELO
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y diagnóstico del modelo"
    ):

        st.write(
            "**Modelo:** Random Forest Regressor"
        )

        st.write(
            "**Horizonte:** 15 días"
        )

        st.write(
            "**Escala gráfica:** 0–7 m"
        )

        st.write(
            "**Precipitación:** "
            + (
                "incluida"
                if models.get(
                    "uses_rain",
                    False,
                )
                else "no incluida"
            )
        )

        st.write(
            "**Caudal:** "
            + (
                "incluido"
                if models.get(
                    "uses_caudal",
                    False,
                )
                else "no incluido"
            )
        )

        st.write(
            "**Niveles aguas arriba:** "
            + (
                "incluidos"
                if models.get(
                    "uses_upstream",
                    False,
                )
                else "no incluidos"
            )
        )

        rmse = metrics.get(
            "RMSE"
        )

        if rmse is not None:

            st.write(
                "**RMSE de validación:** "
                f"{float(rmse):.3f} m"
            )

        st.write(
            "**Observaciones utilizadas:**",
            models.get(
                "observations",
                "-",
            ),
        )

        st.write(
            "**Filas efectivas de entrenamiento:**",
            models.get(
                "training_rows",
                "-",
            ),
        )

        st.warning(
            "El resultado es experimental. "
            "No reemplaza pronósticos, avisos ni alertas "
            "emitidos por organismos oficiales."
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Paraná · San Nicolás V9 | "
    "Nivel y caudal: INA | "
    "Precipitación: Open-Meteo | "
    "Estaciones aguas arriba: INA | "
    "Pronóstico experimental multivariable: 15 días"
)
