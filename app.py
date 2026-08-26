import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import (
    date,
    timedelta,
    datetime,
)

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
# APP V11.0 RESTAURADA
# ============================================================

APP_VERSION = "V11.0"

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
# ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.60rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }

    div[data-testid="stAlert"] {
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
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
    **San Nicolás de los Arroyos**.

    El sistema combina:

    **nivel observado · estaciones aguas arriba · caudal · precipitación**
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
    "Nivel y caudal: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)

st.sidebar.caption(
    "Modelo: experimental"
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

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = (
        x
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
# NORMALIZAR NOMBRE DE ESTACIÓN
# ============================================================

def normalizar_estacion(
    texto,
):

    return (
        str(texto)
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


# ============================================================
# TENDENCIA
# ============================================================

def texto_tendencia(
    delta,
):

    if delta is None:

        return "Sin comparación"

    if not np.isfinite(
        delta
    ):

        return "Sin comparación"

    if delta > 0.01:

        return "↑ Creciendo"

    if delta < -0.01:

        return "↓ Bajando"

    return "→ Estable"


# ============================================================
# EXTENDER 15 A 30 DÍAS
# ============================================================

def extender_pronostico_30(
    forecast,
    df,
):

    if (
        forecast is None
        or not isinstance(
            forecast,
            pd.DataFrame,
        )
        or forecast.empty
    ):

        return pd.DataFrame()

    f = forecast.copy()

    f[
        "datetime"
    ] = pd.to_datetime(
        f[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    f[
        "prediction"
    ] = pd.to_numeric(
        f[
            "prediction"
        ],
        errors="coerce",
    )

    f = f.dropna(
        subset=[
            "datetime",
            "prediction",
        ]
    )

    if f.empty:

        return pd.DataFrame()

    result = f.copy()

    last_date = (
        f[
            "datetime"
        ].iloc[
            -1
        ]
    )

    last_level = float(
        f[
            "prediction"
        ].iloc[
            -1
        ]
    )

    if len(
        f
    ) >= 5:

        recent = (
            f[
                "prediction"
            ]
            .tail(
                5
            )
            .to_numpy(
                dtype=float
            )
        )

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        recent
                    )
                ),
                recent,
                1,
            )[0]
        )

    else:

        niveles = (
            pd.to_numeric(
                df[
                    "nivel"
                ],
                errors="coerce",
            )
            .dropna()
            .tail(
                7
            )
        )

        if len(
            niveles
        ) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(
                            niveles
                        )
                    ),
                    niveles.to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        else:

            slope = 0.0

    # limitar movimientos exagerados
    slope = float(
        np.clip(
            slope,
            -0.10,
            0.10,
        )
    )

    extra = []

    for step in range(
        16,
        TREND_DAYS + 1,
    ):

        damping = np.exp(
            -0.14
            * (
                step
                - FORECAST_DAYS
            )
        )

        daily_change = (
            slope
            * damping
        )

        last_level = float(
            np.clip(
                last_level
                + daily_change,
                Y_MIN,
                Y_MAX,
            )
        )

        last_date = (
            last_date
            + pd.Timedelta(
                days=1
            )
        )

        extra.append(
            {
                "datetime":
                    last_date,

                "prediction":
                    last_level,

                "lower":
                    np.nan,

                "upper":
                    np.nan,
            }
        )

    if extra:

        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    extra
                ),
            ],
            ignore_index=True,
        )

    return result


# ============================================================
# VALIDACIÓN FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a Hasta."
    )


# ============================================================
# ACTUALIZAR
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
            "Consultando nivel de San Nicolás en INA..."
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

                df_ina = (
                    pd.DataFrame()
                )

                error_ina = (
                    f"Error de conexión INA: {exc}"
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
                    "El INA respondió, pero no se obtuvieron "
                    "observaciones hidrométricas válidas."
                )

            else:

                # ============================================
                # PRECIPITACIÓN + CAUDAL
                # ============================================

                with st.spinner(
                    "Consultando precipitación y caudal..."
                ):

                    try:

                        (
                            exog_history,
                            exog_future,
                            exog_meta,
                        ) = (
                            get_exogenous_data(
                                inicio,
                                fin,
                                FORECAST_DAYS,
                            )
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
                            "No fue posible obtener todas "
                            "las variables externas. "
                            f"Detalle: {exc}"
                        )

                # ============================================
                # AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando estaciones aguas arriba..."
                ):

                    try:

                        (
                            upstream_history,
                            upstream_meta,
                        ) = (
                            get_upstream_history(
                                inicio,
                                fin,
                            )
                        )

                    except Exception as exc:

                        upstream_history = (
                            pd.DataFrame()
                        )

                        upstream_meta = {}

                        st.warning(
                            "No fue posible obtener todas "
                            "las estaciones aguas arriba. "
                            f"Detalle: {exc}"
                        )

                # ============================================
                # MODELO
                # ============================================

                with st.spinner(
                    "Entrenando modelo y generando pronóstico..."
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

                        forecast = predict(
                            df,
                            models,
                            days=
                                FORECAST_DAYS,
                            exog_future=
                                exog_future,
                        )

                        forecast30 = (
                            extender_pronostico_30(
                                forecast,
                                df,
                            )
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
                            "No fue posible generar el pronóstico. "
                            f"Detalle: {exc}"
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

if (
    "datos"
    not in st.session_state
):

    st.info(
        "Seleccione el período de análisis y presione "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = st.session_state.get(
        "datos",
        pd.DataFrame(),
    )

    forecast = st.session_state.get(
        "forecast",
        pd.DataFrame(),
    )

    forecast30 = st.session_state.get(
        "forecast30",
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
    # NIVEL ACTUAL
    # ========================================================

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

        st.error(
            "No existen niveles válidos para mostrar."
        )

        st.stop()


    nivel_actual = float(
        niveles.iloc[
            -1
        ]
    )

    nivel_anterior = None
    delta_actual = None

    if len(
        niveles
    ) >= 2:

        nivel_anterior = float(
            niveles.iloc[
                -2
            ]
        )

        delta_actual = (
            nivel_actual
            - nivel_anterior
        )


    ultima_fecha = (
        df[
            "datetime"
        ].iloc[
            -1
        ]
    )


    # ========================================================
    # SITUACIÓN OBSERVADA
    # ========================================================

    st.subheader(
        "📊 Situación observada · San Nicolás"
    )

    c1, c2, c3, c4, c5 = (
        st.columns(
            5
        )
    )

    c1.metric(
        "Último nivel",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_actual:+.2f} m"
            if delta_actual
            is not None
            else None
        ),
    )

    c2.metric(
        "Mínimo período",
        f"{niveles.min():.2f} m",
    )

    c3.metric(
        "Máximo período",
        f"{niveles.max():.2f} m",
    )

    c4.metric(
        "Promedio",
        f"{niveles.mean():.2f} m",
    )

    c5.metric(
        "Tendencia",
        texto_tendencia(
            delta_actual
        ),
    )


    try:

        fecha_texto = (
            pd.to_datetime(
                ultima_fecha
            )
            .strftime(
                "%d/%m/%Y %H:%M"
            )
        )

    except Exception:

        fecha_texto = (
            str(
                ultima_fecha
            )
        )


    st.caption(
        f"Última observación INA: {fecha_texto}"
    )


    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    st.subheader(
        "📈 Nivel observado y pronóstico experimental · 15 días"
    )

    fig = go.Figure()


    # --------------------------------------------------------
    # OBSERVADO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df[
                "datetime"
            ],
            y=df[
                "nivel"
            ],
            mode=
                "lines",
            name=
                "Nivel observado INA",
            line=dict(
                width=3,
            ),
        )
    )


    # --------------------------------------------------------
    # PRONÓSTICO
    # --------------------------------------------------------

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
        and "prediction"
        in forecast.columns
    ):

        f = forecast.copy()

        f[
            "prediction"
        ] = pd.to_numeric(
            f[
                "prediction"
            ],
            errors="coerce",
        )

        f = f.dropna(
            subset=[
                "prediction"
            ]
        )

        if not f.empty:

            # continuidad desde nivel real
            x_forecast = [
                ultima_fecha
            ] + f[
                "datetime"
            ].tolist()

            y_forecast = [
                nivel_actual
            ] + f[
                "prediction"
            ].tolist()

            fig.add_trace(
                go.Scatter(
                    x=x_forecast,
                    y=y_forecast,
                    mode=
                        "lines+markers",
                    name=
                        "Pronóstico 15 días",
                    line=dict(
                        width=3,
                    ),
                )
            )


            # ------------------------------------------------
            # INCERTIDUMBRE
            # ------------------------------------------------

            if (
                "lower"
                in f.columns
                and "upper"
                in f.columns
            ):

                lower = pd.to_numeric(
                    f[
                        "lower"
                    ],
                    errors="coerce",
                )

                upper = pd.to_numeric(
                    f[
                        "upper"
                    ],
                    errors="coerce",
                )

                if (
                    lower.notna().any()
                    and upper.notna().any()
                ):

                    fig.add_trace(
                        go.Scatter(
                            x=f[
                                "datetime"
                            ],
                            y=upper,
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
                            y=lower,
                            mode="lines",
                            fill=
                                "tonexty",
                            line=dict(
                                width=0,
                            ),
                            name=
                                "Rango de incertidumbre",
                        )
                    )


    fig.update_layout(
        height=500,
        hovermode=
            "x unified",
        legend=dict(
            orientation="h",
            y=1.06,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m",
    )

    fig.update_yaxes(
        title_text=
            "Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Niveles aguas arriba"
    )

    st.caption(
        "La variación compara la última medición disponible "
        "con la medición inmediatamente anterior de cada estación."
    )


    rows = []

    if isinstance(
        upstream_meta,
        dict,
    ):

        for (
            station,
            info,
        ) in upstream_meta.items():

            col = (
                "nivel_"
                + normalizar_estacion(
                    station
                )
            )

            ultimo = None
            anterior = None
            delta = None

            fecha_estacion = None

            disponible = False

            if (
                isinstance(
                    upstream_history,
                    pd.DataFrame,
                )
                and col
                in upstream_history.columns
            ):

                temp = upstream_history[
                    [
                        "datetime",
                        col,
                    ]
                ].copy()

                temp[
                    col
                ] = pd.to_numeric(
                    temp[
                        col
                    ],
                    errors="coerce",
                )

                temp = temp.dropna(
                    subset=[
                        col
                    ]
                )

                if not temp.empty:

                    disponible = True

                    ultimo = float(
                        temp[
                            col
                        ].iloc[
                            -1
                        ]
                    )

                    fecha_estacion = (
                        temp[
                            "datetime"
                        ].iloc[
                            -1
                        ]
                    )

                    if len(
                        temp
                    ) >= 2:

                        anterior = float(
                            temp[
                                col
                            ].iloc[
                                -2
                            ]
                        )

                        delta = (
                            ultimo
                            - anterior
                        )

            rows.append(
                {
                    "Estación":
                        station,

                    "Nivel actual (m)":
                        (
                            round(
                                ultimo,
                                2,
                            )
                            if ultimo
                            is not None
                            else np.nan
                        ),

                    "Variación (m)":
                        (
                            round(
                                delta,
                                2,
                            )
                            if delta
                            is not None
                            else np.nan
                        ),

                    "Tendencia":
                        texto_tendencia(
                            delta
                        )
                        if disponible
                        else "Sin datos",

                    "Última fecha":
                        (
                            pd.to_datetime(
                                fecha_estacion
                            )
                            .strftime(
                                "%d/%m/%Y"
                            )
                            if fecha_estacion
                            is not None
                            else ""
                        ),
                }
            )


    if rows:

        tabla_upstream = (
            pd.DataFrame(
                rows
            )
        )

        st.dataframe(
            tabla_upstream,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No hay estaciones aguas arriba disponibles."
        )


    # ========================================================
    # GRÁFICO AGUAS ARRIBA
    # ========================================================

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
    ):

        level_cols = [
            c
            for c
            in upstream_history.columns
            if c.startswith(
                "nivel_"
            )
        ]

        if level_cols:

            fig_up = go.Figure()

            for col in (
                level_cols
            ):

                nombre = (
                    col
                    .replace(
                        "nivel_",
                        "",
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                fig_up.add_trace(
                    go.Scatter(
                        x=upstream_history[
                            "datetime"
                        ],
                        y=upstream_history[
                            col
                        ],
                        mode="lines",
                        name=nombre,
                    )
                )

            fig_up.update_layout(
                height=430,
                hovermode=
                    "x unified",
                legend=dict(
                    orientation="h",
                    y=1.06,
                ),
            )

            fig_up.update_xaxes(
                title_text=
                    "Fecha",
                tickformat=
                    "%d/%m",
            )

            fig_up.update_yaxes(
                title_text=
                    "Nivel hidrométrico (m)",
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=
                    Y_STEP,
            )

            st.plotly_chart(
                fig_up,
                use_container_width=True,
            )


    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "📆 Tendencia extendida · 30 días"
    )

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        forecast30[
            "prediction"
        ] = pd.to_numeric(
            forecast30[
                "prediction"
            ],
            errors="coerce",
        )

        forecast30 = (
            forecast30.dropna(
                subset=[
                    "prediction"
                ]
            )
        )

        if not forecast30.empty:

            nivel_15 = None
            nivel_30 = None

            if len(
                forecast30
            ) >= 15:

                nivel_15 = float(
                    forecast30[
                        "prediction"
                    ].iloc[
                        14
                    ]
                )

            if len(
                forecast30
            ) >= 30:

                nivel_30 = float(
                    forecast30[
                        "prediction"
                    ].iloc[
                        29
                    ]
                )

            elif len(
                forecast30
            ):

                nivel_30 = float(
                    forecast30[
                        "prediction"
                    ].iloc[
                        -1
                    ]
                )


            t1, t2, t3, t4 = (
                st.columns(
                    4
                )
            )

            t1.metric(
                "Nivel actual",
                f"{nivel_actual:.2f} m",
            )

            t2.metric(
                "Nivel día 15",
                (
                    f"{nivel_15:.2f} m"
                    if nivel_15
                    is not None
                    else "Sin dato"
                ),
            )

            t3.metric(
                "Nivel día 30",
                (
                    f"{nivel_30:.2f} m"
                    if nivel_30
                    is not None
                    else "Sin dato"
                ),
            )

            if nivel_30 is not None:

                cambio30 = (
                    nivel_30
                    - nivel_actual
                )

                t4.metric(
                    "Cambio estimado",
                    f"{cambio30:+.2f} m",
                )

            else:

                t4.metric(
                    "Cambio estimado",
                    "Sin dato",
                )


            fig30 = go.Figure()

            reciente = df.tail(
                30
            )

            fig30.add_trace(
                go.Scatter(
                    x=reciente[
                        "datetime"
                    ],
                    y=reciente[
                        "nivel"
                    ],
                    mode="lines",
                    name=
                        "Observado reciente",
                )
            )

            # primeros 15 días
            parte15 = (
                forecast30.head(
                    15
                )
            )

            fig30.add_trace(
                go.Scatter(
                    x=parte15[
                        "datetime"
                    ],
                    y=parte15[
                        "prediction"
                    ],
                    mode=
                        "lines+markers",
                    name=
                        "Pronóstico 1–15 días",
                )
            )

            # días 16 a 30
            parte30 = (
                forecast30.iloc[
                    15:
                ]
            )

            if not parte30.empty:

                fig30.add_trace(
                    go.Scatter(
                        x=parte30[
                            "datetime"
                        ],
                        y=parte30[
                            "prediction"
                        ],
                        mode=
                            "lines+markers",
                        line=dict(
                            dash="dot",
                        ),
                        name=
                            "Tendencia 16–30 días",
                    )
                )

            fig30.update_layout(
                height=430,
                hovermode=
                    "x unified",
                legend=dict(
                    orientation="h",
                    y=1.06,
                ),
            )

            fig30.update_xaxes(
                title_text=
                    "Fecha",
                tickformat=
                    "%d/%m",
            )

            fig30.update_yaxes(
                title_text=
                    "Nivel hidrométrico (m)",
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=
                    Y_STEP,
            )

            st.plotly_chart(
                fig30,
                use_container_width=True,
            )

            st.caption(
                "Los primeros 15 días corresponden al modelo experimental. "
                "Entre los días 16 y 30 se muestra una extrapolación "
                "amortiguada de la tendencia y debe interpretarse "
                "como referencia de mediano plazo."
            )

    else:

        st.info(
            "No se encuentra disponible la tendencia a 30 días."
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
        ] = pd.to_numeric(
            rain[
                "precip_mm"
            ],
            errors="coerce",
        ).fillna(
            0.0
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
            "Días con lluvia ≥ 1 mm",
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
                name=
                    "Precipitación prevista",
            )
        )

        rain_fig.update_layout(
            height=320,
            yaxis_title=
                "Precipitación (mm/día)",
        )

        rain_fig.update_xaxes(
            tickformat=
                "%d/%m",
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
    ):

        q_hist = (
            exog_history.copy()
        )

        q_hist[
            "caudal_m3s"
        ] = pd.to_numeric(
            q_hist[
                "caudal_m3s"
            ],
            errors="coerce",
        )

        q_hist = (
            q_hist.dropna(
                subset=[
                    "caudal_m3s"
                ]
            )
        )

        if not q_hist.empty:

            q_values = (
                q_hist[
                    "caudal_m3s"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            q_actual = float(
                q_values[
                    -1
                ]
            )

            delta3 = None
            delta7 = None

            if len(
                q_values
            ) >= 4:

                delta3 = (
                    q_actual
                    - float(
                        q_values[
                            -4
                        ]
                    )
                )

            if len(
                q_values
            ) >= 8:

                delta7 = (
                    q_actual
                    - float(
                        q_values[
                            -8
                        ]
                    )
                )


            q1, q2, q3 = (
                st.columns(
                    3
                )
            )

            q1.metric(
                "Caudal actual",
                f"{q_actual:,.0f} m³/s",
            )

            q2.metric(
                "Variación 3 días",
                (
                    f"{delta3:+,.0f} m³/s"
                    if delta3
                    is not None
                    else "Sin dato"
                ),
            )

            q3.metric(
                "Variación 7 días",
                (
                    f"{delta7:+,.0f} m³/s"
                    if delta7
                    is not None
                    else "Sin dato"
                ),
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
                    name=
                        "Caudal observado",
                )
            )

            q_fig.update_layout(
                height=350,
                hovermode=
                    "x unified",
                yaxis_title=
                    "Caudal (m³/s)",
            )

            q_fig.update_xaxes(
                tickformat=
                    "%d/%m",
            )

            st.plotly_chart(
                q_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "La serie de caudal no contiene datos válidos."
            )

    else:

        st.info(
            "No existe una serie de caudal disponible."
        )


    # ========================================================
    # ESCENARIO 60 DÍAS
    # ========================================================

    try:

        render_stress_scenario(
            df=df,
            models=models,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
        )

    except Exception as exc:

        st.warning(
            "El escenario de 60 días no pudo calcularse. "
            f"Detalle: {exc}"
        )


    # ========================================================
    # ESTADO DEL SISTEMA
    # ========================================================

    with st.expander(
        "🔧 Estado del sistema"
    ):

        estado_rows = [
            {
                "Variable":
                    "Nivel San Nicolás",

                "Estado":
                    (
                        "✓ Disponible"
                        if not df.empty
                        else "✗ Sin datos"
                    ),
            },
            {
                "Variable":
                    "Pronóstico 15 días",

                "Estado":
                    (
                        "✓ Disponible"
                        if isinstance(
                            forecast,
                            pd.DataFrame,
                        )
                        and not forecast.empty
                        else "✗ Sin datos"
                    ),
            },
            {
                "Variable":
                    "Precipitación",

                "Estado":
                    (
                        "✓ Disponible"
                        if isinstance(
                            exog_future,
                            pd.DataFrame,
                        )
                        and not exog_future.empty
                        else "✗ Sin datos"
                    ),
            },
            {
                "Variable":
                    "Caudal",

                "Estado":
                    (
                        "✓ Disponible"
                        if isinstance(
                            exog_history,
                            pd.DataFrame,
                        )
                        and "caudal_m3s"
                        in exog_history.columns
                        and exog_history[
                            "caudal_m3s"
                        ].notna().any()
                        else "✗ Sin datos"
                    ),
            },
            {
                "Variable":
                    "Estaciones aguas arriba",

                "Estado":
                    (
                        "✓ Disponible"
                        if isinstance(
                            upstream_history,
                            pd.DataFrame,
                        )
                        and not upstream_history.empty
                        else "✗ Sin datos"
                    ),
            },
        ]

        st.dataframe(
            pd.DataFrame(
                estado_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # MODELO
    # ========================================================

    importance = (
        models.get(
            "importance"
        )
        if isinstance(
            models,
            dict,
        )
        else None
    )

    if (
        isinstance(
            importance,
            pd.DataFrame,
        )
        and not importance.empty
    ):

        with st.expander(
            "🧠 Variables con mayor influencia en el modelo"
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
                    name=
                        "Importancia",
                )
            )

            imp_fig.update_layout(
                height=550,
                xaxis_title=
                    "Importancia relativa",
                yaxis_title=
                    "Variable",
            )

            imp_fig.update_yaxes(
                autorange=
                    "reversed"
            )

            st.plotly_chart(
                imp_fig,
                use_container_width=True,
            )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y alcance"
    ):

        rmse = (
            metrics.get(
                "RMSE"
            )
            if isinstance(
                metrics,
                dict,
            )
            else None
        )

        if rmse is not None:

            st.write(
                "**RMSE de validación histórica:** "
                f"{float(rmse):.3f} m"
            )

        st.markdown(
            """
            **Pronóstico principal:** 15 días.

            **Tendencia extendida:** 30 días.

            **Escenario hipotético:** 60 días.

            El modelo utiliza el nivel de San Nicolás y,
            cuando existen datos disponibles, incorpora
            precipitación, caudal y niveles de estaciones
            aguas arriba.

            La sección de 30 días es una extensión de tendencia
            y no debe interpretarse como un pronóstico meteorológico
            diario equivalente al horizonte principal de 15 días.

            El escenario de 60 días representa una simulación
            experimental de condiciones históricamente elevadas.
            """
        )

        st.warning(
            "Esta plataforma es experimental e informativa. "
            "No reemplaza avisos, alertas ni pronósticos "
            "emitidos por organismos oficiales."
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

    Nivel hidrométrico y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Predicción y escenarios: **modelo experimental propio**
    """
)

st.warning(
    "Los resultados tienen carácter experimental e informativo. "
    "Ante situaciones de riesgo deben consultarse las "
    "comunicaciones oficiales de las autoridades competentes."
)

st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Pronóstico experimental 15 días | "
    "Tendencia 30 días | "
    "Escenario hipotético 60 días"
)
