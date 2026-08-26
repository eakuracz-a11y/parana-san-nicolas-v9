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
# V11 RESTAURADA
# ============================================================

APP_VERSION = "V11.0"

FORECAST_DAYS = 15
TREND_DAYS = 30

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title=
        "Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · Plataforma pública experimental "
    "de monitoreo y análisis hidrométrico"
)

st.markdown(
    """
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**.

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

fecha_hasta = (
    date.today()
)

fecha_desde = (
    fecha_hasta
    - timedelta(
        days=120
    )
)


desde = st.sidebar.date_input(
    "Desde",
    value=
        fecha_desde,
    format=
        "DD/MM/YYYY",
)


hasta = st.sidebar.date_input(
    "Hasta",
    value=
        fecha_hasta,
    format=
        "DD/MM/YYYY",
)


actualizar = (
    st.sidebar.button(
        "🔄 Actualizar modelo",
        use_container_width=True,
        type="primary",
    )
)


st.sidebar.divider()

st.sidebar.write(
    "**Pronóstico:** 15 días"
)

st.sidebar.write(
    "**Tendencia:** 30 días"
)

st.sidebar.write(
    "**Escenario:** 60 días"
)

st.sidebar.write(
    "**Escala:** 0–7 m"
)


# ============================================================
# PREPARAR NIVEL
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

        return (
            pd.DataFrame()
        )

    x = df.copy()

    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )

    if (
        "value"
        in x.columns
    ):

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "value"
            ],
            errors="coerce",
        )

    elif (
        "nivel"
        in x.columns
    ):

        x[
            "nivel"
        ] = pd.to_numeric(
            x[
                "nivel"
            ],
            errors="coerce",
        )

    else:

        return (
            pd.DataFrame()
        )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    return (
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


# ============================================================
# ACTUALIZAR
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "El período seleccionado no es válido."
        )

    else:

        inicio = (
            desde.strftime(
                "%Y-%m-%d"
            )
        )

        fin = (
            hasta.strftime(
                "%Y-%m-%d"
            )
        )

        # ----------------------------------------------------
        # NIVEL INA
        # ----------------------------------------------------

        with st.spinner(
            "Consultando nivel de San Nicolás..."
        ):

            (
                df_ina,
                error_ina,
            ) = observed(
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
                    "No se obtuvieron observaciones válidas."
                )

            else:

                # --------------------------------------------
                # EXÓGENOS
                # --------------------------------------------

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
                            f"Variables externas: {exc}"
                        )

                # --------------------------------------------
                # AGUAS ARRIBA
                # --------------------------------------------

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
                            f"Aguas arriba: {exc}"
                        )

                # --------------------------------------------
                # MODELO
                # --------------------------------------------

                with st.spinner(
                    "Entrenando modelo..."
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

                    except Exception as exc:

                        models = {}
                        metrics = {}
                        forecast = (
                            pd.DataFrame()
                        )

                        st.error(
                            f"Modelo: {exc}"
                        )

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
        "Seleccione el período y presione "
        "**Actualizar modelo**."
    )


# ============================================================
# RESULTADOS
# ============================================================

else:

    df = (
        st.session_state[
            "datos"
        ]
    )

    forecast = (
        st.session_state.get(
            "forecast",
            pd.DataFrame(),
        )
    )

    models = (
        st.session_state.get(
            "models",
            {},
        )
    )

    metrics = (
        st.session_state.get(
            "metrics",
            {},
        )
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
    # SITUACIÓN
    # ========================================================

    niveles = pd.to_numeric(
        df[
            "nivel"
        ],
        errors="coerce",
    ).dropna()

    nivel_actual = float(
        niveles.iloc[
            -1
        ]
    )

    c1, c2, c3, c4 = (
        st.columns(
            4
        )
    )

    c1.metric(
        "Último nivel",
        f"{nivel_actual:.2f} m",
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


    # ========================================================
    # NIVEL + FORECAST
    # ========================================================

    st.subheader(
        "📈 Nivel y pronóstico · 15 días"
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
            mode=
                "lines",
            name=
                "Nivel observado",
        )
    )

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        fig.add_trace(
            go.Scatter(
                x=forecast[
                    "datetime"
                ],
                y=forecast[
                    "prediction"
                ],
                mode=
                    "lines+markers",
                name=
                    "Pronóstico",
            )
        )

        if (
            "lower"
            in forecast.columns
            and "upper"
            in forecast.columns
        ):

            fig.add_trace(
                go.Scatter(
                    x=forecast[
                        "datetime"
                    ],
                    y=forecast[
                        "upper"
                    ],
                    mode="lines",
                    line=dict(
                        width=0,
                    ),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast[
                        "datetime"
                    ],
                    y=forecast[
                        "lower"
                    ],
                    mode="lines",
                    fill=
                        "tonexty",
                    name=
                        "Incertidumbre",
                    line=dict(
                        width=0,
                    ),
                )
            )

    fig.update_layout(
        height=470,
        hovermode=
            "x unified",
        yaxis_title=
            "Nivel (m)",
    )

    fig.update_yaxes(
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
    )

    fig.update_xaxes(
        tickformat=
            "%d/%m",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # ESTACIONES AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Estaciones aguas arriba"
    )

    rows = []

    for (
        station,
        info,
    ) in upstream_meta.items():

        disponible = False
        ultimo = None
        delta = None

        nombre = (
            station
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
            col
            in upstream_history.columns
        ):

            valores = pd.to_numeric(
                upstream_history[
                    col
                ],
                errors="coerce",
            ).dropna()

            if not valores.empty:

                disponible = True

                ultimo = float(
                    valores.iloc[
                        -1
                    ]
                )

                if len(
                    valores
                ) >= 2:

                    delta = (
                        ultimo
                        - float(
                            valores.iloc[
                                -2
                            ]
                        )
                    )

        rows.append(
            {
                "Estación":
                    station,

                "Nivel":
                    (
                        round(
                            ultimo,
                            2,
                        )
                        if ultimo
                        is not None
                        else None
                    ),

                "Variación":
                    (
                        round(
                            delta,
                            2,
                        )
                        if delta
                        is not None
                        else None
                    ),

                "Estado":
                    (
                        "↑"
                        if (
                            delta
                            is not None
                            and delta
                            > 0.01
                        )
                        else "↓"
                        if (
                            delta
                            is not None
                            and delta
                            < -0.01
                        )
                        else "→"
                        if disponible
                        else "Sin datos"
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
    # LLUVIA
    # ========================================================

    st.subheader(
        "🌧️ Precipitación prevista"
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

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=exog_future[
                    "datetime"
                ],
                y=exog_future[
                    "precip_mm"
                ],
                name=
                    "Precipitación",
            )
        )

        rain_fig.update_layout(
            height=300,
            yaxis_title=
                "mm/día",
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal"
    )

    if (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and "caudal_m3s"
        in exog_history.columns
    ):

        q = exog_history.dropna(
            subset=[
                "caudal_m3s"
            ]
        )

        if not q.empty:

            q_fig = go.Figure()

            q_fig.add_trace(
                go.Scatter(
                    x=q[
                        "datetime"
                    ],
                    y=q[
                        "caudal_m3s"
                    ],
                    mode="lines",
                    name=
                        "Caudal",
                )
            )

            q_fig.update_layout(
                height=320,
                yaxis_title=
                    "m³/s",
            )

            st.plotly_chart(
                q_fig,
                use_container_width=True,
            )


    # ========================================================
    # ESCENARIO 60 DÍAS
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
    # MODELO
    # ========================================================

    with st.expander(
        "🧠 Modelo"
    ):

        rmse = metrics.get(
            "RMSE"
        )

        if rmse is not None:

            st.write(
                f"**RMSE:** {float(rmse):.3f} m"
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

            st.dataframe(
                importance.head(
                    20
                ),
                hide_index=True,
                use_container_width=True,
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

st.caption(
    "Datos hidrométricos y caudal: INA · "
    "Precipitación: Open-Meteo · "
    "Modelo experimental propio"
)
