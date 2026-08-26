import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta, datetime

from src.ina import observed

from src.model import (
    train,
    predict,
    resumen_niveles_estaciones,
)

from src.exogenous import (
    get_exogenous_data,
)

from src.upstream import (
    get_upstream_history,
)

try:
    from src.stress_ui import (
        render_stress_scenario,
    )
except Exception:
    render_stress_scenario = None


# ============================================================
# PARANÁ · SAN NICOLÁS V12.0
# MODELO DE PROPAGACIÓN HIDROLÓGICA
# ============================================================

APP_VERSION = "V12.0"

FORECAST_DAYS = 15
TREND_DAYS = 30

HISTORY_START = "1900-01-01"

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
        font-size: 1.65rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.86rem;
    }

    .small-note {
        font-size: 0.82rem;
        color: #6b7280;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES
# ============================================================

def preparar_datos(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.DataFrame()

    x = df.copy()

    if "datetime" not in x.columns:
        return pd.DataFrame()

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

    if "nivel" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["nivel"],
            errors="coerce",
        )

    elif "value" in x.columns:

        x["nivel"] = pd.to_numeric(
            x["value"],
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
        x.sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return x


def formatear_fecha(value):

    if value is None:
        return "Sin dato"

    try:

        dt = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(dt):
            return "Sin dato"

        return dt.strftime(
            "%d/%m/%Y"
        )

    except Exception:
        return str(value)


def detectar_columna_caudal(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return None

    if (
        "caudal_m3s" in df.columns
        and pd.to_numeric(
            df["caudal_m3s"],
            errors="coerce",
        ).notna().any()
    ):
        return "caudal_m3s"

    for column in df.columns:

        name = str(column).lower()

        if (
            "caudal" in name
            or name.startswith("q_")
        ):

            if pd.to_numeric(
                df[column],
                errors="coerce",
            ).notna().any():

                return column

    return None


def detectar_columnas_lluvia(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return []

    result = []

    for column in df.columns:

        name = str(column).lower()

        if (
            name == "precip_mm"
            or "precip" in name
            or "lluvia" in name
            or name.startswith("rain_")
        ):

            if pd.to_numeric(
                df[column],
                errors="coerce",
            ).notna().any():

                result.append(column)

    return result


def tendencia_caudal(
    df,
    column,
):

    result = {
        "actual": None,
        "delta_3": None,
        "delta_7": None,
        "estado": "Sin datos",
    }

    if (
        df is None
        or df.empty
        or column is None
        or column not in df.columns
    ):
        return result

    x = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if x.empty:
        return result

    actual = float(x.iloc[-1])

    result["actual"] = actual

    if len(x) >= 4:

        result["delta_3"] = (
            actual
            - float(x.iloc[-4])
        )

    if len(x) >= 8:

        result["delta_7"] = (
            actual
            - float(x.iloc[-8])
        )

    recent = x.tail(
        min(
            7,
            len(x),
        )
    )

    if len(recent) >= 3:

        slope = float(
            np.polyfit(
                np.arange(len(recent)),
                recent.to_numpy(dtype=float),
                1,
            )[0]
        )

    else:
        slope = 0.0

    threshold = max(
        abs(actual) * 0.002,
        1.0,
    )

    if slope > threshold:

        result["estado"] = "↑ Creciendo"

    elif slope < -threshold:

        result["estado"] = "↓ Bajando"

    else:

        result["estado"] = "→ Estable"

    return result


def extender_30_dias(
    forecast15,
    df,
):

    if (
        forecast15 is None
        or not isinstance(
            forecast15,
            pd.DataFrame,
        )
        or forecast15.empty
    ):
        return pd.DataFrame()

    f = forecast15.copy()

    f["datetime"] = pd.to_datetime(
        f["datetime"],
        errors="coerce",
    )

    f["prediction"] = pd.to_numeric(
        f["prediction"],
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

    last_date = f["datetime"].iloc[-1]

    last_level = float(
        f["prediction"].iloc[-1]
    )

    if len(f) >= 5:

        recent = f["prediction"].tail(5)

        slope = float(
            np.polyfit(
                np.arange(len(recent)),
                recent.to_numpy(dtype=float),
                1,
            )[0]
        )

    else:

        niveles = pd.to_numeric(
            df["nivel"],
            errors="coerce",
        ).dropna().tail(5)

        if len(niveles) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(len(niveles)),
                    niveles.to_numpy(dtype=float),
                    1,
                )[0]
            )

        else:
            slope = 0.0

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
            + pd.Timedelta(days=1)
        )

        extra.append(
            {
                "datetime": last_date,
                "prediction": last_level,
                "lower": np.nan,
                "upper": np.nan,
                "delta_prediction":
                    daily_change,
            }
        )

    if extra:

        result = pd.concat(
            [
                result,
                pd.DataFrame(extra),
            ],
            ignore_index=True,
        )

    return result


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · Modelo experimental de propagación hidrológica"
)

st.markdown(
    """
    La plataforma analiza el río como un **sistema aguas arriba → San Nicolás**.

    La fecha seleccionada actúa como **fecha base del cálculo** y el
    pronóstico utiliza niveles hidrométricos, variaciones aguas arriba,
    caudal, precipitación y relaciones históricas de propagación.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Consulta"
)

fecha_base = st.sidebar.date_input(
    "Fecha base",
    value=date.today(),
    format="DD/MM/YYYY",
    help=(
        "El modelo sólo utiliza observaciones "
        "disponibles hasta esta fecha."
    ),
)

dias_historia_visual = st.sidebar.slider(
    "Días visibles de historia",
    min_value=30,
    max_value=365,
    value=120,
    step=30,
)

actualizar = st.sidebar.button(
    "🔄 Actualizar modelo",
    use_container_width=True,
    type="primary",
)

st.sidebar.divider()

st.sidebar.subheader(
    "Modelo"
)

st.sidebar.write(
    "**Pronóstico principal:** 15 días"
)

st.sidebar.write(
    "**Tendencia orientativa:** 30 días"
)

st.sidebar.write(
    "**Escala de nivel:** 0–7 m"
)

st.sidebar.caption(
    "Nivel y caudal: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)


# ============================================================
# ACTUALIZAR DATOS
# ============================================================

if actualizar:

    fin = fecha_base.strftime(
        "%Y-%m-%d"
    )

    inicio_visual = (
        fecha_base
        - timedelta(
            days=dias_historia_visual
        )
    ).strftime(
        "%Y-%m-%d"
    )

    # ========================================================
    # SAN NICOLÁS
    # ========================================================

    with st.spinner(
        "Consultando nivel de San Nicolás..."
    ):

        df_raw, error = observed(
            inicio_visual,
            fin,
        )

    if error:

        st.error(error)

    else:

        df = preparar_datos(
            df_raw
        )

        if df.empty:

            st.error(
                "No existen observaciones válidas "
                "de San Nicolás para la fecha seleccionada."
            )

        else:

            # =================================================
            # HISTORIA COMPLETA
            # =================================================

            with st.spinner(
                "Recuperando historial hidrométrico completo..."
            ):

                try:

                    hist_raw, hist_error = observed(
                        HISTORY_START,
                        fin,
                    )

                    df_hist = preparar_datos(
                        hist_raw
                    )

                    if (
                        hist_error
                        or df_hist.empty
                    ):

                        df_hist = df.copy()

                        st.warning(
                            "No fue posible recuperar todo "
                            "el historial de San Nicolás."
                        )

                except Exception:

                    df_hist = df.copy()

                    st.warning(
                        "No fue posible recuperar todo "
                        "el historial de San Nicolás."
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
                        HISTORY_START,
                        fin,
                        TREND_DAYS,
                    )

                except Exception as exc:

                    exog_history = pd.DataFrame()
                    exog_future = pd.DataFrame()
                    exog_meta = {}

                    st.warning(
                        "No fue posible recuperar todas "
                        f"las variables externas: {exc}"
                    )

            # =================================================
            # ESTACIONES AGUAS ARRIBA
            # =================================================

            with st.spinner(
                "Consultando niveles aguas arriba..."
            ):

                try:

                    (
                        upstream_history,
                        upstream_meta,
                    ) = get_upstream_history(
                        HISTORY_START,
                        fin,
                    )

                except Exception as exc:

                    upstream_history = pd.DataFrame()
                    upstream_meta = {}

                    st.warning(
                        "No fue posible recuperar todas "
                        f"las estaciones aguas arriba: {exc}"
                    )

            # =================================================
            # MODELO V12
            # =================================================

            with st.spinner(
                "Analizando propagación histórica y generando pronóstico..."
            ):

                try:

                    models, metrics = train(
                        df=df_hist,
                        exog_history=exog_history,
                        upstream_history=upstream_history,
                        fecha_base=fin,
                    )

                    forecast = predict(
                        df=df_hist,
                        models=models,
                        days=FORECAST_DAYS,
                        exog_future=exog_future,
                        fecha_base=fin,
                    )

                    forecast30 = extender_30_dias(
                        forecast,
                        df_hist,
                    )

                except Exception as exc:

                    models = {}
                    metrics = {}
                    forecast = pd.DataFrame()
                    forecast30 = pd.DataFrame()

                    st.error(
                        "No fue posible generar "
                        f"el modelo V12: {exc}"
                    )

            # =================================================
            # GUARDAR SESIÓN
            # =================================================

            st.session_state[
                "datos"
            ] = df

            st.session_state[
                "df_hist"
            ] = df_hist

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
                "fecha_base"
            ] = fecha_base

            st.session_state[
                "actualizado"
            ] = datetime.now()

            st.success(
                "✅ Datos y modelo V12 actualizados."
            )


# ============================================================
# ESTADO INICIAL
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione la **fecha base** y presione "
        "**Actualizar modelo**."
    )

else:

    # ========================================================
    # RECUPERAR SESIÓN
    # ========================================================

    df = st.session_state[
        "datos"
    ]

    df_hist = st.session_state.get(
        "df_hist",
        df,
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

    upstream_history = st.session_state.get(
        "upstream_history",
        pd.DataFrame(),
    )

    fecha_base_usada = st.session_state.get(
        "fecha_base"
    )

    actualizado = st.session_state.get(
        "actualizado"
    )

    # ========================================================
    # SAN NICOLÁS
    # ========================================================

    niveles_validos = pd.to_numeric(
        df["nivel"],
        errors="coerce",
    ).dropna()

    nivel_actual = float(
        niveles_validos.iloc[-1]
    )

    nivel_anterior = (
        float(
            niveles_validos.iloc[-2]
        )
        if len(
            niveles_validos
        ) >= 2
        else np.nan
    )

    delta_local = (
        nivel_actual
        - nivel_anterior
        if np.isfinite(
            nivel_anterior
        )
        else np.nan
    )

    if not np.isfinite(
        delta_local
    ):

        tendencia_local = (
            "Sin comparación"
        )

    elif delta_local > 0.01:

        tendencia_local = (
            "↑ Creciendo"
        )

    elif delta_local < -0.01:

        tendencia_local = (
            "↓ Bajando"
        )

    else:

        tendencia_local = (
            "→ Estable"
        )

    ultima_fecha = pd.to_datetime(
        df["datetime"].iloc[-1]
    )

    st.subheader(
        "📍 Fecha base y nivel de San Nicolás"
    )

    a1, a2, a3, a4 = st.columns(
        4
    )

    a1.metric(
        "Nivel San Nicolás",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_local:+.2f} m"
            if np.isfinite(
                delta_local
            )
            else None
        ),
    )

    a2.metric(
        "Tendencia",
        tendencia_local,
    )

    a3.metric(
        "Fecha observada",
        formatear_fecha(
            ultima_fecha
        ),
    )

    a4.metric(
        "Modelo",
        APP_VERSION,
    )

    if (
        fecha_base_usada is not None
        and ultima_fecha.date()
        < fecha_base_usada
    ):

        st.warning(
            "La última medición disponible del INA "
            "es anterior a la fecha base seleccionada. "
            "El pronóstico comienza desde la última "
            "observación real disponible."
        )

    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    st.divider()

    st.subheader(
        "🌊 Niveles aguas arriba"
    )

    station_summary = (
        resumen_niveles_estaciones(
            upstream_history=
                upstream_history,

            df_local=
                df_hist,

            fecha_base=
                fecha_base_usada,
        )
    )

    if station_summary.empty:

        st.warning(
            "No se pudieron construir los niveles "
            "de las estaciones aguas arriba."
        )

    else:

        station_rows = (
            station_summary
            .to_dict(
                "records"
            )
        )

        for start in range(
            0,
            len(
                station_rows
            ),
            4,
        ):

            group = station_rows[
                start:start + 4
            ]

            cols = st.columns(
                len(group)
            )

            for column_ui, row in zip(
                cols,
                group,
            ):

                current = row.get(
                    "Nivel actual",
                    np.nan,
                )

                previous = row.get(
                    "Nivel anterior",
                    np.nan,
                )

                delta = row.get(
                    "Variación",
                    np.nan,
                )

                trend = row.get(
                    "Tendencia",
                    "Sin datos",
                )

                with column_ui:

                    st.metric(
                        row.get(
                            "Estación",
                            "Estación",
                        ),
                        (
                            f"{current:.2f} m"
                            if pd.notna(
                                current
                            )
                            else "Sin dato"
                        ),
                        (
                            f"{delta:+.2f} m"
                            if pd.notna(
                                delta
                            )
                            else None
                        ),
                    )

                    st.caption(
                        f"{trend} · anterior: "
                        + (
                            f"{previous:.2f} m"
                            if pd.notna(
                                previous
                            )
                            else "sin dato"
                        )
                    )

        st.caption(
            "La variación compara la última medición "
            "disponible hasta la fecha base con la "
            "medición anterior de cada estación."
        )

    # ========================================================
    # PERFIL DEL CORREDOR
    # ========================================================

    if not station_summary.empty:

        corridor = station_summary.copy()

        corridor[
            "Nivel actual"
        ] = pd.to_numeric(
            corridor[
                "Nivel actual"
            ],
            errors="coerce",
        )

        corridor = corridor.dropna(
            subset=[
                "Nivel actual"
            ]
        )

        if not corridor.empty:

            st.subheader(
                "🧭 Perfil actual del corredor"
            )

            corridor_fig = go.Figure()

            corridor_fig.add_trace(
                go.Scatter(
                    x=corridor[
                        "Estación"
                    ],
                    y=corridor[
                        "Nivel actual"
                    ],
                    mode="lines+markers+text",
                    text=[
                        f"{value:.2f} m"
                        for value
                        in corridor[
                            "Nivel actual"
                        ]
                    ],
                    textposition=
                        "top center",
                    name=
                        "Nivel actual",
                )
            )

            corridor_fig.update_layout(
                height=400,
                hovermode=
                    "x unified",
                yaxis_title=
                    "Nivel hidrométrico (m)",
                xaxis_title=
                    "Estación",
            )

            corridor_fig.update_yaxes(
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=
                    Y_STEP,
            )

            st.plotly_chart(
                corridor_fig,
                use_container_width=True,
            )

    # ========================================================
    # CORRIENTES → SAN NICOLÁS
    # ========================================================

    st.divider()

    st.subheader(
        "🔗 Relación histórica Corrientes → San Nicolás"
    )

    relation = models.get(
        "corrientes_san_nicolas",
        {},
    )

    if relation:

        lag = relation.get(
            "Desfase días"
        )

        corr = relation.get(
            "Correlación"
        )

        slope = relation.get(
            "Respuesta destino/origen"
        )

        pairs = relation.get(
            "Pares históricos"
        )

        r1, r2, r3, r4 = st.columns(
            4
        )

        r1.metric(
            "Desfase histórico",
            (
                f"{int(lag)} días"
                if pd.notna(
                    lag
                )
                else "Sin dato"
            ),
        )

        r2.metric(
            "Correlación",
            (
                f"{float(corr):.2f}"
                if pd.notna(
                    corr
                )
                else "Sin dato"
            ),
        )

        r3.metric(
            "Respuesta relativa",
            (
                f"{float(slope):.2f} m/m"
                if pd.notna(
                    slope
                )
                else "Sin dato"
            ),
        )

        r4.metric(
            "Pares históricos",
            (
                f"{int(pairs)}"
                if pd.notna(
                    pairs
                )
                else "Sin dato"
            ),
        )

        st.caption(
            "La respuesta relativa expresa el cambio "
            "histórico estimado en San Nicolás por cada "
            "1 m de cambio comparable en Corrientes."
        )

    else:

        st.info(
            "Todavía no existe suficiente información "
            "coincidente para calcular "
            "Corrientes → San Nicolás."
        )

    # ========================================================
    # GRÁFICO CORRIENTES / SAN NICOLÁS
    # ========================================================

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
        and "nivel_corrientes"
        in upstream_history.columns
    ):

        corrientes_hist = (
            upstream_history.copy()
        )

        corrientes_hist[
            "datetime"
        ] = pd.to_datetime(
            corrientes_hist[
                "datetime"
            ],
            errors="coerce",
            utc=True,
        )

        corrientes_hist[
            "nivel_corrientes"
        ] = pd.to_numeric(
            corrientes_hist[
                "nivel_corrientes"
            ],
            errors="coerce",
        )

        local_hist = df_hist[
            [
                "datetime",
                "nivel",
            ]
        ].copy()

        local_hist[
            "datetime"
        ] = pd.to_datetime(
            local_hist[
                "datetime"
            ],
            errors="coerce",
            utc=True,
        )

        comparison = (
            corrientes_hist[
                [
                    "datetime",
                    "nivel_corrientes",
                ]
            ]
            .merge(
                local_hist,
                on="datetime",
                how="inner",
            )
            .dropna()
        )

        if not comparison.empty:

            cutoff_plot = (
                comparison[
                    "datetime"
                ].max()
                - pd.Timedelta(
                    days=1095
                )
            )

            comparison_plot = (
                comparison[
                    comparison[
                        "datetime"
                    ] >= cutoff_plot
                ]
                .copy()
            )

            corr_fig = go.Figure()

            corr_fig.add_trace(
                go.Scatter(
                    x=comparison_plot[
                        "datetime"
                    ],
                    y=comparison_plot[
                        "nivel_corrientes"
                    ],
                    mode="lines",
                    name="Corrientes",
                )
            )

            corr_fig.add_trace(
                go.Scatter(
                    x=comparison_plot[
                        "datetime"
                    ],
                    y=comparison_plot[
                        "nivel"
                    ],
                    mode="lines",
                    name="San Nicolás",
                )
            )

            corr_fig.update_layout(
                height=450,
                hovermode=
                    "x unified",
                legend=dict(
                    orientation="h",
                    y=1.05,
                ),
            )

            corr_fig.update_xaxes(
                title_text="Fecha",
                tickformat=
                    "%d/%m/%Y",
            )

            corr_fig.update_yaxes(
                title_text=
                    "Nivel hidrométrico (m)",
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=Y_STEP,
            )

            st.plotly_chart(
                corr_fig,
                use_container_width=True,
            )

    # ========================================================
    # RESPUESTA NIVEL VS CAUDAL
    # ========================================================

    st.divider()

    st.subheader(
        "💧 Respuesta histórica del nivel ante aumentos de caudal"
    )

    response_flow = models.get(
        "response_flow",
        pd.DataFrame(),
    )

    if (
        isinstance(
            response_flow,
            pd.DataFrame,
        )
        and not response_flow.empty
    ):

        display_flow = (
            response_flow.copy()
        )

        for column in [
            "Correlación",
            "Respuesta nivel/caudal",
            "RMSE",
        ]:

            if column in display_flow.columns:

                display_flow[
                    column
                ] = pd.to_numeric(
                    display_flow[
                        column
                    ],
                    errors="coerce",
                ).round(3)

        st.dataframe(
            display_flow,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Para cada estación se busca históricamente "
            "el desfase en el que un aumento del caudal "
            "presenta la mayor relación con el cambio "
            "posterior del nivel."
        )

    else:

        st.info(
            "No hay suficientes datos coincidentes "
            "de nivel y caudal para construir "
            "esta relación."
        )

    # ========================================================
    # PROPAGACIÓN ENTRE ESTACIONES
    # ========================================================

    propagation = models.get(
        "propagation",
        pd.DataFrame(),
    )

    if (
        isinstance(
            propagation,
            pd.DataFrame,
        )
        and not propagation.empty
    ):

        with st.expander(
            "🌊 Propagación histórica entre estaciones"
        ):

            prop_display = (
                propagation.copy()
            )

            for column in [
                "Correlación",
                "Respuesta destino/origen",
                "RMSE",
            ]:

                if column in prop_display.columns:

                    prop_display[
                        column
                    ] = pd.to_numeric(
                        prop_display[
                            column
                        ],
                        errors="coerce",
                    ).round(3)

            st.dataframe(
                prop_display,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Pronóstico San Nicolás · 15 días"
    )

    fig = go.Figure()

    visible_start = (
        pd.to_datetime(
            ultima_fecha
        )
        - pd.Timedelta(
            days=
                dias_historia_visual
        )
    )

    df_plot = df[
        pd.to_datetime(
            df[
                "datetime"
            ],
            utc=True,
        )
        >= visible_start
    ].copy()

    fig.add_trace(
        go.Scatter(
            x=df_plot[
                "datetime"
            ],
            y=df_plot[
                "nivel"
            ],
            mode="lines",
            name="Nivel observado",
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Nivel: %{y:.2f} m"
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
                name=
                    "Intervalo experimental",
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
                name="Pronóstico V12",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>Pronóstico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        first_forecast = (
            f.iloc[0]
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    ultima_fecha,
                    first_forecast[
                        "datetime"
                    ],
                ],
                y=[
                    nivel_actual,
                    first_forecast[
                        "prediction"
                    ],
                ],
                mode="lines",
                line=dict(
                    dash="dash",
                    width=3,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_vline(
        x=pd.to_datetime(
            ultima_fecha
        ).timestamp()
        * 1000,
        line_dash="dot",
        annotation_text=
            "Fecha base",
        annotation_position=
            "top",
    )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.05,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat=
            "%d/%m/%Y",
    )

    fig.update_yaxes(
        title_text=
            "Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=Y_STEP,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
    ):

        p1, p2, p3, p4 = st.columns(
            4
        )

        day1 = float(
            forecast[
                "prediction"
            ].iloc[0]
        )

        day7 = float(
            forecast[
                "prediction"
            ].iloc[
                min(
                    6,
                    len(
                        forecast
                    ) - 1,
                )
            ]
        )

        day15 = float(
            forecast[
                "prediction"
            ].iloc[-1]
        )

        change15 = (
            day15
            - nivel_actual
        )

        p1.metric(
            "Nivel base",
            f"{nivel_actual:.2f} m",
        )

        p2.metric(
            "Día 1",
            f"{day1:.2f} m",
            f"{day1 - nivel_actual:+.2f} m",
        )

        p3.metric(
            "Día 7",
            f"{day7:.2f} m",
            f"{day7 - nivel_actual:+.2f} m",
        )

        p4.metric(
            "Día 15",
            f"{day15:.2f} m",
            f"{change15:+.2f} m",
        )

    # ========================================================
    # TENDENCIA 30 DÍAS
    # ========================================================

    st.subheader(
        "🧭 Tendencia orientativa · 30 días"
    )

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
    ):

        fig30 = go.Figure()

        obs30 = df.tail(30)

        fig30.add_trace(
            go.Scatter(
                x=obs30[
                    "datetime"
                ],
                y=obs30[
                    "nivel"
                ],
                mode="lines",
                name=
                    "Observado reciente",
            )
        )

        fig30.add_trace(
            go.Scatter(
                x=forecast30[
                    "datetime"
                ],
                y=forecast30[
                    "prediction"
                ],
                mode="lines+markers",
                name=
                    "Tendencia 30 días",
            )
        )

        fig30.update_layout(
            height=400,
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
            title_text=
                "Nivel hidrométrico (m)",
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
            "Los primeros 15 días corresponden al "
            "modelo V12. Entre los días 16 y 30 se "
            "presenta una extensión amortiguada "
            "de tendencia."
        )

    # ========================================================
    # LLUVIA FUTURA
    # ========================================================

    st.divider()

    st.subheader(
        "🌧️ Precipitación utilizada"
    )

    rain_cols = detectar_columnas_lluvia(
        exog_future
    )

    if rain_cols:

        rain = exog_future.copy()

        rain[
            "datetime"
        ] = pd.to_datetime(
            rain[
                "datetime"
            ],
            errors="coerce",
        )

        rain = rain.head(
            FORECAST_DAYS
        )

        rain_fig = go.Figure()

        for column in rain_cols:

            values = pd.to_numeric(
                rain[column],
                errors="coerce",
            )

            rain_fig.add_trace(
                go.Bar(
                    x=rain[
                        "datetime"
                    ],
                    y=values,
                    name=str(column),
                )
            )

        rain_fig.update_layout(
            height=360,
            hovermode="x unified",
            yaxis_title=
                "Precipitación (mm/día)",
            barmode="group",
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No hay precipitación "
            "futura disponible."
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    st.subheader(
        "💧 Caudal"
    )

    q_col = detectar_columna_caudal(
        exog_history
    )

    if q_col:

        q = exog_history.copy()

        q[
            "datetime"
        ] = pd.to_datetime(
            q[
                "datetime"
            ],
            errors="coerce",
        )

        q[
            q_col
        ] = pd.to_numeric(
            q[
                q_col
            ],
            errors="coerce",
        )

        q = q.dropna(
            subset=[
                "datetime",
                q_col,
            ]
        )

        tq = tendencia_caudal(
            q,
            q_col,
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
                ] is not None
                else "Sin dato"
            ),
        )

        q2.metric(
            "Variación 3 días",
            (
                f"{tq['delta_3']:+,.0f} m³/s"
                if tq[
                    "delta_3"
                ] is not None
                else "Sin dato"
            ),
        )

        q3.metric(
            "Variación 7 días",
            (
                f"{tq['delta_7']:+,.0f} m³/s"
                if tq[
                    "delta_7"
                ] is not None
                else "Sin dato"
            ),
        )

        q4.metric(
            "Tendencia",
            tq[
                "estado"
            ],
        )

        q_plot = q.tail(
            180
        )

        q_fig = go.Figure()

        q_fig.add_trace(
            go.Scatter(
                x=q_plot[
                    "datetime"
                ],
                y=q_plot[
                    q_col
                ],
                mode="lines",
                name=
                    "Caudal observado",
            )
        )

        q_fig.update_layout(
            height=380,
            hovermode="x unified",
            yaxis_title=
                "Caudal (m³/s)",
        )

        q_fig.update_xaxes(
            tickformat="%d/%m",
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No existe una serie "
            "de caudal utilizable."
        )

    # ========================================================
    # ESCENARIO HISTÓRICO
    # ========================================================

    if render_stress_scenario is not None:

        with st.expander(
            "⚠️ Escenario histórico de estrés · 60 días"
        ):

            try:

                render_stress_scenario(
                    df=df_hist,
                    models=models,
                    exog_history=
                        exog_history,
                    upstream_history=
                        upstream_history,
                )

            except Exception as exc:

                st.warning(
                    "El escenario de estrés "
                    "todavía requiere adaptación "
                    "completa a V12. "
                    f"Detalle: {exc}"
                )

    # ========================================================
    # IMPORTANCIA DE VARIABLES
    # ========================================================

    importance = models.get(
        "importance",
        pd.DataFrame(),
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
                .head(20)
                .copy()
            )

            fig_imp = go.Figure()

            fig_imp.add_trace(
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

            fig_imp.update_layout(
                height=600,
                xaxis_title=
                    "Importancia relativa",
                yaxis_title=
                    "Variable",
            )

            fig_imp.update_yaxes(
                autorange=
                    "reversed"
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True,
            )

    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y alcance"
    ):

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
            **V12.0 cambia el enfoque del pronóstico.**

            La fecha seleccionada se utiliza como corte temporal:
            el modelo no debe usar información posterior para construir
            el estado inicial.

            El nivel futuro de San Nicolás se calcula a partir de la
            **variación diaria estimada**, incorporando el nivel local,
            niveles aguas arriba, cambios de nivel, caudal, tendencia
            de caudal, precipitación y acumulados de lluvia.

            Además se calculan relaciones históricas de propagación
            entre estaciones y una relación específica
            **Corrientes → San Nicolás**.

            Las relaciones estadísticas no constituyen una ecuación
            hidráulica determinista ni sustituyen modelos oficiales.
            """
        )

        st.warning(
            "Esta plataforma es experimental e informativa. "
            "No reemplaza alertas, pronósticos ni comunicaciones "
            "de organismos oficiales."
        )

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
    Predicción y análisis de propagación: **modelo experimental propio**
    """
)

st.warning(
    "Los resultados tienen carácter experimental e informativo. "
    "Ante situaciones de riesgo deben consultarse las comunicaciones "
    "oficiales de las autoridades y organismos competentes."
)

st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Modelo de propagación hidrológica | "
    "Pronóstico principal: 15 días"
)
