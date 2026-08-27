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
    resumen_niveles_estaciones,
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
# APP V11.2
# PROPAGACIÓN HIDROLÓGICA
# ============================================================

APP_VERSION = "V11.2"

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
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1550px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.50rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    div[data-testid="stAlert"] {
        padding-top: 0.65rem;
        padding-bottom: 0.65rem;
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
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**.

    El modelo integra:

    **nivel observado · estaciones aguas arriba · propagación histórica ·
    caudal · precipitación**
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
        days=365
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
    "Pronóstico principal: **15 días**"
)

st.sidebar.write(
    "Tendencia extendida: **30 días**"
)

st.sidebar.write(
    "Escenario hipotético: **60 días**"
)


st.sidebar.subheader(
    "Modelo"
)

st.sidebar.write(
    "Propagación de niveles aguas arriba"
)

st.sidebar.write(
    "Relación histórica Corrientes → San Nicolás"
)

st.sidebar.write(
    "Escala fija: **0–7 m**"
)


st.sidebar.divider()

st.sidebar.caption(
    "Nivel: INA A5"
)

st.sidebar.caption(
    "Lluvia: Open-Meteo"
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

    x = df.copy()

    if "datetime" not in x.columns:
        return pd.DataFrame()

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
        utc=True,
    )

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
# TEXTO TENDENCIA
# ============================================================

def texto_tendencia(delta):

    if delta is None:
        return "Sin comparación"

    try:

        delta = float(delta)

    except Exception:

        return "Sin comparación"

    if not np.isfinite(delta):
        return "Sin comparación"

    if delta > 0.01:
        return "↑ Creciendo"

    if delta < -0.01:
        return "↓ Bajando"

    return "→ Estable"


# ============================================================
# NOMBRE DE ESTACIÓN
# ============================================================

def normalizar_estacion(texto):

    return (
        str(texto)
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace(" ", "_")
    )


# ============================================================
# TABLA NIVELES AGUAS ARRIBA
# ============================================================

def construir_tabla_upstream(
    upstream_history,
    upstream_meta,
):

    rows = []

    if not isinstance(
        upstream_meta,
        dict,
    ):
        return pd.DataFrame()

    for station, info in upstream_meta.items():

        col = (
            "nivel_"
            + normalizar_estacion(
                station
            )
        )

        actual = None
        anterior = None
        variacion = None
        fecha_ultima = None

        if (
            isinstance(
                upstream_history,
                pd.DataFrame,
            )
            and not upstream_history.empty
            and col
            in upstream_history.columns
        ):

            temp = upstream_history[
                [
                    "datetime",
                    col,
                ]
            ].copy()

            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce",
            )

            temp = (
                temp
                .dropna(
                    subset=[col]
                )
                .sort_values(
                    "datetime"
                )
            )

            if not temp.empty:

                actual = float(
                    temp[col].iloc[-1]
                )

                fecha_ultima = (
                    temp["datetime"].iloc[-1]
                )

                if len(temp) >= 2:

                    anterior = float(
                        temp[col].iloc[-2]
                    )

                    variacion = (
                        actual
                        - anterior
                    )

        series_id = None

        if isinstance(
            info,
            dict,
        ):
            series_id = info.get(
                "series_id"
            )

        rows.append(
            {
                "Estación":
                    station,

                "Nivel actual (m)":
                    (
                        round(
                            actual,
                            2,
                        )
                        if actual
                        is not None
                        else np.nan
                    ),

                "Medición anterior (m)":
                    (
                        round(
                            anterior,
                            2,
                        )
                        if anterior
                        is not None
                        else np.nan
                    ),

                "Variación (m)":
                    (
                        round(
                            variacion,
                            2,
                        )
                        if variacion
                        is not None
                        else np.nan
                    ),

                "Tendencia":
                    (
                        texto_tendencia(
                            variacion
                        )
                        if actual
                        is not None
                        else "Sin datos"
                    ),

                "Última fecha":
                    (
                        pd.to_datetime(
                            fecha_ultima
                        ).strftime(
                            "%d/%m/%Y"
                        )
                        if fecha_ultima
                        is not None
                        else ""
                    ),

                "Serie INA":
                    series_id,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXTENSIÓN A 30 DÍAS
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

    last_date = (
        f["datetime"].iloc[-1]
    )

    last_level = float(
        f["prediction"].iloc[-1]
    )

    if len(f) >= 5:

        recent = (
            f["prediction"]
            .tail(5)
            .to_numpy(
                dtype=float
            )
        )

        slope = float(
            np.polyfit(
                np.arange(
                    len(recent)
                ),
                recent,
                1,
            )[0]
        )

    else:

        niveles = (
            pd.to_numeric(
                df["nivel"],
                errors="coerce",
            )
            .dropna()
            .tail(7)
        )

        if len(niveles) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(niveles)
                    ),
                    niveles.to_numpy(
                        dtype=float
                    ),
                    1,
                )[0]
            )

        else:

            slope = 0.0

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

                "delta_prediction":
                    daily_change,
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
# RELACIÓN CORRIENTES → SAN NICOLÁS
# ============================================================

def construir_relacion_corrientes(
    df,
    upstream_history,
    relation_summary,
):

    if (
        df is None
        or df.empty
        or upstream_history is None
        or upstream_history.empty
        or "nivel_corrientes"
        not in upstream_history.columns
    ):
        return (
            pd.DataFrame(),
            None,
            None,
        )

    lag = None
    correlation = None

    if (
        isinstance(
            relation_summary,
            pd.DataFrame,
        )
        and not relation_summary.empty
        and "estacion"
        in relation_summary.columns
    ):

        temp_summary = (
            relation_summary.copy()
        )

        mask = (
            temp_summary[
                "estacion"
            ]
            .astype(str)
            .str.lower()
            ==
            "corrientes"
        )

        if mask.any():

            row = (
                temp_summary[
                    mask
                ].iloc[0]
            )

            lag_value = row.get(
                "mejor_lag_dias"
            )

            corr_value = row.get(
                "correlacion"
            )

            if pd.notna(
                lag_value
            ):
                lag = int(
                    lag_value
                )

            if pd.notna(
                corr_value
            ):
                correlation = float(
                    corr_value
                )

    if lag is None:
        lag = 7

    sn = df[
        [
            "datetime",
            "nivel",
        ]
    ].copy()

    sn["datetime"] = pd.to_datetime(
        sn["datetime"],
        errors="coerce",
        utc=True,
    )

    sn["datetime"] = (
        sn["datetime"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    sn = (
        sn
        .groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .mean()
    )

    corr = upstream_history[
        [
            "datetime",
            "nivel_corrientes",
        ]
    ].copy()

    corr["datetime"] = pd.to_datetime(
        corr["datetime"],
        errors="coerce",
    ).dt.normalize()

    corr[
        "nivel_corrientes"
    ] = pd.to_numeric(
        corr[
            "nivel_corrientes"
        ],
        errors="coerce",
    )

    corr = (
        corr
        .dropna(
            subset=[
                "datetime",
                "nivel_corrientes",
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )[
            "nivel_corrientes"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
    )

    # El nivel de Corrientes observado hoy
    # se compara con San Nicolás varios días después.
    corr[
        "datetime_sn"
    ] = (
        corr[
            "datetime"
        ]
        + pd.to_timedelta(
            lag,
            unit="D",
        )
    )

    relation = sn.merge(
        corr[
            [
                "datetime_sn",
                "nivel_corrientes",
            ]
        ],
        left_on=
            "datetime",
        right_on=
            "datetime_sn",
        how="inner",
    )

    relation = relation.dropna(
        subset=[
            "nivel",
            "nivel_corrientes",
        ]
    )

    if (
        correlation is None
        and len(relation) >= 10
    ):

        correlation = relation[
            "nivel"
        ].corr(
            relation[
                "nivel_corrientes"
            ]
        )

        if pd.notna(
            correlation
        ):
            correlation = float(
                correlation
            )

    return (
        relation,
        lag,
        correlation,
    )


# ============================================================
# VALIDAR FECHAS
# ============================================================

if desde > hasta:

    st.sidebar.error(
        "La fecha Desde no puede ser posterior a Hasta."
    )


# ============================================================
# ACTUALIZACIÓN
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
        # SAN NICOLÁS - INA A5
        # ====================================================

        with st.spinner(
            "Consultando nivel de San Nicolás..."
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
                    "No se obtuvieron niveles válidos "
                    "de San Nicolás."
                )

            else:

                # ============================================
                # VARIABLES EXTERNAS
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

                        exog_history = (
                            pd.DataFrame()
                        )

                        exog_future = (
                            pd.DataFrame()
                        )

                        exog_meta = {}

                        st.warning(
                            "Variables externas parcialmente "
                            f"disponibles: {exc}"
                        )

                # ============================================
                # AGUAS ARRIBA
                # ============================================

                with st.spinner(
                    "Consultando niveles aguas arriba..."
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
                            "No fue posible obtener todas "
                            "las estaciones aguas arriba. "
                            f"{exc}"
                        )

                # ============================================
                # MODELO
                # ============================================

                with st.spinner(
                    "Analizando propagación y entrenando modelo..."
                ):

                    try:

                        (
                            models,
                            metrics,
                        ) = train(
                            df=df,
                            exog_history=
                                exog_history,
                            upstream_history=
                                upstream_history,
                        )

                        forecast = predict(
                            df=df,
                            models=models,
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

                        relation_summary = (
                            models.get(
                                "relation_summary",
                                pd.DataFrame(),
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

                        relation_summary = (
                            pd.DataFrame()
                        )

                        st.error(
                            "No fue posible generar el modelo. "
                            f"Detalle: {exc}"
                        )

                # ============================================
                # SESSION STATE
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
                    "relation_summary"
                ] = relation_summary

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
        "Seleccione el período y presione "
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

    upstream_history = st.session_state.get(
        "upstream_history",
        pd.DataFrame(),
    )

    upstream_meta = st.session_state.get(
        "upstream_meta",
        {},
    )

    relation_summary = st.session_state.get(
        "relation_summary",
        pd.DataFrame(),
    )

    actualizado = st.session_state.get(
        "actualizado"
    )


    # ========================================================
    # SITUACIÓN SAN NICOLÁS
    # ========================================================

    niveles = pd.to_numeric(
        df["nivel"],
        errors="coerce",
    ).dropna()

    if niveles.empty:

        st.error(
            "No hay niveles válidos de San Nicolás."
        )

        st.stop()

    nivel_actual = float(
        niveles.iloc[-1]
    )

    delta_actual = None

    if len(niveles) >= 2:

        delta_actual = (
            nivel_actual
            - float(
                niveles.iloc[-2]
            )
        )

    ultima_fecha = (
        df["datetime"].iloc[-1]
    )

    st.subheader(
        "📊 Situación observada · San Nicolás"
    )

    c1, c2, c3, c4, c5 = st.columns(
        5
    )

    c1.metric(
        "Nivel actual",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_actual:+.2f} m"
            if delta_actual
            is not None
            else None
        ),
    )

    c2.metric(
        "Mínimo",
        f"{niveles.min():.2f} m",
    )

    c3.metric(
        "Máximo",
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

    st.caption(
        "Última observación INA: "
        + pd.to_datetime(
            ultima_fecha
        ).strftime(
            "%d/%m/%Y %H:%M"
        )
    )


    # ========================================================
    # PRONÓSTICO 15 DÍAS
    # ========================================================

    st.subheader(
        "📈 Nivel observado y pronóstico · 15 días"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["nivel"],
            mode="lines",
            name=
                "Nivel observado INA",
            line=dict(
                width=3,
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
                    "Pronóstico 15 días",
                line=dict(
                    width=3,
                ),
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
                    hoverinfo="skip",
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
                    line=dict(
                        width=0,
                    ),
                    fill="tonexty",
                    name=
                        "Incertidumbre",
                )
            )

    fig.update_layout(
        height=490,
        hovermode=
            "x unified",
        legend=dict(
            orientation="h",
            y=1.07,
        ),
    )

    fig.update_xaxes(
        tickformat="%d/%m",
        title_text="Fecha",
    )

    fig.update_yaxes(
        title_text="Nivel (m)",
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


    # ========================================================
    # NIVELES AGUAS ARRIBA
    # ========================================================

    st.subheader(
        "🌊 Niveles aguas arriba"
    )

    tabla_upstream = (
        construir_tabla_upstream(
            upstream_history,
            upstream_meta,
        )
    )

    if not tabla_upstream.empty:

        st.dataframe(
            tabla_upstream,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No se recuperaron niveles aguas arriba."
        )


    # ========================================================
    # PROPAGACIÓN HACIA SAN NICOLÁS
    # ========================================================

    st.subheader(
        "🛰️ Propagación hacia San Nicolás"
    )

    st.caption(
        "El retardo representa el número de días con que "
        "la señal histórica de una estación aguas arriba "
        "presenta la mejor asociación con San Nicolás."
    )

    if (
        isinstance(
            relation_summary,
            pd.DataFrame,
        )
        and not relation_summary.empty
    ):

        tabla_relacion = (
            relation_summary.copy()
        )

        rename = {
            "estacion":
                "Estación",

            "nivel_actual":
                "Nivel actual (m)",

            "nivel_anterior":
                "Nivel anterior (m)",

            "variacion":
                "Variación (m)",

            "mejor_lag_dias":
                "Retardo estimado (días)",

            "correlacion":
                "Correlación histórica",
        }

        tabla_relacion = (
            tabla_relacion.rename(
                columns=rename
            )
        )

        for col in [
            "Nivel actual (m)",
            "Nivel anterior (m)",
            "Variación (m)",
            "Correlación histórica",
        ]:

            if col in tabla_relacion.columns:

                tabla_relacion[col] = (
                    pd.to_numeric(
                        tabla_relacion[
                            col
                        ],
                        errors="coerce",
                    )
                    .round(2)
                )

        st.dataframe(
            tabla_relacion,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Todavía no existen suficientes datos "
            "para calcular relaciones de propagación."
        )


    # ========================================================
    # CORRIENTES → SAN NICOLÁS
    # ========================================================

    st.subheader(
        "📡 Relación histórica · Corrientes → San Nicolás"
    )

    (
        corr_relation,
        corr_lag,
        corr_value,
    ) = construir_relacion_corrientes(
        df,
        upstream_history,
        relation_summary,
    )

    corr_actual = None
    corr_delta = None

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and "nivel_corrientes"
        in upstream_history.columns
    ):

        corr_values = (
            pd.to_numeric(
                upstream_history[
                    "nivel_corrientes"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if not corr_values.empty:

            corr_actual = float(
                corr_values.iloc[-1]
            )

            if len(corr_values) >= 2:

                corr_delta = (
                    corr_actual
                    - float(
                        corr_values.iloc[-2]
                    )
                )

    rc1, rc2, rc3, rc4 = (
        st.columns(
            4
        )
    )

    rc1.metric(
        "Nivel Corrientes",
        (
            f"{corr_actual:.2f} m"
            if corr_actual
            is not None
            else "Sin dato"
        ),
        (
            f"{corr_delta:+.2f} m"
            if corr_delta
            is not None
            else None
        ),
    )

    rc2.metric(
        "Retardo estimado",
        (
            f"{corr_lag} días"
            if corr_lag
            is not None
            else "Sin dato"
        ),
    )

    rc3.metric(
        "Correlación histórica",
        (
            f"{corr_value:.2f}"
            if corr_value
            is not None
            and np.isfinite(
                corr_value
            )
            else "Sin dato"
        ),
    )

    rc4.metric(
        "Tendencia Corrientes",
        texto_tendencia(
            corr_delta
        ),
    )


    # ========================================================
    # GRÁFICO CORRIENTES / SAN NICOLÁS
    # ========================================================

    if (
        not corr_relation.empty
        and len(
            corr_relation
        ) >= 10
    ):

        relation_fig = go.Figure()

        relation_fig.add_trace(
            go.Scatter(
                x=corr_relation[
                    "nivel_corrientes"
                ],
                y=corr_relation[
                    "nivel"
                ],
                mode="markers",
                name=
                    "Observaciones históricas",
                hovertemplate=(
                    "Corrientes: %{x:.2f} m"
                    "<br>"
                    "San Nicolás: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

        try:

            x_values = (
                corr_relation[
                    "nivel_corrientes"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y_values = (
                corr_relation[
                    "nivel"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            slope, intercept = (
                np.polyfit(
                    x_values,
                    y_values,
                    1,
                )
            )

            x_line = np.linspace(
                np.nanmin(
                    x_values
                ),
                np.nanmax(
                    x_values
                ),
                100,
            )

            y_line = (
                intercept
                + slope
                * x_line
            )

            relation_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=
                        "Relación histórica",
                )
            )

        except Exception:

            pass

        relation_fig.update_layout(
            height=420,
            xaxis_title=(
                "Nivel Corrientes (m) "
                f"· desplazado {corr_lag} días"
            ),
            yaxis_title=
                "Nivel San Nicolás (m)",
        )

        relation_fig.update_yaxes(
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )

        st.plotly_chart(
            relation_fig,
            use_container_width=True,
        )

        st.caption(
            "La relación se calcula desplazando las observaciones "
            "de Corrientes según el retardo histórico estimado. "
            "La correlación indica asociación estadística y no "
            "implica por sí sola causalidad."
        )


    # ========================================================
    # GRÁFICO TEMPORAL AGUAS ARRIBA
    # ========================================================

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
    ):

        level_cols = [
            col
            for col
            in upstream_history.columns
            if col.startswith(
                "nivel_"
            )
        ]

        if level_cols:

            with st.expander(
                "📈 Evolución de todas las estaciones aguas arriba",
                expanded=False,
            ):

                upstream_fig = (
                    go.Figure()
                )

                for col in level_cols:

                    station_name = (
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

                    upstream_fig.add_trace(
                        go.Scatter(
                            x=upstream_history[
                                "datetime"
                            ],
                            y=upstream_history[
                                col
                            ],
                            mode="lines",
                            name=
                                station_name,
                        )
                    )

                upstream_fig.update_layout(
                    height=480,
                    hovermode=
                        "x unified",
                    legend=dict(
                        orientation="h",
                        y=1.08,
                    ),
                )

                upstream_fig.update_xaxes(
                    tickformat="%d/%m",
                )

                upstream_fig.update_yaxes(
                    title_text=
                        "Nivel (m)",
                    range=[
                        Y_MIN,
                        Y_MAX,
                    ],
                    dtick=
                        Y_STEP,
                )

                st.plotly_chart(
                    upstream_fig,
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

        trend_fig = (
            go.Figure()
        )

        recent = df.tail(
            30
        )

        trend_fig.add_trace(
            go.Scatter(
                x=recent[
                    "datetime"
                ],
                y=recent[
                    "nivel"
                ],
                mode="lines",
                name=
                    "Observado",
            )
        )

        f15 = forecast30.head(
            15
        )

        trend_fig.add_trace(
            go.Scatter(
                x=f15[
                    "datetime"
                ],
                y=f15[
                    "prediction"
                ],
                mode=
                    "lines+markers",
                name=
                    "Modelo 1–15 días",
            )
        )

        f16_30 = (
            forecast30.iloc[
                15:
            ]
        )

        if not f16_30.empty:

            trend_fig.add_trace(
                go.Scatter(
                    x=f16_30[
                        "datetime"
                    ],
                    y=f16_30[
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

        trend_fig.update_layout(
            height=420,
            hovermode=
                "x unified",
            legend=dict(
                orientation="h",
                y=1.08,
            ),
        )

        trend_fig.update_xaxes(
            tickformat="%d/%m",
        )

        trend_fig.update_yaxes(
            title_text=
                "Nivel (m)",
            range=[
                Y_MIN,
                Y_MAX,
            ],
            dtick=Y_STEP,
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True,
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

        p1, p2, p3 = st.columns(
            3
        )

        p1.metric(
            "Acumulado previsto",
            f"{rain['precip_mm'].sum():.1f} mm",
        )

        p2.metric(
            "Máximo diario",
            f"{rain['precip_mm'].max():.1f} mm",
        )

        p3.metric(
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
                name=
                    "Precipitación",
            )
        )

        rain_fig.update_layout(
            height=310,
            yaxis_title=
                "mm/día",
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
            "No hay pronóstico de precipitación disponible."
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

        q = exog_history[
            [
                "datetime",
                "caudal_m3s",
            ]
        ].copy()

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

        if not q.empty:

            q_actual = float(
                q[
                    "caudal_m3s"
                ].iloc[-1]
            )

            q_delta = None

            if len(q) >= 2:

                q_delta = (
                    q_actual
                    - float(
                        q[
                            "caudal_m3s"
                        ].iloc[-2]
                    )
                )

            q1, q2 = st.columns(
                2
            )

            q1.metric(
                "Caudal actual",
                f"{q_actual:,.0f} m³/s",
            )

            q2.metric(
                "Variación",
                (
                    f"{q_delta:+,.0f} m³/s"
                    if q_delta
                    is not None
                    else "Sin comparación"
                ),
            )

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
                height=340,
                yaxis_title=
                    "m³/s",
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
                "No hay caudal válido disponible."
            )

    else:

        st.info(
            "No se encontró una serie de caudal."
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
            "No fue posible construir el escenario de 60 días. "
            f"{exc}"
        )


    # ========================================================
    # MODELO / IMPORTANCIA
    # ========================================================

    with st.expander(
        "🧠 Diagnóstico del modelo"
    ):

        rmse = metrics.get(
            "RMSE"
        )

        if rmse is not None:

            st.metric(
                "RMSE histórico",
                f"{float(rmse):.3f} m",
            )

        st.write(
            "**Usa precipitación:**",
            (
                "Sí"
                if models.get(
                    "uses_rain",
                    False,
                )
                else "No"
            ),
        )

        st.write(
            "**Usa caudal:**",
            (
                "Sí"
                if models.get(
                    "uses_caudal",
                    False,
                )
                else "No"
            ),
        )

        st.write(
            "**Usa niveles aguas arriba:**",
            (
                "Sí"
                if models.get(
                    "uses_upstream",
                    False,
                )
                else "No"
            ),
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

            st.markdown(
                "#### Variables más influyentes"
            )

            st.dataframe(
                importance.head(
                    20
                ),
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # ACTUALIZACIÓN
    # ========================================================

    if actualizado:

        st.caption(
            "Última actualización: "
            f"{actualizado.strftime('%d/%m/%Y %H:%M')}"
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    f"Paraná · San Nicolás {APP_VERSION} | "
    "Nivel observado: INA A5 | "
    "Modelo experimental propio"
)

st.warning(
    "La información y las proyecciones son experimentales. "
    "No reemplazan alertas ni pronósticos emitidos por "
    "organismos oficiales."
)
