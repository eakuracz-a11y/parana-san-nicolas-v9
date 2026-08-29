# ============================================================
# PARANÁ · SAN NICOLÁS
# app.py
# V11.10 COMPLETO
#
# SISTEMA DE PRONÓSTICO HIDROLÓGICO MULTIESTACIÓN
#
# ------------------------------------------------------------
# - Nivel San Nicolás INA
# - Niveles aguas arriba INA
# - Caudales por estación INA
# - Lluvia por estación Open-Meteo
# - Propagación Corrientes -> San Nicolás
# - Retardos por tramo
# - Eventos históricos similares
# - Pronóstico:
#       1-15 días
#       16-30 días
#       31-45 días
#       46-60 días
# - Bandas de incertidumbre
# - Comparación año contra año
# - Escala automática de nivel
# - Diagnóstico de variables utilizadas
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import (
    date,
    datetime,
    timedelta,
)


# ============================================================
# INA
# ============================================================

from src.ina import (
    observed,
)


# ============================================================
# MODELO
# ============================================================

from src.model import (
    train,
    predict,
    resumen_niveles_estaciones,
)


# ============================================================
# VARIABLES EXÓGENAS
# ============================================================

from src.exogenous import (
    get_exogenous_data,
)


# ============================================================
# AGUAS ARRIBA
# ============================================================

from src.upstream import (
    get_upstream_history,
)


# ============================================================
# HIDROLOGÍA
# ============================================================

try:

    from src.hydrology import (
        analizar_corrientes_san_nicolas,
    )

except Exception:

    analizar_corrientes_san_nicolas = None


# ============================================================
# VERSIÓN
# ============================================================

APP_VERSION = "V11.10"


# ============================================================
# CONFIGURACIÓN
# ============================================================

FORECAST_DAYS = 60

VISUAL_DEFAULT_DAYS = 120

# INA puede tener historia anterior.
# Para nivel de San Nicolás mantenemos toda la historia.
LEVEL_HISTORY_START = "1900-01-01"

# Para variables meteorológicas y caudales multivariables
# usamos un período amplio pero compatible con fuentes
# meteorológicas históricas.
EXOG_HISTORY_START = "1940-01-01"


# ============================================================
# ESTACIONES
# ============================================================

STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
    "San Nicolás",
]


LEVEL_COLUMNS = {

    "Corrientes":
        "nivel_corrientes",

    "Goya":
        "nivel_goya",

    "La Paz":
        "nivel_la_paz",

    "Paraná":
        "nivel_parana",

    "Diamante":
        "nivel_diamante",

    "Rosario":
        "nivel_rosario",

    "Villa Constitución":
        "nivel_villa_constitucion",

    "San Nicolás":
        "nivel",
}


FLOW_COLUMNS = {

    "Corrientes":
        "q_corrientes",

    "Goya":
        "q_goya",

    "La Paz":
        "q_la_paz",

    "Paraná":
        "q_parana",

    "Diamante":
        "q_diamante",

    "Rosario":
        "q_rosario",

    "Villa Constitución":
        "q_villa_constitucion",

    "San Nicolás":
        "q_san_nicolas",
}


RAIN_COLUMNS = {

    "Corrientes":
        "rain_corrientes",

    "Goya":
        "rain_goya",

    "La Paz":
        "rain_la_paz",

    "Paraná":
        "rain_parana",

    "Diamante":
        "rain_diamante",

    "Rosario":
        "rain_rosario",

    "Villa Constitución":
        "rain_villa_constitucion",

    "San Nicolás":
        "rain_san_nicolas",
}


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# CSS RESPONSIVE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1 {
        margin-bottom: 0.25rem;
    }

    [data-testid="stMetric"] {
        background: rgba(127,127,127,0.055);
        border: 1px solid rgba(127,127,127,0.15);
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.80rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.42rem;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.76rem;
    }

    .small-note {
        font-size: 0.80rem;
        opacity: 0.75;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-top: 0.8rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1.05rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.72rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES
# ============================================================

def datetime_naive(values):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def safe_float(
    value,
    default=None,
):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def fmt_num(
    value,
    decimals=2,
    suffix="",
):

    value = safe_float(
        value
    )

    if value is None:
        return "Sin dato"

    return (
        f"{value:,.{decimals}f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
        + suffix
    )


def fmt_date(value):

    try:

        value = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(value):
            return "Sin dato"

        return value.strftime(
            "%d/%m/%Y"
        )

    except Exception:

        return "Sin dato"


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
    ] = datetime_naive(
        x[
            "datetime"
        ]
    )


    if "nivel" in x.columns:

        level_col = "nivel"

    elif "value" in x.columns:

        level_col = "value"

    elif "nivel_san_nicolas" in x.columns:

        level_col = "nivel_san_nicolas"

    else:

        return pd.DataFrame()


    x[
        "nivel"
    ] = pd.to_numeric(
        x[
            level_col
        ],
        errors="coerce",
    )


    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )


    x = x[
        (
            x["nivel"]
            >= -5
        )
        &
        (
            x["nivel"]
            <= 20
        )
    ]


    x[
        "datetime"
    ] = (
        x[
            "datetime"
        ]
        .dt
        .normalize()
    )


    x = (
        x
        .groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    return x


# ============================================================
# RANGO AUTOMÁTICO Y
# ============================================================

def auto_y_range(
    values,
    minimum_padding=0.20,
):

    numeric = []


    for values_item in values:

        try:

            series = (
                pd.to_numeric(
                    values_item,
                    errors="coerce",
                )
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            numeric.extend(
                series.tolist()
            )

        except Exception:
            pass


    if not numeric:

        return None


    ymin = min(
        numeric
    )

    ymax = max(
        numeric
    )


    span = ymax - ymin


    padding = max(
        span * 0.12,
        minimum_padding,
    )


    return [
        ymin - padding,
        ymax + padding,
    ]


# ============================================================
# TENDENCIA
# ============================================================

def nivel_tendencia(
    df,
    days=7,
):

    result = {

        "actual":
            None,

        "delta":
            None,

        "estado":
            "Sin datos",

        "slope":
            None,
    }


    if (
        df is None
        or df.empty
        or "nivel"
        not in df.columns
    ):

        return result


    values = (
        pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        )
        .dropna()
    )


    if values.empty:

        return result


    actual = float(
        values.iloc[-1]
    )


    result[
        "actual"
    ] = actual


    lookback = min(
        days,
        len(values) - 1,
    )


    if lookback > 0:

        previous = float(
            values.iloc[
                -lookback - 1
            ]
        )


        delta = (
            actual
            - previous
        )


        result[
            "delta"
        ] = delta


    recent = values.tail(
        min(
            days + 1,
            len(values),
        )
    )


    if len(recent) >= 3:

        slope = float(
            np.polyfit(
                np.arange(
                    len(recent),
                    dtype=float,
                ),
                recent.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    else:

        slope = 0.0


    result[
        "slope"
    ] = slope


    if slope > 0.01:

        result[
            "estado"
        ] = "↑ Creciente"

    elif slope < -0.01:

        result[
            "estado"
        ] = "↓ Decreciente"

    else:

        result[
            "estado"
        ] = "→ Estable"


    return result


# ============================================================
# CAUDAL PRINCIPAL
# ============================================================

def detectar_caudal_principal(
    exog_history,
):

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
    ):

        return None


    priority = [

        "q_san_nicolas",
        "q_villa_constitucion",
        "q_rosario",
        "q_diamante",
        "q_parana",
        "q_la_paz",
        "q_goya",
        "q_corrientes",
        "caudal_m3s",
    ]


    for col in priority:

        if col not in exog_history.columns:
            continue


        valid = (
            pd.to_numeric(
                exog_history[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )


        if len(valid) >= 3:

            return col


    return None


def flow_station_from_column(
    column,
):

    if column is None:
        return None


    mapping = {

        "q_corrientes":
            "Corrientes",

        "q_goya":
            "Goya",

        "q_la_paz":
            "La Paz",

        "q_parana":
            "Paraná",

        "q_diamante":
            "Diamante",

        "q_rosario":
            "Rosario",

        "q_villa_constitucion":
            "Villa Constitución",

        "q_san_nicolas":
            "San Nicolás",

        "caudal_m3s":
            "Serie principal",
    }


    return mapping.get(
        column,
        column,
    )


def tendencia_caudal(
    df,
    column,
):

    result = {

        "actual":
            None,

        "delta_3":
            None,

        "delta_7":
            None,

        "estado":
            "Sin datos",
    }


    if (
        df is None
        or df.empty
        or column is None
        or column not in df.columns
    ):

        return result


    x = (
        pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )
        .dropna()
    )


    if x.empty:

        return result


    actual = float(
        x.iloc[-1]
    )


    result[
        "actual"
    ] = actual


    if len(x) >= 4:

        result[
            "delta_3"
        ] = (
            actual
            - float(
                x.iloc[-4]
            )
        )


    if len(x) >= 8:

        result[
            "delta_7"
        ] = (
            actual
            - float(
                x.iloc[-8]
            )
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
                np.arange(
                    len(recent),
                    dtype=float,
                ),
                recent.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    else:

        slope = 0.0


    threshold = max(
        abs(
            actual
        )
        * 0.0015,
        1.0,
    )


    if slope > threshold:

        result[
            "estado"
        ] = "↑ Creciendo"

    elif slope < -threshold:

        result[
            "estado"
        ] = "↓ Bajando"

    else:

        result[
            "estado"
        ] = "→ Estable"


    return result


# ============================================================
# PREPARAR UPSTREAM PARA GRÁFICOS
# ============================================================

def preparar_upstream_visual(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime"
        not in df.columns
    ):

        return pd.DataFrame()


    x = df.copy()


    x[
        "datetime"
    ] = datetime_naive(
        x[
            "datetime"
        ]
    )


    return (
        x
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


# ============================================================
# TABLA ACTUAL DEL CORREDOR
# ============================================================

def tabla_corredor_actual(
    df,
    upstream_history,
    exog_history,
):

    rows = []


    upstream = preparar_upstream_visual(
        upstream_history
    )


    for station in STATIONS:

        level_col = LEVEL_COLUMNS[
            station
        ]


        if station == "San Nicolás":

            source = df

        else:

            source = upstream


        level = None
        previous = None
        delta = None


        if (
            source is not None
            and not source.empty
            and level_col
            in source.columns
        ):

            values = (
                source[
                    [
                        "datetime",
                        level_col,
                    ]
                ]
                .copy()
            )


            values[
                level_col
            ] = pd.to_numeric(
                values[
                    level_col
                ],
                errors="coerce",
            )


            values = (
                values
                .dropna(
                    subset=[
                        level_col
                    ]
                )
                .sort_values(
                    "datetime"
                )
            )


            if not values.empty:

                level = float(
                    values.iloc[
                        -1
                    ][
                        level_col
                    ]
                )


                if len(values) >= 2:

                    previous = float(
                        values.iloc[
                            -2
                        ][
                            level_col
                        ]
                    )


                    delta = (
                        level
                        - previous
                    )


        if delta is None:

            tendencia = "Sin dato"

        elif delta > 0.01:

            tendencia = "↑ Sube"

        elif delta < -0.01:

            tendencia = "↓ Baja"

        else:

            tendencia = "→ Estable"


        q_col = FLOW_COLUMNS[
            station
        ]


        flow = None


        if (
            exog_history is not None
            and not exog_history.empty
            and q_col
            in exog_history.columns
        ):

            q = (
                pd.to_numeric(
                    exog_history[
                        q_col
                    ],
                    errors="coerce",
                )
                .dropna()
            )


            if not q.empty:

                flow = float(
                    q.iloc[-1]
                )


        rain_col = RAIN_COLUMNS[
            station
        ]


        rain_7 = None


        if (
            exog_history is not None
            and not exog_history.empty
            and rain_col
            in exog_history.columns
        ):

            rain = (
                pd.to_numeric(
                    exog_history[
                        rain_col
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .tail(
                    7
                )
            )


            if not rain.empty:

                rain_7 = float(
                    rain.sum()
                )


        rows.append(
            {
                "Estación":
                    station,

                "Nivel actual (m)":
                    level,

                "Δ última medición (m)":
                    delta,

                "Tendencia":
                    tendencia,

                "Caudal (m³/s)":
                    flow,

                "Lluvia 7 días (mm)":
                    rain_7,
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPARACIÓN AÑO CONTRA AÑO
# ============================================================

def construir_comparacion_anual(
    source,
    value_col,
    start_month,
    start_day,
    end_month,
    end_day,
    years,
):

    if (
        source is None
        or source.empty
        or value_col
        not in source.columns
    ):

        return pd.DataFrame()


    x = source[
        [
            "datetime",
            value_col,
        ]
    ].copy()


    x[
        "datetime"
    ] = datetime_naive(
        x[
            "datetime"
        ]
    )


    x[
        value_col
    ] = pd.to_numeric(
        x[
            value_col
        ],
        errors="coerce",
    )


    x = x.dropna(
        subset=[
            "datetime",
            value_col,
        ]
    )


    output = []


    crosses_year = (
        (
            end_month,
            end_day,
        )
        <
        (
            start_month,
            start_day,
        )
    )


    for year in years:

        try:

            start_date = pd.Timestamp(
                year=int(
                    year
                ),
                month=int(
                    start_month
                ),
                day=int(
                    start_day
                ),
            )


            end_year = (
                int(
                    year
                )
                + 1
                if crosses_year
                else int(
                    year
                )
            )


            end_date = pd.Timestamp(
                year=end_year,
                month=int(
                    end_month
                ),
                day=int(
                    end_day
                ),
            )

        except Exception:

            continue


        period = x[
            (
                x[
                    "datetime"
                ]
                >= start_date
            )
            &
            (
                x[
                    "datetime"
                ]
                <= end_date
            )
        ].copy()


        if period.empty:
            continue


        period[
            "day_index"
        ] = (
            period[
                "datetime"
            ]
            - start_date
        ).dt.days


        period[
            "year_label"
        ] = str(
            year
        )


        output.append(
            period
        )


    if not output:

        return pd.DataFrame()


    return pd.concat(
        output,
        ignore_index=True,
    )


# ============================================================
# CABECERA
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)


st.caption(
    f"{APP_VERSION} · Pronóstico hidrológico multivariable"
)


st.markdown(
    """
    El modelo analiza el Paraná como un **sistema hidrológico
    aguas arriba → San Nicolás**, incorporando niveles,
    caudales, lluvias y propagación histórica.

    Los horizontes de **15, 30, 45 y 60 días** se calculan
    de manera continua, aumentando progresivamente la
    incertidumbre a medida que se extiende el horizonte.
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
        "Se utilizarán observaciones disponibles "
        "hasta esta fecha."
    ),
)


dias_historia_visual = st.sidebar.slider(
    "Historia visible en gráfico",
    min_value=30,
    max_value=365,
    value=VISUAL_DEFAULT_DAYS,
    step=30,
)


actualizar = st.sidebar.button(
    "🔄 Actualizar datos y modelo",
    type="primary",
    use_container_width=True,
)


st.sidebar.divider()


st.sidebar.subheader(
    "Horizontes"
)


st.sidebar.write(
    "🟠 **1–15 días:** pronóstico"
)


st.sidebar.write(
    "🟢 **16–30 días:** proyección"
)


st.sidebar.write(
    "🟣 **31–45 días:** escenario probabilístico"
)


st.sidebar.write(
    "🔴 **46–60 días:** tendencia extendida"
)


st.sidebar.caption(
    "La escala vertical de los gráficos es automática."
)


st.sidebar.caption(
    "Nivel y caudal: INA · Precipitación: Open-Meteo"
)


# ============================================================
# ACTUALIZACIÓN
# ============================================================

if actualizar:

    fin = fecha_base.strftime(
        "%Y-%m-%d"
    )


    inicio_visual = (
        fecha_base
        - timedelta(
            days=
                dias_historia_visual
        )
    ).strftime(
        "%Y-%m-%d"
    )


    # ========================================================
    # NIVEL VISIBLE SAN NICOLÁS
    # ========================================================

    with st.spinner(
        "Consultando nivel de San Nicolás..."
    ):

        try:

            raw, error = observed(
                inicio_visual,
                fin,
            )

        except Exception as exc:

            raw = pd.DataFrame()

            error = str(
                exc
            )


    if error:

        st.error(
            f"INA San Nicolás: {error}"
        )


    df = preparar_datos(
        raw
    )


    if df.empty:

        st.error(
            "No existen observaciones válidas de "
            "San Nicolás para la fecha seleccionada."
        )


    else:

        # ====================================================
        # HISTORIA SAN NICOLÁS
        # ====================================================

        with st.spinner(
            "Recuperando historial de San Nicolás..."
        ):

            try:

                hist_raw, hist_error = observed(
                    LEVEL_HISTORY_START,
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


            except Exception:

                df_hist = df.copy()


        # ====================================================
        # LLUVIA + CAUDALES MULTIESTACIÓN
        # ====================================================

        with st.spinner(
            "Consultando lluvias y caudales del corredor..."
        ):

            try:

                (
                    exog_history,
                    exog_future,
                    exog_meta,
                ) = get_exogenous_data(

                    EXOG_HISTORY_START,

                    fin,

                    forecast_days=
                        FORECAST_DAYS,
                )


            except Exception as exc:

                exog_history = (
                    pd.DataFrame()
                )

                exog_future = (
                    pd.DataFrame()
                )

                exog_meta = {
                    "error":
                        str(
                            exc
                        )
                }


                st.warning(
                    "No fue posible recuperar todas las "
                    "variables de lluvia/caudal. "
                    f"Detalle: {exc}"
                )


        # ====================================================
        # NIVELES AGUAS ARRIBA
        # ====================================================

        with st.spinner(
            "Consultando historial de niveles aguas arriba..."
        ):

            try:

                (
                    upstream_history,
                    upstream_meta,
                ) = get_upstream_history(
                    LEVEL_HISTORY_START,
                    fin,
                )


            except Exception as exc:

                upstream_history = (
                    pd.DataFrame()
                )

                upstream_meta = {
                    "error":
                        str(
                            exc
                        )
                }


                st.warning(
                    "No fue posible obtener todos los "
                    "niveles aguas arriba. "
                    f"Detalle: {exc}"
                )


        # ====================================================
        # HIDROLOGÍA
        # ====================================================

        hydrology = {}


        if (
            analizar_corrientes_san_nicolas
            is not None
        ):

            with st.spinner(
                "Analizando propagación histórica..."
            ):

                try:

                    hydrology = (
                        analizar_corrientes_san_nicolas(

                            df_hist,

                            upstream_history=
                                upstream_history,

                            exog_history=
                                exog_history,

                            usar_historial_completo=
                                True,
                        )
                    )


                except Exception as exc:

                    hydrology = {
                        "status":
                            "error",

                        "error":
                            str(
                                exc
                            ),
                    }


        # ====================================================
        # ENTRENAMIENTO
        # ====================================================

        with st.spinner(
            "Entrenando modelo hidrológico multivariable..."
        ):

            try:

                models, metrics = train(

                    df_hist,

                    exog_history=
                        exog_history,

                    upstream_history=
                        upstream_history,

                    hydrology=
                        hydrology,
                )


                model_error = None


            except Exception as exc:

                models = {}

                metrics = {}

                model_error = str(
                    exc
                )


        # ====================================================
        # PRONÓSTICO 60 DÍAS
        # ====================================================

        forecast = (
            pd.DataFrame()
        )


        if models:

            with st.spinner(
                "Generando pronóstico de 60 días..."
            ):

                try:

                    forecast = predict(

                        df_hist,

                        models,

                        days=
                            FORECAST_DAYS,

                        exog_future=
                            exog_future,

                        upstream_future=
                            None,

                        hydrology=
                            hydrology,
                    )


                except Exception as exc:

                    model_error = str(
                        exc
                    )


        # ====================================================
        # SESIÓN
        # ====================================================

        st.session_state[
            "datos"
        ] = df


        st.session_state[
            "df_hist"
        ] = df_hist


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
            "hydrology"
        ] = hydrology


        st.session_state[
            "models"
        ] = models


        st.session_state[
            "metrics"
        ] = metrics


        st.session_state[
            "forecast"
        ] = forecast


        st.session_state[
            "model_error"
        ] = model_error


        st.session_state[
            "fecha_base"
        ] = fecha_base


        st.session_state[
            "actualizado"
        ] = datetime.now()


        if (
            model_error is None
            and not forecast.empty
        ):

            st.success(
                "✅ Datos y modelo actualizados correctamente."
            )

        else:

            st.warning(
                "Los datos fueron actualizados, pero el "
                "pronóstico no pudo generarse. "
                f"{model_error or 'Sin detalle'}"
            )


# ============================================================
# SIN DATOS TODAVÍA
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "Seleccione la fecha base y presione "
        "**Actualizar datos y modelo**."
    )


# ============================================================
# PRESENTACIÓN
# ============================================================

else:

    df = st.session_state[
        "datos"
    ]


    df_hist = st.session_state.get(
        "df_hist",
        df,
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


    hydrology = st.session_state.get(
        "hydrology",
        {},
    )


    models = st.session_state.get(
        "models",
        {},
    )


    metrics = st.session_state.get(
        "metrics",
        {},
    )


    forecast = st.session_state.get(
        "forecast",
        pd.DataFrame(),
    )


    model_error = st.session_state.get(
        "model_error"
    )


    fecha_base_usada = (
        st.session_state.get(
            "fecha_base"
        )
    )


    actualizado = st.session_state.get(
        "actualizado"
    )


    # ========================================================
    # ESTADO ACTUAL
    # ========================================================

    st.subheader(
        "Estado actual"
    )


    trend = nivel_tendencia(
        df,
        7,
    )


    q_col = detectar_caudal_principal(
        exog_history
    )


    q_trend = tendencia_caudal(
        exog_history,
        q_col,
    )


    current_level = trend.get(
        "actual"
    )


    delta_7 = trend.get(
        "delta"
    )


    current_flow = q_trend.get(
        "actual"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(

        "San Nicolás",

        fmt_num(
            current_level,
            2,
            " m",
        ),

        (
            fmt_num(
                delta_7,
                2,
                " m / 7 d",
            )
            if delta_7
            is not None
            else None
        ),
    )


    c2.metric(

        "Tendencia 7 días",

        trend.get(
            "estado",
            "Sin dato",
        ),
    )


    c3.metric(

        "Caudal",

        fmt_num(
            current_flow,
            0,
            " m³/s",
        ),

        q_trend.get(
            "estado"
        ),
    )


    c4.metric(

        "Serie de caudal",

        flow_station_from_column(
            q_col
        )
        or "Sin dato",
    )


    if fecha_base_usada:

        st.caption(
            "Fecha base del cálculo: "
            f"{fecha_base_usada.strftime('%d/%m/%Y')}"
        )


    # ========================================================
    # PRONÓSTICO PRINCIPAL
    # ========================================================

    st.divider()


    st.subheader(
        "Pronóstico de nivel · 15 / 30 / 45 / 60 días"
    )


    if forecast.empty:

        st.warning(
            "No hay un pronóstico disponible."
        )


        if model_error:

            st.code(
                model_error
            )


    else:

        f = forecast.copy()


        f[
            "datetime"
        ] = datetime_naive(
            f[
                "datetime"
            ]
        )


        for col in [
            "prediction",
            "lower",
            "upper",
            "horizon_day",
        ]:

            if col in f.columns:

                f[
                    col
                ] = pd.to_numeric(
                    f[
                        col
                    ],
                    errors="coerce",
                )


        history_plot = (
            df.tail(
                dias_historia_visual
            )
            .copy()
        )


        fig = go.Figure()


        # ----------------------------------------------------
        # HISTORIA
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=history_plot[
                    "datetime"
                ],

                y=history_plot[
                    "nivel"
                ],

                mode="lines",

                name="Nivel observado",

                line=dict(
                    color="#2563eb",
                    width=2.5,
                ),
            )
        )


        # ----------------------------------------------------
        # CONEXIÓN ÚLTIMO OBSERVADO
        # ----------------------------------------------------

        if (
            not history_plot.empty
            and not f.empty
        ):

            fig.add_trace(

                go.Scatter(

                    x=[
                        history_plot[
                            "datetime"
                        ].iloc[-1],

                        f[
                            "datetime"
                        ].iloc[0],
                    ],

                    y=[
                        history_plot[
                            "nivel"
                        ].iloc[-1],

                        f[
                            "prediction"
                        ].iloc[0],
                    ],

                    mode="lines",

                    showlegend=False,

                    line=dict(
                        color="#f59e0b",
                        width=2.5,
                    ),
                )
            )


        # ----------------------------------------------------
        # 1–15
        # ----------------------------------------------------

        f15 = f[
            f[
                "horizon_day"
            ]
            <= 15
        ]


        if not f15.empty:

            fig.add_trace(

                go.Scatter(

                    x=f15[
                        "datetime"
                    ],

                    y=f15[
                        "prediction"
                    ],

                    mode="lines",

                    name="1–15 días",

                    line=dict(
                        color="#f59e0b",
                        width=3,
                    ),
                )
            )


        # ----------------------------------------------------
        # 16–30
        # ----------------------------------------------------

        f30 = f[
            (
                f[
                    "horizon_day"
                ]
                >= 16
            )
            &
            (
                f[
                    "horizon_day"
                ]
                <= 30
            )
        ]


        if not f30.empty:

            fig.add_trace(

                go.Scatter(

                    x=f30[
                        "datetime"
                    ],

                    y=f30[
                        "prediction"
                    ],

                    mode="lines",

                    name="16–30 días",

                    line=dict(
                        color="#16a34a",
                        width=3,
                    ),
                )
            )


        # ----------------------------------------------------
        # 31–45
        # ----------------------------------------------------

        f45 = f[
            (
                f[
                    "horizon_day"
                ]
                >= 31
            )
            &
            (
                f[
                    "horizon_day"
                ]
                <= 45
            )
        ]


        if not f45.empty:

            fig.add_trace(

                go.Scatter(

                    x=f45[
                        "datetime"
                    ],

                    y=f45[
                        "prediction"
                    ],

                    mode="lines",

                    name="31–45 días",

                    line=dict(
                        color="#9333ea",
                        width=3,
                    ),
                )
            )


        # ----------------------------------------------------
        # 46–60
        # ----------------------------------------------------

        f60 = f[
            f[
                "horizon_day"
            ]
            >= 46
        ]


        if not f60.empty:

            fig.add_trace(

                go.Scatter(

                    x=f60[
                        "datetime"
                    ],

                    y=f60[
                        "prediction"
                    ],

                    mode="lines",

                    name="46–60 días",

                    line=dict(
                        color="#dc2626",
                        width=3,
                    ),
                )
            )


        # ----------------------------------------------------
        # INCERTIDUMBRE
        # ----------------------------------------------------

        if (
            "lower"
            in f.columns
            and "upper"
            in f.columns
        ):

            uncertainty = f[
                [
                    "datetime",
                    "lower",
                    "upper",
                ]
            ].dropna()


            if not uncertainty.empty:

                fig.add_trace(

                    go.Scatter(

                        x=uncertainty[
                            "datetime"
                        ],

                        y=uncertainty[
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

                        x=uncertainty[
                            "datetime"
                        ],

                        y=uncertainty[
                            "lower"
                        ],

                        mode="lines",

                        line=dict(
                            width=0,
                        ),

                        fill="tonexty",

                        fillcolor="rgba(100,116,139,0.12)",

                        name="Incertidumbre",
                    )
                )


        y_range = auto_y_range(
            [
                history_plot[
                    "nivel"
                ],
                f[
                    "prediction"
                ],
                f.get(
                    "lower",
                    pd.Series(
                        dtype=float
                    ),
                ),
                f.get(
                    "upper",
                    pd.Series(
                        dtype=float
                    ),
                ),
            ]
        )


        fig.update_layout(

            height=520,

            hovermode="x unified",

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),

            yaxis_title=
                "Nivel hidrométrico San Nicolás (m)",

            xaxis_title="Fecha",
        )


        fig.update_xaxes(
            tickformat="%d/%m/%Y",
        )


        if y_range:

            fig.update_yaxes(
                range=
                    y_range
            )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # ====================================================
        # VALORES DE LOS 4 HORIZONTES
        # ====================================================

        horizon_cols = st.columns(
            4
        )


        for idx, horizon in enumerate(
            [
                15,
                30,
                45,
                60,
            ]
        ):

            row = f[
                f[
                    "horizon_day"
                ]
                == horizon
            ]


            if not row.empty:

                prediction_value = (
                    safe_float(
                        row.iloc[
                            0
                        ][
                            "prediction"
                        ]
                    )
                )


                delta_total = (
                    prediction_value
                    - current_level
                    if (
                        prediction_value
                        is not None
                        and current_level
                        is not None
                    )
                    else None
                )


                horizon_cols[
                    idx
                ].metric(

                    f"Día {horizon}",

                    fmt_num(
                        prediction_value,
                        2,
                        " m",
                    ),

                    (
                        fmt_num(
                            delta_total,
                            2,
                            " m vs hoy",
                        )
                        if delta_total
                        is not None
                        else None
                    ),
                )


        st.caption(
            "1–15 días corresponde al horizonte de mayor "
            "utilidad operativa. A partir de 30 días aumenta "
            "la dependencia de tendencias y escenarios históricos."
        )


    # ========================================================
    # CORREDOR HIDROLÓGICO
    # ========================================================

    st.divider()


    st.subheader(
        "Corredor hidrológico aguas arriba"
    )


    corridor_table = tabla_corredor_actual(

        df,

        upstream_history,

        exog_history,
    )


    if corridor_table.empty:

        st.info(
            "No hay datos suficientes para construir "
            "el estado del corredor."
        )


    else:

        display_corridor = (
            corridor_table.copy()
        )


        for col in [

            "Nivel actual (m)",
            "Δ última medición (m)",
            "Caudal (m³/s)",
            "Lluvia 7 días (mm)",

        ]:

            if col in display_corridor.columns:

                display_corridor[
                    col
                ] = pd.to_numeric(
                    display_corridor[
                        col
                    ],
                    errors="coerce",
                ).round(
                    2
                )


        st.dataframe(

            display_corridor,

            use_container_width=True,

            hide_index=True,
        )


        st.caption(
            "Los niveles corresponden a escalas hidrométricas "
            "locales diferentes. No deben interpretarse como "
            "una única superficie de agua continua."
        )


    # ========================================================
    # PROPAGACIÓN
    # ========================================================

    st.divider()


    st.subheader(
        "Propagación Corrientes → San Nicolás"
    )


    if (
        not hydrology
        or hydrology.get(
            "status"
        )
        != "ok"
    ):

        st.info(
            "No existe un análisis de propagación "
            "suficiente para esta actualización."
        )


        if hydrology.get(
            "error"
        ):

            st.caption(
                hydrology.get(
                    "error"
                )
            )


    else:

        estimate = hydrology.get(
            "current_estimate",
            {},
        )


        stats = hydrology.get(
            "statistics",
            {},
        )


        lag_info = hydrology.get(
            "lag",
            {},
        )


        p1, p2, p3, p4 = st.columns(
            4
        )


        p1.metric(

            "Demora probable",

            (
                fmt_num(
                    estimate.get(
                        "delay_days"
                    ),
                    0,
                    " días",
                )
            ),
        )


        delay_min = safe_float(
            estimate.get(
                "delay_min_days"
            )
        )


        delay_max = safe_float(
            estimate.get(
                "delay_max_days"
            )
        )


        if (
            delay_min is not None
            and delay_max is not None
        ):

            delay_range = (
                f"{delay_min:.0f}–"
                f"{delay_max:.0f} días"
            )

        else:

            delay_range = (
                "Sin dato"
            )


        p2.metric(
            "Rango histórico similar",
            delay_range,
        )


        p3.metric(

            "Correlación de variaciones",

            fmt_num(
                lag_info.get(
                    "correlation"
                ),
                2,
            ),
        )


        p4.metric(

            "Eventos históricos",

            str(
                stats.get(
                    "event_count",
                    0,
                )
            ),
        )


        impact = estimate.get(
            "impact_date"
        )


        if impact is not None:

            st.info(
                "📍 Fecha central estimada de propagación "
                f"del pulso actual: **{fmt_date(impact)}**"
            )


        corridor_lags = hydrology.get(
            "corridor_lags",
            pd.DataFrame(),
        )


        if (
            isinstance(
                corridor_lags,
                pd.DataFrame,
            )
            and not corridor_lags.empty
        ):

            with st.expander(
                "Ver retardos estimados por tramo"
            ):

                display_lags = (
                    corridor_lags.copy()
                )


                if "lag_days" in display_lags.columns:

                    display_lags[
                        "lag_days"
                    ] = pd.to_numeric(
                        display_lags[
                            "lag_days"
                        ],
                        errors="coerce",
                    ).round(
                        0
                    )


                if "correlation" in display_lags.columns:

                    display_lags[
                        "correlation"
                    ] = pd.to_numeric(
                        display_lags[
                            "correlation"
                        ],
                        errors="coerce",
                    ).round(
                        2
                    )


                st.dataframe(
                    display_lags,
                    use_container_width=True,
                    hide_index=True,
                )


        similar = hydrology.get(
            "similar_events",
            pd.DataFrame(),
        )


        if (
            isinstance(
                similar,
                pd.DataFrame,
            )
            and not similar.empty
        ):

            with st.expander(
                "Eventos históricos más similares al estado actual"
            ):

                selected_cols = [

                    c
                    for c in [

                        "fecha_max_corrientes",
                        "max_corrientes_m",
                        "crecida_corrientes_m",
                        "fecha_max_san_nicolas",
                        "max_san_nicolas_m",
                        "respuesta_san_nicolas_m",
                        "lag_real_dias",
                        "similarity_score",

                    ]
                    if c
                    in similar.columns
                ]


                st.dataframe(

                    similar[
                        selected_cols
                    ],

                    use_container_width=True,

                    hide_index=True,
                )


    # ========================================================
    # CAUDALES
    # ========================================================

    st.divider()


    st.subheader(
        "Caudales históricos y actuales"
    )


    available_flow_columns = [

        (
            station,
            FLOW_COLUMNS[
                station
            ],
        )

        for station in STATIONS

        if (
            FLOW_COLUMNS[
                station
            ]
            in exog_history.columns
            and pd.to_numeric(
                exog_history[
                    FLOW_COLUMNS[
                        station
                    ]
                ],
                errors="coerce",
            )
            .notna()
            .any()
        )
    ]


    if not available_flow_columns:

        st.info(
            "No se encontraron series de caudal "
            "multies­tación utilizables."
        )


    else:

        flow_station_names = [
            item[
                0
            ]
            for item in available_flow_columns
        ]


        selected_flow_station = st.selectbox(

            "Estación de caudal",

            flow_station_names,

            key=
                "flow_station_select",
        )


        selected_flow_col = FLOW_COLUMNS[
            selected_flow_station
        ]


        q_plot = (
            exog_history[
                [
                    "datetime",
                    selected_flow_col,
                ]
            ]
            .copy()
        )


        q_plot[
            "datetime"
        ] = datetime_naive(
            q_plot[
                "datetime"
            ]
        )


        q_plot[
            selected_flow_col
        ] = pd.to_numeric(
            q_plot[
                selected_flow_col
            ],
            errors="coerce",
        )


        q_plot = (
            q_plot
            .dropna()
            .tail(
                365 * 3
            )
        )


        q_fig = go.Figure()


        q_fig.add_trace(

            go.Scatter(

                x=q_plot[
                    "datetime"
                ],

                y=q_plot[
                    selected_flow_col
                ],

                mode="lines",

                name=
                    selected_flow_station,
            )
        )


        if (
            selected_flow_col
            in exog_future.columns
        ):

            q_future = (
                exog_future[
                    [
                        "datetime",
                        selected_flow_col,
                    ]
                ]
                .copy()
            )


            q_future[
                "datetime"
            ] = datetime_naive(
                q_future[
                    "datetime"
                ]
            )


            q_future[
                selected_flow_col
            ] = pd.to_numeric(
                q_future[
                    selected_flow_col
                ],
                errors="coerce",
            )


            q_future = q_future.dropna()


            if not q_future.empty:

                q_fig.add_trace(

                    go.Scatter(

                        x=q_future[
                            "datetime"
                        ],

                        y=q_future[
                            selected_flow_col
                        ],

                        mode="lines",

                        name=
                            "Proyección de caudal",

                        line=dict(
                            dash="dash",
                        ),
                    )
                )


        q_fig.update_layout(

            height=390,

            hovermode="x unified",

            yaxis_title=
                "Caudal (m³/s)",

            xaxis_title=
                "Fecha",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )


        q_fig.update_xaxes(
            tickformat="%d/%m/%Y",
        )


        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # FUENTE EXACTA
        # ----------------------------------------------------

        flow_series_meta = (
            exog_meta.get(
                "flow_series",
                {}
            )
            if isinstance(
                exog_meta,
                dict,
            )
            else {}
        )


        station_meta = (
            flow_series_meta.get(
                selected_flow_station,
                {}
            )
            if isinstance(
                flow_series_meta,
                dict,
            )
            else {}
        )


        if station_meta:

            st.caption(
                "Serie INA utilizada · "
                f"Estación: {selected_flow_station} · "
                f"series_id: "
                f"{station_meta.get('series_id', 'Sin dato')} · "
                f"{station_meta.get('series_name', '')}"
            )


    # ========================================================
    # LLUVIAS
    # ========================================================

    st.divider()


    st.subheader(
        "Lluvia por punto del corredor"
    )


    selected_rain_station = st.selectbox(

        "Punto de precipitación",

        STATIONS,

        index=0,

        key=
            "rain_station_select",
    )


    rain_col = RAIN_COLUMNS[
        selected_rain_station
    ]


    rain_fig = go.Figure()


    if (
        rain_col
        in exog_history.columns
    ):

        rain_hist = (
            exog_history[
                [
                    "datetime",
                    rain_col,
                ]
            ]
            .copy()
        )


        rain_hist[
            "datetime"
        ] = datetime_naive(
            rain_hist[
                "datetime"
            ]
        )


        rain_hist[
            rain_col
        ] = pd.to_numeric(
            rain_hist[
                rain_col
            ],
            errors="coerce",
        ).fillna(
            0
        )


        rain_hist = rain_hist.tail(
            90
        )


        rain_fig.add_trace(

            go.Bar(

                x=rain_hist[
                    "datetime"
                ],

                y=rain_hist[
                    rain_col
                ],

                name="Observado",
            )
        )


    if (
        rain_col
        in exog_future.columns
    ):

        rain_future = (
            exog_future[
                [
                    "datetime",
                    rain_col,
                ]
            ]
            .copy()
        )


        rain_future[
            "datetime"
        ] = datetime_naive(
            rain_future[
                "datetime"
            ]
        )


        rain_future[
            rain_col
        ] = pd.to_numeric(
            rain_future[
                rain_col
            ],
            errors="coerce",
        ).fillna(
            0
        )


        # Mostrar únicamente período meteorológico real
        # de corto plazo.
        rain_future = rain_future.head(
            16
        )


        rain_fig.add_trace(

            go.Bar(

                x=rain_future[
                    "datetime"
                ],

                y=rain_future[
                    rain_col
                ],

                name="Pronóstico meteorológico",
            )
        )


    rain_fig.update_layout(

        height=360,

        hovermode="x unified",

        yaxis_title=
            "Precipitación (mm/día)",

        xaxis_title=
            "Fecha",

        barmode="group",

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )


    rain_fig.update_xaxes(
        tickformat="%d/%m",
    )


    st.plotly_chart(
        rain_fig,
        use_container_width=True,
    )


    st.caption(
        "La precipitación meteorológica futura real se utiliza "
        "hasta el horizonte disponible. Los horizontes más largos "
        "no se presentan como pronóstico meteorológico determinista."
    )


    # ========================================================
    # COMPARACIÓN AÑO CONTRA AÑO
    # ========================================================

    st.divider()


    st.subheader(
        "Comparación histórica año contra año"
    )


    comparison_station = st.selectbox(

        "Estación a comparar",

        STATIONS,

        index=
            len(
                STATIONS
            )
            - 1,

        key=
            "comparison_station",
    )


    reference_col1, reference_col2 = st.columns(
        2
    )


    reference_start = reference_col1.date_input(

        "Inicio del período",

        value=
            date(
                2000,
                1,
                1,
            ),

        format=
            "DD/MM/YYYY",

        key=
            "comparison_start",
    )


    reference_end = reference_col2.date_input(

        "Fin del período",

        value=
            date(
                2000,
                3,
                31,
            ),

        format=
            "DD/MM/YYYY",

        key=
            "comparison_end",
    )


    if comparison_station == "San Nicolás":

        comparison_source = (
            df_hist.copy()
        )

        comparison_value_col = (
            "nivel"
        )

    else:

        comparison_source = (
            preparar_upstream_visual(
                upstream_history
            )
        )

        comparison_value_col = (
            LEVEL_COLUMNS[
                comparison_station
            ]
        )


    available_years = []


    if (
        comparison_source
        is not None
        and not comparison_source.empty
        and "datetime"
        in comparison_source.columns
        and comparison_value_col
        in comparison_source.columns
    ):

        dates_year = datetime_naive(
            comparison_source[
                "datetime"
            ]
        )


        available_years = sorted(
            dates_year
            .dropna()
            .dt
            .year
            .unique()
            .tolist()
        )


    default_years = (
        available_years[
            -5:
        ]
        if available_years
        else []
    )


    selected_years = st.multiselect(

        "Años a comparar",

        available_years,

        default=
            default_years,
    )


    if selected_years:

        comparison = construir_comparacion_anual(

            comparison_source,

            comparison_value_col,

            reference_start.month,

            reference_start.day,

            reference_end.month,

            reference_end.day,

            selected_years,
        )


        if comparison.empty:

            st.info(
                "No existen datos para el período "
                "y los años seleccionados."
            )


        else:

            year_fig = go.Figure()


            for year_label, group in (
                comparison.groupby(
                    "year_label"
                )
            ):

                year_fig.add_trace(

                    go.Scatter(

                        x=group[
                            "day_index"
                        ],

                        y=group[
                            comparison_value_col
                        ],

                        mode="lines",

                        name=
                            str(
                                year_label
                            ),
                    )
                )


            y_range_year = auto_y_range(
                [
                    comparison[
                        comparison_value_col
                    ]
                ]
            )


            year_fig.update_layout(

                height=460,

                hovermode="x unified",

                xaxis_title=
                    "Días desde el inicio del período",

                yaxis_title=
                    (
                        "Nivel hidrométrico "
                        f"{comparison_station} (m)"
                    ),

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
                ),

                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    x=0,
                ),
            )


            if y_range_year:

                year_fig.update_yaxes(
                    range=
                        y_range_year
                )


            st.plotly_chart(
                year_fig,
                use_container_width=True,
            )


            st.caption(
                "Cada curva comienza en el mismo día/mes y se "
                "superpone mediante días transcurridos, permitiendo "
                "comparar la evolución estacional entre años."
            )


    else:

        st.info(
            "Seleccione al menos un año."
        )


    # ========================================================
    # IMPORTANCIA DE VARIABLES
    # ========================================================

    st.divider()


    importance = (
        models.get(
            "importance",
            pd.DataFrame(),
        )
        if isinstance(
            models,
            dict,
        )
        else pd.DataFrame()
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
                    25
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

                xaxis_title=
                    "Importancia relativa",

                yaxis_title=
                    "Variable",

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20,
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
    # VARIABLES UTILIZADAS
    # ========================================================

    with st.expander(
        "🔬 Cobertura de variables del modelo"
    ):

        coverage = pd.DataFrame(
            [
                {
                    "Grupo":
                        "Nivel San Nicolás",

                    "Utilizado":
                        True,

                    "Variables":
                        1,
                },

                {
                    "Grupo":
                        "Niveles aguas arriba",

                    "Utilizado":
                        models.get(
                            "uses_upstream",
                            False,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else False,

                    "Variables":
                        models.get(
                            "upstream_feature_count",
                            0,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else 0,
                },

                {
                    "Grupo":
                        "Caudales",

                    "Utilizado":
                        models.get(
                            "uses_caudal",
                            False,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else False,

                    "Variables":
                        models.get(
                            "flow_feature_count",
                            0,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else 0,
                },

                {
                    "Grupo":
                        "Lluvias",

                    "Utilizado":
                        models.get(
                            "uses_rain",
                            False,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else False,

                    "Variables":
                        models.get(
                            "rain_feature_count",
                            0,
                        )
                        if isinstance(
                            models,
                            dict,
                        )
                        else 0,
                },
            ]
        )


        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True,
        )


        if isinstance(
            exog_meta,
            dict,
        ):

            flow_stations = (
                exog_meta.get(
                    "flow_stations",
                    []
                )
            )


            if flow_stations:

                st.write(
                    "**Estaciones con caudal INA validado:** "
                    + ", ".join(
                        flow_stations
                    )
                )


    # ========================================================
    # DIAGNÓSTICO MODELO
    # ========================================================

    with st.expander(
        "📊 Diagnóstico del modelo"
    ):

        d1, d2, d3, d4 = st.columns(
            4
        )


        d1.metric(

            "RMSE",

            fmt_num(
                metrics.get(
                    "RMSE"
                ),
                3,
                " m",
            ),
        )


        d2.metric(

            "MAE",

            fmt_num(
                metrics.get(
                    "MAE"
                ),
                3,
                " m",
            ),
        )


        d3.metric(

            "Filas entrenamiento",

            str(
                metrics.get(
                    "training_rows",
                    0,
                )
            ),
        )


        d4.metric(

            "Variables",

            str(
                metrics.get(
                    "features",
                    0,
                )
            ),
        )


        if model_error:

            st.error(
                model_error
            )


        if isinstance(
            exog_meta,
            dict,
        ):

            st.write(
                "**Fuente de caudal:**",
                exog_meta.get(
                    "caudal_source",
                    "Sin dato",
                ),
            )


            st.write(
                "**Fuente de lluvia:**",
                exog_meta.get(
                    "rain_source",
                    "Sin dato",
                ),
            )


            st.write(
                "**Caudales disponibles:**",
                exog_meta.get(
                    "flow_stations",
                    [],
                ),
            )


        if isinstance(
            upstream_meta,
            dict,
        ):

            st.write(
                "**Diagnóstico aguas arriba:**"
            )


            st.json(
                upstream_meta
            )


    # ========================================================
    # METODOLOGÍA
    # ========================================================

    with st.expander(
        "ℹ️ Metodología y alcance"
    ):

        st.markdown(
            """
            ### Horizonte 1–15 días

            Es el horizonte de mayor utilidad operativa. Combina el
            modelo estadístico, nivel actual de San Nicolás,
            niveles aguas arriba, caudales disponibles, tendencia
            de los caudales y precipitación meteorológica prevista.

            ### Horizonte 16–30 días

            Aumenta el peso de las señales de propagación
            hidrológica y de las tendencias del corredor.

            ### Horizonte 31–45 días

            Se interpreta como **escenario probabilístico**.
            La incertidumbre meteorológica es considerablemente
            mayor.

            ### Horizonte 46–60 días

            Se presenta como **tendencia hidrológica extendida**,
            no como un pronóstico meteorológico determinista.

            ### Propagación

            El tiempo Corrientes → San Nicolás se estima mediante
            variaciones históricas y eventos de creciente.
            Los niveles absolutos de dos estaciones no deben
            compararse como si utilizaran el mismo cero
            hidrométrico.

            ### Caudal y lluvia

            Cuando INA dispone de series utilizables, cada estación
            conserva su propio caudal. La lluvia también se mantiene
            separada por punto del corredor para que el modelo pueda
            distinguir dónde ocurre cada precipitación.
            """
        )


        st.warning(
            "La plataforma es experimental e informativa. "
            "No sustituye avisos, alertas ni pronósticos "
            "emitidos por organismos oficiales."
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
# PIE
# ============================================================

st.divider()


st.markdown(
    """
    **Fuentes**

    Nivel hidrométrico y caudal: **Instituto Nacional del Agua (INA)**  
    Precipitación: **Open-Meteo**  
    Pronóstico y análisis de propagación: **modelo experimental propio**
    """
)


st.caption(
    f"Paraná · San Nicolás {APP_VERSION} · "
    "Pronóstico multivariable 15 / 30 / 45 / 60 días"
)
