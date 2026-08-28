# ============================================================
# PARANÁ · SAN NICOLÁS
# app.py
# V11.9.4
#
# Compatible con:
#   src/ina.py estable A5 serie 36
#   src/upstream.py V11.9.1
#   src/exogenous.py V11.9.3
#
# MEJORAS:
# - visualización responsive
# - upstream compacto
# - Corrientes vs San Nicolás legible
# - selector histórico
# - caudal INA visible + diagnóstico
# - 15 / 30 / 60 días
# - fechas sin timezone en merges
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import date, timedelta, datetime


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

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

try:
    from src.upstream import diagnostic_table as upstream_diagnostic_table
except Exception:
    upstream_diagnostic_table = None


try:
    from src.exogenous import diagnostic as exogenous_diagnostic
except Exception:
    exogenous_diagnostic = None


try:
    from src.stress_ui import get_stress_scenario
except Exception:
    get_stress_scenario = None


try:
    from src.hydrology import analizar_corrientes_san_nicolas
except Exception:
    analizar_corrientes_san_nicolas = None


# ============================================================
# VERSIÓN
# ============================================================

APP_VERSION = "V11.9.4"

FORECAST_DAYS = 15
TREND_DAYS = 30
STRESS_DAYS = 60

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
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS RESPONSIVE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(127,127,127,0.045);
        border: 1px solid rgba(127,127,127,0.16);
        padding: 10px 12px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.42rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    h1 {
        font-size: 2.05rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
    }

    h3 {
        font-size: 1.18rem !important;
    }

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            padding-top: 0.7rem;
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

    x["datetime"] = datetime_naive(
        x["datetime"]
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
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return x


def normalizar_frame_temporal(df):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or "datetime" not in df.columns
    ):
        return df

    x = df.copy()

    x["datetime"] = datetime_naive(
        x["datetime"]
    )

    return x


def valor_seguro(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


# ============================================================
# TENDENCIA DE NIVEL
# ============================================================

def tendencia_nivel(df):

    if (
        df is None
        or df.empty
        or "nivel" not in df.columns
    ):
        return "Sin datos", np.nan

    values = (
        pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )
        .dropna()
    )

    if len(values) < 2:

        return "Sin datos", np.nan

    delta = float(
        values.iloc[-1]
        - values.iloc[-2]
    )

    if delta > 0.02:
        estado = "↑ Creciendo"

    elif delta < -0.02:
        estado = "↓ Bajando"

    else:
        estado = "→ Estable"

    return estado, delta


# ============================================================
# TENDENCIA DE CAUDAL
# ============================================================

def calcular_tendencia_caudal(df):

    result = {
        "actual": None,
        "delta_3": None,
        "delta_7": None,
        "pct_7": None,
        "estado": "Sin datos",
    }

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or "caudal_m3s" not in df.columns
    ):
        return result

    x = df.copy()

    x["caudal_m3s"] = pd.to_numeric(
        x["caudal_m3s"],
        errors="coerce",
    )

    x = x.dropna(
        subset=["caudal_m3s"]
    )

    if x.empty:
        return result

    q = x["caudal_m3s"].to_numpy(
        dtype=float
    )

    actual = float(q[-1])

    result["actual"] = actual

    if len(q) >= 4:

        result["delta_3"] = (
            actual
            - float(q[-4])
        )

    if len(q) >= 8:

        q7 = float(q[-8])

        delta7 = actual - q7

        result["delta_7"] = delta7

        if q7 != 0:

            result["pct_7"] = (
                delta7
                / q7
                * 100.0
            )

    recent = q[
        -min(
            7,
            len(q),
        ):
    ]

    if len(recent) >= 3:

        slope = float(
            np.polyfit(
                np.arange(len(recent)),
                recent,
                1,
            )[0]
        )

        threshold = max(
            actual * 0.002,
            20.0,
        )

        if slope > threshold:
            result["estado"] = "↑ Creciente"

        elif slope < -threshold:
            result["estado"] = "↓ Bajante"

        else:
            result["estado"] = "→ Estable"

    return result


# ============================================================
# RESUMEN UPSTREAM
# ============================================================

UPSTREAM_LABELS = {
    "nivel_corrientes": "Corrientes",
    "nivel_goya": "Goya",
    "nivel_la_paz": "La Paz",
    "nivel_parana": "Paraná",
    "nivel_diamante": "Diamante",
    "nivel_rosario": "Rosario",
    "nivel_villa_constitucion": "Villa Constitución",
}


def construir_tabla_upstream(
    upstream_history,
):

    rows = []

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
    ):

        return pd.DataFrame()

    x = normalizar_frame_temporal(
        upstream_history
    )

    for column, station in UPSTREAM_LABELS.items():

        if column not in x.columns:
            continue

        serie = pd.to_numeric(
            x[column],
            errors="coerce",
        )

        temp = pd.DataFrame(
            {
                "datetime":
                    x["datetime"],

                "value":
                    serie,
            }
        ).dropna()

        if temp.empty:

            rows.append(
                {
                    "Estación": station,
                    "Nivel": np.nan,
                    "Anterior": np.nan,
                    "Variación": np.nan,
                    "Tendencia": "Sin datos",
                    "Fecha": None,
                }
            )

            continue

        temp = (
            temp
            .sort_values("datetime")
            .drop_duplicates(
                "datetime",
                keep="last",
            )
        )

        actual = float(
            temp["value"].iloc[-1]
        )

        fecha = temp[
            "datetime"
        ].iloc[-1]

        if len(temp) >= 2:

            anterior = float(
                temp["value"].iloc[-2]
            )

            delta = (
                actual
                - anterior
            )

        else:

            anterior = np.nan
            delta = np.nan

        if pd.notna(delta):

            if delta > 0.015:
                trend = "↑ Creciendo"

            elif delta < -0.015:
                trend = "↓ Bajando"

            else:
                trend = "→ Estable"

        else:

            trend = "Sin datos"

        rows.append(
            {
                "Estación": station,
                "Nivel": actual,
                "Anterior": anterior,
                "Variación": delta,
                "Tendencia": trend,
                "Fecha": fecha,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EXTENSIÓN 16 -> 30
# ============================================================

def extender_pronostico_30(
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

    f["datetime"] = datetime_naive(
        f["datetime"]
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
        result[
            "datetime"
        ].iloc[-1]
    )

    last_level = float(
        result[
            "prediction"
        ].iloc[-1]
    )

    recent = (
        result[
            "prediction"
        ]
        .dropna()
        .tail(5)
    )

    if len(recent) >= 3:

        slope = float(
            np.polyfit(
                np.arange(
                    len(recent)
                ),
                recent.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    else:

        local = (
            pd.to_numeric(
                df["nivel"],
                errors="coerce",
            )
            .dropna()
            .tail(7)
        )

        if len(local) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(local)
                    ),
                    local.to_numpy(
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
            -0.08,
            0.08,
        )
    )

    extra = []

    for day in range(
        16,
        TREND_DAYS + 1,
    ):

        damping = np.exp(
            -0.12
            * (
                day
                - 15
            )
        )

        daily_change = float(
            np.clip(
                slope
                * damping,
                -0.04,
                0.06,
            )
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
                pd.DataFrame(extra),
            ],
            ignore_index=True,
        )

    return result


# ============================================================
# ANCLA DÍA 30
# ============================================================

def obtener_ancla_dia_30(
    forecast30,
):

    if (
        forecast30 is None
        or forecast30.empty
    ):

        return None, None

    x = forecast30.copy()

    x["datetime"] = datetime_naive(
        x["datetime"]
    )

    x["prediction"] = pd.to_numeric(
        x["prediction"],
        errors="coerce",
    )

    x = x.dropna(
        subset=[
            "datetime",
            "prediction",
        ]
    )

    if x.empty:

        return None, None

    row = x.iloc[-1]

    return (
        row["datetime"],
        float(
            row["prediction"]
        ),
    )


# ============================================================
# ESTRÉS COMPATIBLE
# ============================================================

def adaptar_stress_antiguo(
    stress,
    anchor_date,
    anchor_level,
):

    if (
        stress is None
        or not isinstance(
            stress,
            pd.DataFrame,
        )
        or stress.empty
        or "stress_level"
        not in stress.columns
    ):

        return pd.DataFrame()

    temp = stress.copy()

    if (
        "scenario_day"
        in temp.columns
    ):

        selected = temp[
            pd.to_numeric(
                temp[
                    "scenario_day"
                ],
                errors="coerce",
            )
            > 30
        ].copy()

        if selected.empty:

            selected = (
                temp
                .iloc[
                    30:60
                ]
                .copy()
            )

    else:

        selected = (
            temp
            .iloc[
                30:60
            ]
            .copy()
        )

    if selected.empty:

        return pd.DataFrame()

    selected = (
        selected
        .reset_index(
            drop=True
        )
    )

    selected[
        "stress_level"
    ] = pd.to_numeric(
        selected[
            "stress_level"
        ],
        errors="coerce",
    )

    first_level = (
        selected[
            "stress_level"
        ].dropna()
    )

    if first_level.empty:

        return pd.DataFrame()

    offset = (
        float(
            anchor_level
        )
        - float(
            first_level.iloc[0]
        )
    )

    selected[
        "stress_level"
    ] = np.clip(
        selected[
            "stress_level"
        ]
        + offset,
        Y_MIN,
        Y_MAX,
    )

    selected[
        "datetime"
    ] = pd.date_range(
        start=(
            pd.Timestamp(
                anchor_date
            )
            + pd.Timedelta(
                days=1
            )
        ),
        periods=len(
            selected
        ),
        freq="D",
    )

    selected[
        "scenario_day"
    ] = np.arange(
        31,
        31 + len(
            selected
        ),
    )

    previous = (
        pd.Series(
            [float(anchor_level)]
            + selected[
                "stress_level"
            ]
            .iloc[:-1]
            .tolist()
        )
    )

    selected[
        "daily_change"
    ] = (
        selected[
            "stress_level"
        ].reset_index(
            drop=True
        )
        - previous
    )

    selected[
        "anchor_level"
    ] = float(
        anchor_level
    )

    selected[
        "anchor_day"
    ] = 30

    return selected


def construir_stress_compatible(
    df,
    models,
    exog_history,
    upstream_history,
    anchor_date,
    anchor_level,
):

    if get_stress_scenario is None:

        return (
            pd.DataFrame(),
            "no_disponible",
        )

    try:

        scenario = get_stress_scenario(
            df=df,
            models=models,
            exog_history=exog_history,
            upstream_history=upstream_history,
            days=STRESS_DAYS,
            anchor_date=anchor_date,
            anchor_level=anchor_level,
            anchor_day=30,
        )

        return (
            scenario,
            "stress_nuevo",
        )

    except TypeError as exc:

        text = str(exc)

        if (
            "anchor_date" not in text
            and "anchor_level" not in text
            and "anchor_day" not in text
        ):

            raise

        old = get_stress_scenario(
            df=df,
            models=models,
            exog_history=exog_history,
            upstream_history=upstream_history,
            days=STRESS_DAYS,
        )

        adapted = adaptar_stress_antiguo(
            old,
            anchor_date,
            anchor_level,
        )

        return (
            adapted,
            "stress_compatibilidad",
        )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · Monitoreo y pronóstico experimental"
)

st.markdown(
    """
    La plataforma analiza la evolución del Paraná en **San Nicolás**
    considerando el nivel local, estaciones aguas arriba,
    **caudal**, precipitación y relaciones históricas de propagación.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Consulta"
)

today = date.today()

default_start = (
    today
    - timedelta(
        days=365
    )
)

desde = st.sidebar.date_input(
    "Desde",
    value=default_start,
    format="DD/MM/YYYY",
)

hasta = st.sidebar.date_input(
    "Hasta",
    value=today,
    format="DD/MM/YYYY",
)

actualizar = st.sidebar.button(
    "🔄 Actualizar modelo",
    use_container_width=True,
    type="primary",
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

st.sidebar.caption(
    "Nivel y caudal: INA"
)

st.sidebar.caption(
    "Precipitación: Open-Meteo"
)


# ============================================================
# ACTUALIZAR
# ============================================================

if actualizar:

    if desde > hasta:

        st.error(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    else:

        inicio = desde.strftime(
            "%Y-%m-%d"
        )

        fin = hasta.strftime(
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # SAN NICOLÁS
        # ----------------------------------------------------

        with st.spinner(
            "Consultando San Nicolás..."
        ):

            df_raw, error_ina = observed(
                inicio,
                fin,
            )

        if error_ina:

            st.error(
                error_ina
            )

        else:

            df = preparar_datos(
                df_raw
            )

            if df.empty:

                st.error(
                    "No existen observaciones válidas "
                    "para el período seleccionado."
                )

            else:

                # --------------------------------------------
                # EXÓGENAS
                # --------------------------------------------

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

                        exog_history = (
                            normalizar_frame_temporal(
                                exog_history
                            )
                        )

                        exog_future = (
                            normalizar_frame_temporal(
                                exog_future
                            )
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
                                str(exc)
                        }

                        st.warning(
                            "No fue posible obtener completamente "
                            f"lluvia/caudal: {exc}"
                        )

                # --------------------------------------------
                # UPSTREAM
                # --------------------------------------------

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

                        upstream_history = (
                            normalizar_frame_temporal(
                                upstream_history
                            )
                        )

                    except Exception as exc:

                        upstream_history = (
                            pd.DataFrame()
                        )

                        upstream_meta = {}

                        st.warning(
                            "No fue posible obtener todas las "
                            f"estaciones aguas arriba: {exc}"
                        )

                # --------------------------------------------
                # HIDROLOGÍA
                # --------------------------------------------

                hydrology = {}

                if (
                    analizar_corrientes_san_nicolas
                    is not None
                ):

                    try:

                        hydrology = (
                            analizar_corrientes_san_nicolas(
                                san_nicolas=df,
                                upstream_history=
                                    upstream_history,
                                exog_history=
                                    exog_history,
                                max_lag=20,
                            )
                        )

                    except TypeError:

                        try:

                            hydrology = (
                                analizar_corrientes_san_nicolas(
                                    san_nicolas=df,
                                    upstream_history=
                                        upstream_history,
                                    exog_history=
                                        exog_history,
                                )
                            )

                        except Exception:

                            hydrology = {}

                    except Exception:

                        hydrology = {}

                # --------------------------------------------
                # MODELO
                # --------------------------------------------

                with st.spinner(
                    "Entrenando modelo y generando pronóstico..."
                ):

                    try:

                        train_result = train(
                            df,
                            exog_history=
                                exog_history,
                            upstream_history=
                                upstream_history,
                        )

                        if isinstance(
                            train_result,
                            tuple,
                        ):

                            models = (
                                train_result[0]
                            )

                            metrics = (
                                train_result[1]
                                if len(
                                    train_result
                                ) > 1
                                else {}
                            )

                        else:

                            models = (
                                train_result
                                if isinstance(
                                    train_result,
                                    dict,
                                )
                                else {}
                            )

                            metrics = {}

                        try:

                            forecast15 = predict(
                                models,
                                df,
                                days=
                                    FORECAST_DAYS,
                                exog_future=
                                    exog_future,
                            )

                        except TypeError:

                            try:

                                forecast15 = predict(
                                    models,
                                    df,
                                    FORECAST_DAYS,
                                )

                            except TypeError:

                                forecast15 = predict(
                                    models,
                                    df,
                                )

                        forecast15 = (
                            normalizar_frame_temporal(
                                forecast15
                            )
                        )

                        forecast30 = (
                            extender_pronostico_30(
                                forecast15,
                                df,
                            )
                        )

                        (
                            anchor_date,
                            anchor_level,
                        ) = obtener_ancla_dia_30(
                            forecast30
                        )

                        stress60 = (
                            pd.DataFrame()
                        )

                        stress_source = (
                            "no_disponible"
                        )

                        if (
                            anchor_date
                            is not None
                            and anchor_level
                            is not None
                        ):

                            try:

                                (
                                    stress60,
                                    stress_source,
                                ) = (
                                    construir_stress_compatible(
                                        df=df,
                                        models=models,
                                        exog_history=
                                            exog_history,
                                        upstream_history=
                                            upstream_history,
                                        anchor_date=
                                            anchor_date,
                                        anchor_level=
                                            anchor_level,
                                    )
                                )

                            except Exception as exc:

                                st.warning(
                                    "No fue posible construir "
                                    "el escenario 31–60 días. "
                                    f"{exc}"
                                )

                    except Exception as exc:

                        models = {}
                        metrics = {}
                        forecast15 = pd.DataFrame()
                        forecast30 = pd.DataFrame()
                        stress60 = pd.DataFrame()
                        anchor_date = None
                        anchor_level = None
                        stress_source = "error"

                        st.error(
                            "No fue posible generar el pronóstico "
                            f"de 15 días. {exc}"
                        )

                # --------------------------------------------
                # SESIÓN
                # --------------------------------------------

                st.session_state[
                    "datos"
                ] = df

                st.session_state[
                    "forecast15"
                ] = forecast15

                st.session_state[
                    "forecast30"
                ] = forecast30

                st.session_state[
                    "stress60"
                ] = stress60

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
                    "hydrology"
                ] = hydrology

                st.session_state[
                    "stress_source"
                ] = stress_source

                st.session_state[
                    "anchor_date"
                ] = anchor_date

                st.session_state[
                    "anchor_level"
                ] = anchor_level

                st.session_state[
                    "fecha_inicio"
                ] = inicio

                st.session_state[
                    "fecha_fin"
                ] = fin

                st.session_state[
                    "actualizado"
                ] = datetime.now()

                st.success(
                    "✅ Datos y modelo actualizados correctamente."
                )


# ============================================================
# SIN DATOS
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

    forecast15 = (
        st.session_state.get(
            "forecast15",
            pd.DataFrame(),
        )
    )

    forecast30 = (
        st.session_state.get(
            "forecast30",
            pd.DataFrame(),
        )
    )

    stress60 = (
        st.session_state.get(
            "stress60",
            pd.DataFrame(),
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

    stress_source = (
        st.session_state.get(
            "stress_source",
            "—",
        )
    )


    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    nivel_actual = float(
        df[
            "nivel"
        ].iloc[-1]
    )

    fecha_actual = (
        df[
            "datetime"
        ].iloc[-1]
    )

    estado_nivel, delta_nivel = (
        tendencia_nivel(
            df
        )
    )


    # ========================================================
    # CAUDAL ACTUAL
    # ========================================================

    tq = calcular_tendencia_caudal(
        exog_history
    )

    caudal_actual = tq[
        "actual"
    ]


    # ========================================================
    # MÉTRICAS PRINCIPALES
    # ========================================================

    m1, m2, m3, m4 = st.columns(
        4
    )

    m1.metric(
        "Nivel San Nicolás",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_nivel:+.2f} m"
            if pd.notna(
                delta_nivel
            )
            else None
        ),
    )

    m2.metric(
        "Tendencia",
        estado_nivel,
    )

    m3.metric(
        "Caudal",
        (
            f"{caudal_actual:,.0f} m³/s"
            if caudal_actual
            is not None
            else "Sin dato"
        ),
    )

    m4.metric(
        "Última medición",
        pd.Timestamp(
            fecha_actual
        ).strftime(
            "%d/%m/%Y"
        ),
    )


    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    st.subheader(
        "📈 Nivel observado y proyección"
    )

    fig = go.Figure()

    visible_start = (
        df[
            "datetime"
        ].max()
        - pd.Timedelta(
            days=120
        )
    )

    observed_plot = df[
        df[
            "datetime"
        ] >= visible_start
    ]

    fig.add_trace(
        go.Scatter(
            x=observed_plot[
                "datetime"
            ],
            y=observed_plot[
                "nivel"
            ],
            mode="lines",
            name="Observado",
            line=dict(
                color="#2563EB",
                width=2.5,
            ),
        )
    )

    if (
        forecast15 is not None
        and not forecast15.empty
    ):

        f15 = (
            forecast15
            .head(
                FORECAST_DAYS
            )
            .copy()
        )

        connection = pd.DataFrame(
            {
                "datetime": [
                    df[
                        "datetime"
                    ].iloc[-1]
                ],

                "prediction": [
                    nivel_actual
                ],
            }
        )

        plot15 = pd.concat(
            [
                connection,
                f15[
                    [
                        "datetime",
                        "prediction",
                    ]
                ],
            ],
            ignore_index=True,
        )

        fig.add_trace(
            go.Scatter(
                x=plot15[
                    "datetime"
                ],
                y=plot15[
                    "prediction"
                ],
                mode="lines+markers",
                name="Pronóstico 1–15 días",
                line=dict(
                    color="#F59E0B",
                    width=3,
                ),
                marker=dict(
                    size=5
                ),
            )
        )

        if (
            "lower"
            in f15.columns
            and "upper"
            in f15.columns
        ):

            lower = pd.to_numeric(
                f15[
                    "lower"
                ],
                errors="coerce",
            )

            upper = pd.to_numeric(
                f15[
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
                        x=pd.concat(
                            [
                                f15[
                                    "datetime"
                                ],
                                f15[
                                    "datetime"
                                ].iloc[
                                    ::-1
                                ],
                            ]
                        ),
                        y=pd.concat(
                            [
                                upper,
                                lower.iloc[
                                    ::-1
                                ],
                            ]
                        ),
                        fill="toself",
                        fillcolor=(
                            "rgba(245,158,11,0.10)"
                        ),
                        line=dict(
                            color=(
                                "rgba(255,255,255,0)"
                            )
                        ),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Incertidumbre",
                    )
                )

    if (
        forecast30 is not None
        and not forecast30.empty
        and len(
            forecast30
        ) > FORECAST_DAYS
    ):

        extension = (
            forecast30
            .iloc[
                FORECAST_DAYS - 1:
            ]
            .copy()
        )

        fig.add_trace(
            go.Scatter(
                x=extension[
                    "datetime"
                ],
                y=extension[
                    "prediction"
                ],
                mode="lines+markers",
                name="Tendencia 16–30 días",
                line=dict(
                    color="#16A34A",
                    width=3,
                ),
                marker=dict(
                    size=4,
                ),
            )
        )

    if (
        stress60 is not None
        and isinstance(
            stress60,
            pd.DataFrame,
        )
        and not stress60.empty
        and "stress_level"
        in stress60.columns
    ):

        stress_plot = (
            stress60.copy()
        )

        stress_plot[
            "datetime"
        ] = datetime_naive(
            stress_plot[
                "datetime"
            ]
        )

        if (
            st.session_state.get(
                "anchor_date"
            )
            is not None
            and st.session_state.get(
                "anchor_level"
            )
            is not None
        ):

            stress_connection = (
                pd.DataFrame(
                    {
                        "datetime": [
                            st.session_state[
                                "anchor_date"
                            ]
                        ],

                        "stress_level": [
                            st.session_state[
                                "anchor_level"
                            ]
                        ],
                    }
                )
            )

            stress_plot = pd.concat(
                [
                    stress_connection,
                    stress_plot[
                        [
                            "datetime",
                            "stress_level",
                        ]
                    ],
                ],
                ignore_index=True,
            )

        fig.add_trace(
            go.Scatter(
                x=stress_plot[
                    "datetime"
                ],
                y=stress_plot[
                    "stress_level"
                ],
                mode="lines",
                name="Escenario 31–60 días",
                line=dict(
                    color="#DC2626",
                    width=3,
                    dash="dash",
                ),
            )
        )

    fig.update_layout(
        height=510,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_yaxes(
        title_text="Nivel (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
    )

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m/%Y",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # HORIZONTES
    # ========================================================

    h1, h2, h3 = st.columns(
        3
    )

    if (
        forecast15 is not None
        and not forecast15.empty
    ):

        h1.metric(
            "Día 15",
            f"{float(forecast15['prediction'].iloc[-1]):.2f} m",
        )

    else:

        h1.metric(
            "Día 15",
            "Sin dato",
        )

    if (
        forecast30 is not None
        and not forecast30.empty
    ):

        h2.metric(
            "Día 30",
            f"{float(forecast30['prediction'].iloc[-1]):.2f} m",
        )

    else:

        h2.metric(
            "Día 30",
            "Sin dato",
        )

    if (
        stress60 is not None
        and not stress60.empty
        and "stress_level"
        in stress60.columns
    ):

        valid60 = pd.to_numeric(
            stress60[
                "stress_level"
            ],
            errors="coerce",
        ).dropna()

        if not valid60.empty:

            h3.metric(
                "Día 60",
                f"{float(valid60.iloc[-1]):.2f} m",
            )

        else:

            h3.metric(
                "Día 60",
                "Sin dato",
            )

    else:

        h3.metric(
            "Día 60",
            "Sin dato",
        )


    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    st.divider()

    st.subheader(
        "🌊 Niveles aguas arriba"
    )

    upstream_table = (
        construir_tabla_upstream(
            upstream_history
        )
    )

    if upstream_table.empty:

        st.warning(
            "No hay observaciones disponibles "
            "de estaciones aguas arriba."
        )

    else:

        records = upstream_table.to_dict(
            "records"
        )

        for start_index in range(
            0,
            len(records),
            4,
        ):

            group = records[
                start_index:
                start_index + 4
            ]

            cols = st.columns(
                len(group)
            )

            for ui, row in zip(
                cols,
                group,
            ):

                with ui:

                    nivel = row[
                        "Nivel"
                    ]

                    delta = row[
                        "Variación"
                    ]

                    st.metric(
                        row[
                            "Estación"
                        ],
                        (
                            f"{nivel:.2f} m"
                            if pd.notna(
                                nivel
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
                        row[
                            "Tendencia"
                        ]
                    )


    # ========================================================
    # PERFIL DEL CORREDOR
    # ========================================================

    valid_upstream = (
        upstream_table.dropna(
            subset=[
                "Nivel"
            ]
        )
        if not upstream_table.empty
        else pd.DataFrame()
    )

    if not valid_upstream.empty:

        st.subheader(
            "🧭 Perfil actual del corredor"
        )

        profile = valid_upstream[
            [
                "Estación",
                "Nivel",
            ]
        ].copy()

        profile = pd.concat(
            [
                profile,
                pd.DataFrame(
                    [
                        {
                            "Estación":
                                "San Nicolás",

                            "Nivel":
                                nivel_actual,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        corridor_fig = go.Figure()

        corridor_fig.add_trace(
            go.Scatter(
                x=profile[
                    "Estación"
                ],
                y=profile[
                    "Nivel"
                ],
                mode="lines+markers+text",
                text=[
                    f"{v:.2f} m"
                    for v in profile[
                        "Nivel"
                    ]
                ],
                textposition="top center",
                name="Nivel actual",
                line=dict(
                    width=2.5,
                ),
                marker=dict(
                    size=9,
                ),
            )
        )

        corridor_fig.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10,
            ),
            yaxis_title="Nivel (m)",
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
        "🔗 Corrientes → San Nicolás"
    )

    periodo = st.selectbox(
        "Período visible",
        [
            "1 año",
            "5 años",
            "10 años",
            "20 años",
            "Todo el historial",
        ],
        index=1,
        key="corrientes_periodo",
    )

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
        and "nivel_corrientes"
        in upstream_history.columns
    ):

        corr = upstream_history[
            [
                "datetime",
                "nivel_corrientes",
            ]
        ].copy()

        corr[
            "datetime"
        ] = datetime_naive(
            corr[
                "datetime"
            ]
        )

        corr[
            "nivel_corrientes"
        ] = pd.to_numeric(
            corr[
                "nivel_corrientes"
            ],
            errors="coerce",
        )

        corr = corr.dropna()

        local = df[
            [
                "datetime",
                "nivel",
            ]
        ].copy()

        local[
            "datetime"
        ] = datetime_naive(
            local[
                "datetime"
            ]
        )

        local[
            "nivel"
        ] = pd.to_numeric(
            local[
                "nivel"
            ],
            errors="coerce",
        )

        local = local.dropna()

        if (
            not corr.empty
            and not local.empty
        ):

            max_date = max(
                corr[
                    "datetime"
                ].max(),
                local[
                    "datetime"
                ].max(),
            )

            if periodo == "1 año":

                cutoff = (
                    max_date
                    - pd.DateOffset(
                        years=1
                    )
                )

            elif periodo == "5 años":

                cutoff = (
                    max_date
                    - pd.DateOffset(
                        years=5
                    )
                )

            elif periodo == "10 años":

                cutoff = (
                    max_date
                    - pd.DateOffset(
                        years=10
                    )
                )

            elif periodo == "20 años":

                cutoff = (
                    max_date
                    - pd.DateOffset(
                        years=20
                    )
                )

            else:

                cutoff = None

            if cutoff is not None:

                corr_plot = corr[
                    corr[
                        "datetime"
                    ] >= cutoff
                ].copy()

                local_plot = local[
                    local[
                        "datetime"
                    ] >= cutoff
                ].copy()

            else:

                corr_plot = corr.copy()
                local_plot = local.copy()

            # -----------------------------------------------
            # Períodos largos -> promedio mensual.
            # Los datos originales NO se modifican.
            # -----------------------------------------------

            if periodo in [
                "10 años",
                "20 años",
                "Todo el historial",
            ]:

                corr_plot = (
                    corr_plot
                    .set_index(
                        "datetime"
                    )[
                        "nivel_corrientes"
                    ]
                    .resample(
                        "MS"
                    )
                    .mean()
                    .reset_index()
                )

                local_plot = (
                    local_plot
                    .set_index(
                        "datetime"
                    )[
                        "nivel"
                    ]
                    .resample(
                        "MS"
                    )
                    .mean()
                    .reset_index()
                )

            relation_fig = go.Figure()

            relation_fig.add_trace(
                go.Scatter(
                    x=corr_plot[
                        "datetime"
                    ],
                    y=corr_plot[
                        "nivel_corrientes"
                    ],
                    mode="lines",
                    name="Corrientes",
                    line=dict(
                        color="#8B5CF6",
                        width=2.2,
                    ),
                )
            )

            relation_fig.add_trace(
                go.Scatter(
                    x=local_plot[
                        "datetime"
                    ],
                    y=local_plot[
                        "nivel"
                    ],
                    mode="lines",
                    name="San Nicolás",
                    line=dict(
                        color="#2563EB",
                        width=2.4,
                    ),
                )
            )

            relation_fig.update_layout(
                height=460,
                hovermode="x unified",
                margin=dict(
                    l=10,
                    r=10,
                    t=25,
                    b=10,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            relation_fig.update_yaxes(
                title_text="Nivel (m)",
                range=[
                    Y_MIN,
                    Y_MAX,
                ],
                dtick=
                    Y_STEP,
            )

            relation_fig.update_xaxes(
                title_text="Fecha",
                tickformat="%m/%Y",
            )

            st.plotly_chart(
                relation_fig,
                use_container_width=True,
            )

            if periodo in [
                "10 años",
                "20 años",
                "Todo el historial",
            ]:

                st.caption(
                    "Para mejorar la lectura, los períodos "
                    "largos se muestran como promedio mensual. "
                    "El análisis del modelo conserva la resolución "
                    "original disponible."
                )

        else:

            st.info(
                "No existen suficientes datos para comparar "
                "Corrientes y San Nicolás."
            )

    else:

        st.info(
            "Todavía no existe una serie utilizable de Corrientes."
        )


    # ========================================================
    # LLUVIA
    # ========================================================

    st.divider()

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

        rain = (
            exog_future[
                [
                    "datetime",
                    "precip_mm",
                ]
            ]
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

        rain15 = rain.head(
            FORECAST_DAYS
        )

        r1, r2, r3 = st.columns(
            3
        )

        r1.metric(
            "Acumulado 15 días",
            (
                f"{rain15['precip_mm'].sum():.1f} mm"
            ),
        )

        r2.metric(
            "Máximo diario",
            (
                f"{rain15['precip_mm'].max():.1f} mm"
            ),
        )

        r3.metric(
            "Días ≥ 1 mm",
            int(
                (
                    rain15[
                        "precip_mm"
                    ]
                    >= 1.0
                ).sum()
            ),
        )

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=rain15[
                    "datetime"
                ],
                y=rain15[
                    "precip_mm"
                ],
                name="Precipitación",
                marker_color="#38BDF8",
            )
        )

        rain_fig.update_layout(
            height=320,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
            yaxis_title="mm/día",
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
            "No hay precipitación prevista disponible."
        )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.divider()

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

        q_hist[
            "datetime"
        ] = datetime_naive(
            q_hist[
                "datetime"
            ]
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

        texto7 = "Sin dato"

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

        q3.metric(
            "Variación 7 días",
            texto7,
        )

        q4.metric(
            "Tendencia",
            tq[
                "estado"
            ],
        )

        # -----------------------------------------------
        # ORIGEN DEL CAUDAL
        # -----------------------------------------------

        flow_info = {}

        if isinstance(
            exog_meta,
            dict,
        ):

            candidate = exog_meta.get(
                "caudal_series"
            )

            if isinstance(
                candidate,
                dict,
            ):

                flow_info = candidate

        station_flow = (
            flow_info.get(
                "station"
            )
            or flow_info.get(
                "series_name"
            )
        )

        series_flow = (
            flow_info.get(
                "series_id"
            )
        )

        if station_flow:

            caption = (
                f"Serie de caudal INA utilizada: "
                f"**{station_flow}**"
            )

            if series_flow is not None:

                caption += (
                    f" · series_id **{series_flow}**"
                )

            st.caption(
                caption
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
                line=dict(
                    color="#0EA5E9",
                    width=2.5,
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
            ].notna().any()
        ):

            q_future = (
                exog_future
                .dropna(
                    subset=[
                        "caudal_m3s"
                    ]
                )
                .copy()
            )

            q_future[
                "datetime"
            ] = datetime_naive(
                q_future[
                    "datetime"
                ]
            )

            if not q_future.empty:

                connection = pd.DataFrame(
                    {
                        "datetime": [
                            q_hist[
                                "datetime"
                            ].iloc[-1]
                        ],

                        "caudal_m3s": [
                            q_hist[
                                "caudal_m3s"
                            ].iloc[-1]
                        ],
                    }
                )

                q_future_plot = pd.concat(
                    [
                        connection,
                        q_future[
                            [
                                "datetime",
                                "caudal_m3s",
                            ]
                        ],
                    ],
                    ignore_index=True,
                )

                q_fig.add_trace(
                    go.Scatter(
                        x=q_future_plot[
                            "datetime"
                        ],
                        y=q_future_plot[
                            "caudal_m3s"
                        ],
                        mode="lines+markers",
                        name="Proyección de caudal",
                        line=dict(
                            color="#F59E0B",
                            width=2.5,
                            dash="dash",
                        ),
                        marker=dict(
                            size=4,
                        ),
                    )
                )

        q_fig.update_layout(
            height=390,
            hovermode="x unified",
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10,
            ),
            yaxis_title="Caudal (m³/s)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        q_fig.update_xaxes(
            tickformat="%d/%m/%Y",
        )

        st.plotly_chart(
            q_fig,
            use_container_width=True,
        )

    else:

        st.warning(
            "INA no devolvió todavía una serie de caudal "
            "utilizable para el período seleccionado."
        )


    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🛠️ Diagnóstico de fuentes"
    ):

        st.write(
            "**Versión de la aplicación:**",
            APP_VERSION,
        )

        st.write(
            "**Escenario 60 días:**",
            stress_source,
        )

        st.write(
            "**Registros aguas arriba:**",
            (
                len(
                    upstream_history
                )
                if isinstance(
                    upstream_history,
                    pd.DataFrame,
                )
                else 0
            ),
        )

        if isinstance(
            exog_meta,
            dict,
        ):

            st.write(
                "**Registros de caudal:**",
                exog_meta.get(
                    "caudal_records",
                    0,
                ),
            )

            st.write(
                "**Fuente de caudal:**",
                exog_meta.get(
                    "caudal_source",
                    "—",
                ),
            )

            flow_info = (
                exog_meta.get(
                    "caudal_series"
                )
            )

            if flow_info:

                st.json(
                    flow_info
                )

        if (
            upstream_diagnostic_table
            is not None
            and st.session_state.get(
                "fecha_inicio"
            )
            and st.session_state.get(
                "fecha_fin"
            )
        ):

            try:

                diag_up = (
                    upstream_diagnostic_table(
                        st.session_state[
                            "fecha_inicio"
                        ],
                        st.session_state[
                            "fecha_fin"
                        ],
                    )
                )

                if (
                    isinstance(
                        diag_up,
                        pd.DataFrame,
                    )
                    and not diag_up.empty
                ):

                    st.write(
                        "**Series de nivel aguas arriba**"
                    )

                    st.dataframe(
                        diag_up,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:

                st.caption(
                    f"Diagnóstico upstream: {exc}"
                )


    # ========================================================
    # FUENTES
    # ========================================================

    st.divider()

    st.caption(
        "Nivel y caudal: Instituto Nacional del Agua (INA) · "
        "Precipitación: Open-Meteo · "
        "Pronóstico: modelo experimental."
    )

    actualizado = st.session_state.get(
        "actualizado"
    )

    if actualizado:

        st.caption(
            "Última actualización: "
            + actualizado.strftime(
                "%d/%m/%Y %H:%M"
            )
        )
