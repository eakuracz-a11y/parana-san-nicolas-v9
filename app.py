import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import (
    date,
    timedelta,
    datetime,
)

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
    get_stress_scenario,
)

from src.hydrology import (
    analizar_corrientes_san_nicolas,
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# APP V11.8
#
# - Interfaz responsive
# - Pronóstico 1–15 días
# - Tendencia 16–30 días
# - Escenario 31–60 días
# - Compatibilidad con stress_ui viejo y nuevo
# - Continuidad gráfica día 30 -> día 31
# - Niveles aguas arriba
# - Corrientes -> San Nicolás
# ============================================================


APP_VERSION = "V11.8"

FORECAST_DAYS = 15
TREND_DAYS = 30
STRESS_DAYS = 60

Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.80rem;
    }

    div[data-testid="stButton"] > button {
        min-height: 3rem;
        border-radius: 0.65rem;
        font-weight: 600;
    }

    div[data-testid="stAlert"] {
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
    }

    [data-testid="stDataFrame"] {
        overflow-x: auto;
    }

    .period-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 0.2rem;
        margin-bottom: 0.15rem;
    }

    .period-help {
        font-size: 0.86rem;
        opacity: 0.75;
        margin-bottom: 0.45rem;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-top: 0.55rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-bottom: 2rem !important;
        }

        h1 {
            font-size: 1.50rem !important;
            line-height: 1.12 !important;
            margin-bottom: 0.25rem !important;
        }

        h2 {
            font-size: 1.23rem !important;
        }

        h3 {
            font-size: 1.07rem !important;
        }

        p {
            font-size: 0.88rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.08rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.68rem !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.68rem !important;
        }

        div[data-testid="stButton"] > button {
            min-height: 3.15rem !important;
            width: 100% !important;
            font-size: 0.95rem !important;
        }

        div[data-testid="stDateInput"] input {
            font-size: 0.92rem !important;
        }

        div[data-testid="stAlert"] {
            font-size: 0.80rem !important;
        }

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
    f"{APP_VERSION} · Monitoreo y pronóstico experimental"
)

st.markdown(
    """
    Seguimiento del río Paraná en **San Nicolás de los Arroyos**,
    incorporando nivel, estaciones aguas arriba, caudal,
    precipitación y comportamiento histórico.
    """
)


# ============================================================
# PERÍODO
# ============================================================

st.markdown(
    '<div class="period-title">📅 Período de análisis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="period-help">
    Este período se utiliza para la situación actual
    y el entrenamiento del pronóstico.
    </div>
    """,
    unsafe_allow_html=True,
)


fecha_hasta_default = date.today()

fecha_desde_default = (
    fecha_hasta_default
    - timedelta(
        days=365
    )
)


fc1, fc2 = st.columns(
    2,
    gap="small",
)


with fc1:

    desde = st.date_input(
        "Desde",
        value=fecha_desde_default,
        format="DD/MM/YYYY",
        key="fecha_desde_principal",
    )


with fc2:

    hasta = st.date_input(
        "Hasta",
        value=fecha_hasta_default,
        format="DD/MM/YYYY",
        key="fecha_hasta_principal",
    )


actualizar = st.button(
    "🔄 Actualizar modelo",
    type="primary",
    use_container_width=True,
)


h1, h2, h3 = st.columns(
    3
)

h1.caption(
    "🟠 Pronóstico · 1–15 días"
)

h2.caption(
    "🟢 Tendencia · 16–30 días"
)

h3.caption(
    "🔴 Escenario · 31–60 días"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌊 Paraná"
)

st.sidebar.subheader(
    "San Nicolás"
)

st.sidebar.caption(
    f"Versión {APP_VERSION}"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Horizontes**

    🟠 1–15 días  
    Pronóstico

    🟢 16–30 días  
    Tendencia extendida

    🔴 31–60 días  
    Escenario hidrológico
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Variables**

    • Nivel San Nicolás  
    • Niveles aguas arriba  
    • Caudal  
    • Precipitación  
    • Corrientes → San Nicolás
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Fuentes**

    INA  
    Open-Meteo
    """
)

st.sidebar.caption(
    "Escala de nivel: 0–7 m"
)


# ============================================================
# AUXILIARES
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


def texto_tendencia(
    delta,
):

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


def normalizar_estacion(
    texto,
):

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
# EXÓGENAS
# ============================================================


def obtener_exogenas(
    inicio,
    fin,
):

    try:

        result = get_exogenous_data(
            inicio,
            fin,
            FORECAST_DAYS,
        )

        if isinstance(
            result,
            tuple,
        ):

            if len(result) >= 3:

                return (
                    result[0],
                    result[1],
                    result[2],
                    None,
                )

            if len(result) == 2:

                return (
                    result[0],
                    result[1],
                    {},
                    None,
                )

        if isinstance(
            result,
            pd.DataFrame,
        ):

            return (
                result,
                pd.DataFrame(),
                {},
                None,
            )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            "Formato de respuesta no reconocido.",
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            str(exc),
        )


# ============================================================
# UPSTREAM
# ============================================================


def obtener_upstream(
    inicio,
    fin,
):

    try:

        result = get_upstream_history(
            inicio,
            fin,
        )

        if isinstance(
            result,
            tuple,
        ):

            history = (
                result[0]
                if len(result) >= 1
                else pd.DataFrame()
            )

            meta = (
                result[1]
                if len(result) >= 2
                else {}
            )

        elif isinstance(
            result,
            pd.DataFrame,
        ):

            history = result
            meta = {}

        else:

            history = pd.DataFrame()
            meta = {}

        return (
            history,
            meta,
            None,
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            {},
            str(exc),
        )


# ============================================================
# TABLA AGUAS ARRIBA
# ============================================================


def construir_tabla_upstream(
    upstream_history,
    upstream_meta,
):

    stations = [
        "Corrientes",
        "Goya",
        "La Paz",
        "Paraná",
        "Diamante",
        "Rosario",
        "Villa Constitución",
    ]

    rows = []

    for station in stations:

        col = (
            "nivel_"
            + normalizar_estacion(
                station
            )
        )

        actual = None
        variacion = None
        fecha = None

        if (
            isinstance(
                upstream_history,
                pd.DataFrame,
            )
            and not upstream_history.empty
            and col in upstream_history.columns
        ):

            temp = upstream_history[
                [
                    "datetime",
                    col,
                ]
            ].copy()

            temp["datetime"] = pd.to_datetime(
                temp["datetime"],
                errors="coerce",
                utc=True,
            )

            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce",
            )

            temp = (
                temp
                .dropna()
                .sort_values(
                    "datetime"
                )
            )

            if not temp.empty:

                actual = float(
                    temp[col].iloc[-1]
                )

                fecha = (
                    temp["datetime"].iloc[-1]
                )

                if len(temp) >= 2:

                    variacion = (
                        actual
                        - float(
                            temp[col].iloc[-2]
                        )
                    )

        info = {}

        if isinstance(
            upstream_meta,
            dict,
        ):

            info = upstream_meta.get(
                station,
                {},
            )

            if not isinstance(
                info,
                dict,
            ):
                info = {}

        rows.append(
            {
                "Estación":
                    station,

                "Nivel":
                    (
                        f"{actual:.2f} m"
                        if actual is not None
                        else "—"
                    ),

                "Variación":
                    (
                        f"{variacion:+.2f} m"
                        if variacion is not None
                        else "—"
                    ),

                "Tendencia":
                    (
                        texto_tendencia(
                            variacion
                        )
                        if actual is not None
                        else "Sin datos"
                    ),

                "Fecha":
                    (
                        pd.to_datetime(
                            fecha
                        ).strftime(
                            "%d/%m"
                        )
                        if fecha is not None
                        else "—"
                    ),

                "Serie":
                    info.get(
                        "series_id",
                        "—",
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TENDENCIA 30 DÍAS
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

    last_date = pd.Timestamp(
        f["datetime"].iloc[-1]
    )

    last_level = float(
        f["prediction"].iloc[-1]
    )

    recent_forecast = (
        f["prediction"]
        .tail(5)
        .dropna()
    )

    if len(
        recent_forecast
    ) >= 3:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        recent_forecast
                    )
                ),
                recent_forecast.to_numpy(
                    dtype=float
                ),
                1,
            )[0]
        )

    else:

        recent_obs = (
            pd.to_numeric(
                df["nivel"],
                errors="coerce",
            )
            .dropna()
            .tail(7)
        )

        if len(
            recent_obs
        ) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(
                            recent_obs
                        )
                    ),
                    recent_obs.to_numpy(
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
        FORECAST_DAYS + 1,
        TREND_DAYS + 1,
    ):

        damping = np.exp(
            -0.12
            * (
                day
                - FORECAST_DAYS
            )
        )

        daily_change = (
            slope
            * damping
        )

        daily_change = float(
            np.clip(
                daily_change,
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
# ANCLA DÍA 30
# ============================================================


def obtener_ancla_dia_30(
    forecast30,
):

    if (
        forecast30 is None
        or not isinstance(
            forecast30,
            pd.DataFrame,
        )
        or forecast30.empty
        or "datetime"
        not in forecast30.columns
        or "prediction"
        not in forecast30.columns
    ):

        return (
            None,
            None,
        )

    x = forecast30.copy()

    x["datetime"] = pd.to_datetime(
        x["datetime"],
        errors="coerce",
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

        return (
            None,
            None,
        )

    if len(x) >= TREND_DAYS:

        row = x.iloc[
            TREND_DAYS - 1
        ]

    else:

        row = x.iloc[-1]

    return (
        pd.Timestamp(
            row["datetime"]
        ),
        float(
            row["prediction"]
        ),
    )


# ============================================================
# ADAPTAR STRESS ANTIGUO
# ============================================================


def adaptar_stress_antiguo(
    stress_old,
    anchor_date,
    anchor_level,
):

    if (
        stress_old is None
        or not isinstance(
            stress_old,
            pd.DataFrame,
        )
        or stress_old.empty
        or "stress_level"
        not in stress_old.columns
    ):

        return pd.DataFrame()

    temp = stress_old.copy()

    temp["datetime"] = pd.to_datetime(
        temp["datetime"],
        errors="coerce",
    )

    temp["stress_level"] = pd.to_numeric(
        temp["stress_level"],
        errors="coerce",
    )

    temp = temp.dropna(
        subset=[
            "datetime",
            "stress_level",
        ]
    )

    if temp.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # TOMAR DÍAS 31-60
    # --------------------------------------------------------

    if "scenario_day" in temp.columns:

        temp["scenario_day"] = pd.to_numeric(
            temp["scenario_day"],
            errors="coerce",
        )

        temp = temp[
            temp["scenario_day"]
            > TREND_DAYS
        ].copy()

    else:

        temp = temp.iloc[
            TREND_DAYS:
            STRESS_DAYS
        ].copy()

        temp["scenario_day"] = np.arange(
            TREND_DAYS + 1,
            TREND_DAYS + 1 + len(temp),
        )

    if temp.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # DESPLAZAR TODA LA CURVA PARA QUE ARRANQUE
    # EN EL NIVEL DEL DÍA 30
    # --------------------------------------------------------

    first_old_level = float(
        temp[
            "stress_level"
        ].iloc[0]
    )

    offset = (
        float(anchor_level)
        - first_old_level
    )

    temp["stress_level"] = (
        temp["stress_level"]
        + offset
    )

    temp["stress_level"] = (
        temp["stress_level"]
        .clip(
            lower=Y_MIN,
            upper=Y_MAX,
        )
    )

    # --------------------------------------------------------
    # FECHAS CORRECTAS 31-60
    # --------------------------------------------------------

    temp["datetime"] = pd.date_range(
        start=(
            pd.Timestamp(
                anchor_date
            )
            + pd.Timedelta(
                days=1
            )
        ),
        periods=len(temp),
        freq="D",
    )

    # --------------------------------------------------------
    # VARIACIÓN DIARIA
    # --------------------------------------------------------

    temp["daily_change"] = (
        temp[
            "stress_level"
        ].diff()
    )

    temp.loc[
        temp.index[0],
        "daily_change",
    ] = (
        float(
            temp[
                "stress_level"
            ].iloc[0]
        )
        - float(
            anchor_level
        )
    )

    temp["anchor_level"] = float(
        anchor_level
    )

    temp["anchor_day"] = (
        TREND_DAYS
    )

    return temp.reset_index(
        drop=True
    )


# ============================================================
# CREAR ESCENARIO 31-60
# ============================================================


def construir_stress_compatible(
    df,
    models,
    exog_history,
    upstream_history,
    anchor_date,
    anchor_level,
):

    if (
        anchor_date is None
        or anchor_level is None
    ):

        return (
            pd.DataFrame(),
            "sin_ancla",
        )

    # ========================================================
    # PRIMER INTENTO:
    # stress_ui V11.7+
    # ========================================================

    try:

        scenario = get_stress_scenario(
            df=df,
            models=models,
            exog_history=
                exog_history,
            upstream_history=
                upstream_history,
            days=
                STRESS_DAYS,
            anchor_date=
                anchor_date,
            anchor_level=
                anchor_level,
            anchor_day=
                TREND_DAYS,
        )

        return (
            scenario,
            "stress_nuevo",
        )

    except TypeError as exc:

        mensaje = str(exc)

        incompatible = (
            "anchor_date" in mensaje
            or "anchor_level" in mensaje
            or "anchor_day" in mensaje
        )

        if not incompatible:

            raise

    # ========================================================
    # SEGUNDO INTENTO:
    # stress_ui ANTERIOR
    # ========================================================

    stress_old = get_stress_scenario(
        df=df,
        models=models,
        exog_history=
            exog_history,
        upstream_history=
            upstream_history,
        days=
            STRESS_DAYS,
    )

    scenario = adaptar_stress_antiguo(
        stress_old=
            stress_old,
        anchor_date=
            anchor_date,
        anchor_level=
            anchor_level,
    )

    return (
        scenario,
        "stress_compatibilidad",
    )


# ============================================================
# TABLA EVENTOS
# ============================================================


def preparar_tabla_eventos(
    events,
):

    if (
        events is None
        or not isinstance(
            events,
            pd.DataFrame,
        )
        or events.empty
    ):
        return pd.DataFrame()

    tabla = events.copy()

    rename = {
        "fecha_max_corrientes":
            "Fecha Corrientes",

        "max_corrientes_m":
            "Máx. Corrientes",

        "fecha_max_san_nicolas":
            "Fecha San Nicolás",

        "max_san_nicolas_m":
            "Máx. San Nicolás",

        "lag_real_dias":
            "Retardo",

        "respuesta_san_nicolas_m":
            "Crecimiento SN",

        "lluvia_previa_mm":
            "Lluvia previa",

        "caudal_medio_m3s":
            "Caudal medio",
    }

    tabla = tabla.rename(
        columns=rename
    )

    for col in [
        "Fecha Corrientes",
        "Fecha San Nicolás",
    ]:

        if col in tabla.columns:

            tabla[col] = (
                pd.to_datetime(
                    tabla[col],
                    errors="coerce",
                )
                .dt.strftime(
                    "%d/%m/%Y"
                )
            )

    for col in [
        "Máx. Corrientes",
        "Máx. San Nicolás",
        "Crecimiento SN",
    ]:

        if col in tabla.columns:

            tabla[col] = (
                pd.to_numeric(
                    tabla[col],
                    errors="coerce",
                )
                .round(2)
            )

    return tabla


# ============================================================
# VALIDACIÓN
# ============================================================

if desde > hasta:

    st.error(
        "⚠️ La fecha inicial no puede ser posterior "
        "a la fecha final."
    )


# ============================================================
# ACTUALIZACIÓN
# ============================================================

if actualizar:

    if desde <= hasta:

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
            "Consultando San Nicolás..."
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

                df_ina = pd.DataFrame()
                error_ina = str(exc)

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
                # EXÓGENAS
                # ============================================

                with st.spinner(
                    "Consultando lluvia y caudal..."
                ):

                    (
                        exog_history,
                        exog_future,
                        exog_meta,
                        exog_error,
                    ) = obtener_exogenas(
                        inicio,
                        fin,
                    )

                if exog_error:

                    st.warning(
                        "Algunos datos externos "
                        "no están disponibles. "
                        f"{exog_error}"
                    )

                # ============================================
                # UPSTREAM
                # ============================================

                with st.spinner(
                    "Consultando estaciones aguas arriba..."
                ):

                    (
                        upstream_history,
                        upstream_meta,
                        upstream_error,
                    ) = obtener_upstream(
                        inicio,
                        fin,
                    )

                if upstream_error:

                    st.warning(
                        "Algunas estaciones aguas arriba "
                        "no están disponibles. "
                        f"{upstream_error}"
                    )

                # ============================================
                # HIDROLOGÍA
                # ============================================

                with st.spinner(
                    "Analizando Corrientes → San Nicolás..."
                ):

                    try:

                        hydro_analysis = (
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

                            hydro_analysis = (
                                analizar_corrientes_san_nicolas(
                                    san_nicolas=df,
                                    upstream_history=
                                        upstream_history,
                                    exog_history=
                                        exog_history,
                                )
                            )

                        except Exception as exc:

                            hydro_analysis = {}

                            st.warning(
                                "No se pudo completar "
                                "el análisis Corrientes → San Nicolás. "
                                f"{exc}"
                            )

                    except Exception as exc:

                        hydro_analysis = {}

                        st.warning(
                            "No se pudo completar "
                            "el análisis Corrientes → San Nicolás. "
                            f"{exc}"
                        )

                # ============================================
                # MODELO 15 DÍAS
                # ============================================

                with st.spinner(
                    "Entrenando pronóstico..."
                ):

                    try:

                        train_result = train(
                            df=df,
                            exog_history=
                                exog_history,
                            upstream_history=
                                upstream_history,
                        )

                        if (
                            isinstance(
                                train_result,
                                tuple,
                            )
                            and len(
                                train_result
                            ) >= 2
                        ):

                            models = train_result[0]
                            metrics = train_result[1]

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

                            forecast = predict(
                                df=df,
                                models=models,
                                days=
                                    FORECAST_DAYS,
                                exog_future=
                                    exog_future,
                            )

                        except TypeError:

                            forecast = predict(
                                df=df,
                                models=models,
                                days=
                                    FORECAST_DAYS,
                            )

                    except Exception as exc:

                        models = {}
                        metrics = {}
                        forecast = pd.DataFrame()

                        st.error(
                            "No fue posible generar "
                            "el pronóstico de 15 días. "
                            f"{exc}"
                        )

                # ============================================
                # TENDENCIA 30 DÍAS
                # ============================================

                forecast30 = (
                    extender_pronostico_30(
                        forecast,
                        df,
                    )
                )

                (
                    anchor_date,
                    anchor_level,
                ) = obtener_ancla_dia_30(
                    forecast30
                )

                # ============================================
                # ESCENARIO 31–60
                # ============================================

                stress_source = (
                    "no_disponible"
                )

                with st.spinner(
                    "Construyendo escenario 31–60 días..."
                ):

                    try:

                        (
                            stress60,
                            stress_source,
                        ) = construir_stress_compatible(
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

                    except Exception as exc:

                        stress60 = (
                            pd.DataFrame()
                        )

                        stress_source = (
                            "error"
                        )

                        st.warning(
                            "No fue posible construir "
                            "el escenario de 60 días. "
                            f"{exc}"
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
                    "forecast30"
                ] = forecast30

                st.session_state[
                    "stress60"
                ] = stress60

                st.session_state[
                    "stress_source"
                ] = stress_source

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
                    "hydro_analysis"
                ] = hydro_analysis

                st.session_state[
                    "anchor_date"
                ] = anchor_date

                st.session_state[
                    "anchor_level"
                ] = anchor_level

                st.session_state[
                    "actualizado"
                ] = datetime.now()

                st.success(
                    "✅ Datos y modelo actualizados correctamente."
                )


# ============================================================
# SIN DATOS TODAVÍA
# ============================================================

if "datos" not in st.session_state:

    st.info(
        "👆 Seleccioná el período y presioná "
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

    stress60 = st.session_state.get(
        "stress60",
        pd.DataFrame(),
    )

    stress_source = st.session_state.get(
        "stress_source",
        "no_disponible",
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

    hydro_analysis = st.session_state.get(
        "hydro_analysis",
        {},
    )

    anchor_date = st.session_state.get(
        "anchor_date"
    )

    anchor_level = st.session_state.get(
        "anchor_level"
    )

    actualizado = st.session_state.get(
        "actualizado"
    )

    # ========================================================
    # NIVEL ACTUAL
    # ========================================================

    niveles = (
        pd.to_numeric(
            df["nivel"],
            errors="coerce",
        )
        .dropna()
    )

    if niveles.empty:

        st.error(
            "No hay niveles válidos."
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

    # ========================================================
    # CAUDAL
    # ========================================================

    caudal_actual = None

    if (
        isinstance(
            exog_history,
            pd.DataFrame,
        )
        and not exog_history.empty
        and "caudal_m3s"
        in exog_history.columns
    ):

        q = (
            pd.to_numeric(
                exog_history[
                    "caudal_m3s"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if not q.empty:

            caudal_actual = float(
                q.iloc[-1]
            )

    # ========================================================
    # LLUVIA
    # ========================================================

    lluvia_15 = None

    if (
        isinstance(
            exog_future,
            pd.DataFrame,
        )
        and not exog_future.empty
        and "precip_mm"
        in exog_future.columns
    ):

        lluvia_15 = float(
            pd.to_numeric(
                exog_future[
                    "precip_mm"
                ],
                errors="coerce",
            )
            .fillna(0)
            .head(
                FORECAST_DAYS
            )
            .sum()
        )

    # ========================================================
    # SITUACIÓN ACTUAL
    # ========================================================

    st.subheader(
        "📍 Situación actual"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Nivel San Nicolás",
        f"{nivel_actual:.2f} m",
        (
            f"{delta_actual:+.2f} m"
            if delta_actual is not None
            else None
        ),
    )

    c2.metric(
        "Tendencia",
        texto_tendencia(
            delta_actual
        ),
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Caudal",
        (
            f"{caudal_actual:,.0f} m³/s"
            if caudal_actual is not None
            else "Sin dato"
        ),
    )

    c4.metric(
        "Lluvia prevista · 15 días",
        (
            f"{lluvia_15:.1f} mm"
            if lluvia_15 is not None
            else "Sin dato"
        ),
    )

    # ========================================================
    # SEGMENTOS
    # ========================================================

    observed_plot = (
        df[
            [
                "datetime",
                "nivel",
            ]
        ]
        .copy()
    )

    observed_plot[
        "datetime"
    ] = pd.to_datetime(
        observed_plot[
            "datetime"
        ],
        errors="coerce",
    )

    observed_plot[
        "nivel"
    ] = pd.to_numeric(
        observed_plot[
            "nivel"
        ],
        errors="coerce",
    )

    observed_plot = (
        observed_plot.dropna()
    )

    # --------------------------------------------------------
    # 1-15
    # --------------------------------------------------------

    seg15 = pd.DataFrame()

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
        and "prediction"
        in forecast.columns
    ):

        seg15 = (
            forecast.head(
                FORECAST_DAYS
            )[
                [
                    "datetime",
                    "prediction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "prediction":
                        "nivel"
                }
            )
        )

        seg15["datetime"] = pd.to_datetime(
            seg15["datetime"],
            errors="coerce",
        )

        seg15["nivel"] = pd.to_numeric(
            seg15["nivel"],
            errors="coerce",
        )

        seg15 = seg15.dropna()

    # --------------------------------------------------------
    # 16-30
    # --------------------------------------------------------

    seg30 = pd.DataFrame()

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and not forecast30.empty
        and "prediction"
        in forecast30.columns
    ):

        temp30 = forecast30.copy()

        temp30["datetime"] = pd.to_datetime(
            temp30["datetime"],
            errors="coerce",
        )

        temp30["prediction"] = pd.to_numeric(
            temp30["prediction"],
            errors="coerce",
        )

        temp30 = temp30.dropna(
            subset=[
                "datetime",
                "prediction",
            ]
        )

        seg30 = (
            temp30.iloc[
                FORECAST_DAYS:
                TREND_DAYS
            ][
                [
                    "datetime",
                    "prediction",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "prediction":
                        "nivel"
                }
            )
        )

    # --------------------------------------------------------
    # 31-60
    # --------------------------------------------------------

    seg60 = pd.DataFrame()

    if (
        isinstance(
            stress60,
            pd.DataFrame,
        )
        and not stress60.empty
        and "stress_level"
        in stress60.columns
    ):

        seg60 = (
            stress60[
                [
                    "datetime",
                    "stress_level",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "stress_level":
                        "nivel"
                }
            )
        )

        seg60["datetime"] = pd.to_datetime(
            seg60["datetime"],
            errors="coerce",
        )

        seg60["nivel"] = pd.to_numeric(
            seg60["nivel"],
            errors="coerce",
        )

        seg60 = seg60.dropna()

    # ========================================================
    # CONTINUIDAD VISUAL
    # ========================================================

    if (
        not observed_plot.empty
        and not seg15.empty
    ):

        start15 = pd.DataFrame(
            [
                {
                    "datetime":
                        observed_plot[
                            "datetime"
                        ].iloc[-1],

                    "nivel":
                        float(
                            observed_plot[
                                "nivel"
                            ].iloc[-1]
                        ),
                }
            ]
        )

        seg15_plot = pd.concat(
            [
                start15,
                seg15,
            ],
            ignore_index=True,
        )

    else:

        seg15_plot = seg15

    if (
        not seg15.empty
        and not seg30.empty
    ):

        start30 = pd.DataFrame(
            [
                {
                    "datetime":
                        seg15[
                            "datetime"
                        ].iloc[-1],

                    "nivel":
                        float(
                            seg15[
                                "nivel"
                            ].iloc[-1]
                        ),
                }
            ]
        )

        seg30_plot = pd.concat(
            [
                start30,
                seg30,
            ],
            ignore_index=True,
        )

    else:

        seg30_plot = seg30

    if (
        not seg30.empty
        and not seg60.empty
    ):

        start60 = pd.DataFrame(
            [
                {
                    "datetime":
                        seg30[
                            "datetime"
                        ].iloc[-1],

                    "nivel":
                        float(
                            seg30[
                                "nivel"
                            ].iloc[-1]
                        ),
                }
            ]
        )

        seg60_plot = pd.concat(
            [
                start60,
                seg60,
            ],
            ignore_index=True,
        )

    else:

        seg60_plot = seg60

    # ========================================================
    # GRÁFICO 60 DÍAS
    # ========================================================

    st.subheader(
        "📈 Proyección hidrológica · 60 días"
    )

    st.caption(
        "La curva se presenta como una trayectoria continua: "
        "pronóstico 1–15 días, tendencia 16–30 días y "
        "escenario hidrológico 31–60 días."
    )

    fig = go.Figure()

    # Observado
    if not observed_plot.empty:

        recent_obs = (
            observed_plot.tail(60)
        )

        fig.add_trace(
            go.Scatter(
                x=recent_obs[
                    "datetime"
                ],
                y=recent_obs[
                    "nivel"
                ],
                mode="lines",
                name="Observado",
                line=dict(
                    color="#2563EB",
                    width=3,
                ),
            )
        )

    # 1–15
    if not seg15_plot.empty:

        fig.add_trace(
            go.Scatter(
                x=seg15_plot[
                    "datetime"
                ],
                y=seg15_plot[
                    "nivel"
                ],
                mode="lines+markers",
                name="1–15 días",
                line=dict(
                    color="#F59E0B",
                    width=4,
                ),
                marker=dict(
                    size=5,
                ),
            )
        )

    # 16–30
    if not seg30_plot.empty:

        fig.add_trace(
            go.Scatter(
                x=seg30_plot[
                    "datetime"
                ],
                y=seg30_plot[
                    "nivel"
                ],
                mode="lines+markers",
                name="16–30 días",
                line=dict(
                    color="#16A34A",
                    width=4,
                ),
                marker=dict(
                    size=4,
                ),
            )
        )

    # 31–60
    if not seg60_plot.empty:

        fig.add_trace(
            go.Scatter(
                x=seg60_plot[
                    "datetime"
                ],
                y=seg60_plot[
                    "nivel"
                ],
                mode="lines+markers",
                name="31–60 días",
                line=dict(
                    color="#DC2626",
                    width=4,
                ),
                marker=dict(
                    size=4,
                ),
            )
        )

    # ========================================================
    # INCERTIDUMBRE
    # ========================================================

    if (
        isinstance(
            forecast,
            pd.DataFrame,
        )
        and not forecast.empty
        and "upper" in forecast.columns
        and "lower" in forecast.columns
    ):

        uncertainty = (
            forecast.head(
                FORECAST_DAYS
            )
            .copy()
        )

        uncertainty["datetime"] = pd.to_datetime(
            uncertainty["datetime"],
            errors="coerce",
        )

        uncertainty["upper"] = pd.to_numeric(
            uncertainty["upper"],
            errors="coerce",
        )

        uncertainty["lower"] = pd.to_numeric(
            uncertainty["lower"],
            errors="coerce",
        )

        uncertainty = (
            uncertainty.dropna(
                subset=[
                    "datetime",
                    "upper",
                    "lower",
                ]
            )
        )

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
                    fillcolor=
                        "rgba(245,158,11,0.15)",
                    name="Incertidumbre",
                    hoverinfo="skip",
                )
            )

    # ========================================================
    # SEPARADORES
    # ========================================================

    if not seg15.empty:

        fig.add_vline(
            x=seg15[
                "datetime"
            ].iloc[-1],
            line_width=1,
            line_dash="dot",
            line_color="#6B7280",
        )

    if not seg30.empty:

        fig.add_vline(
            x=seg30[
                "datetime"
            ].iloc[-1],
            line_width=1,
            line_dash="dot",
            line_color="#6B7280",
        )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=30,
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

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m",
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

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # HORIZONTES
    # ========================================================

    m1, m2, m3 = st.columns(3)

    if not seg15.empty:

        nivel15 = float(
            seg15["nivel"].iloc[-1]
        )

        m1.metric(
            "🟠 Día 15",
            f"{nivel15:.2f} m",
            f"{nivel15 - nivel_actual:+.2f} m",
        )

    else:

        m1.metric(
            "🟠 Día 15",
            "Sin dato",
        )

    if anchor_level is not None:

        nivel30 = float(
            anchor_level
        )

        m2.metric(
            "🟢 Día 30",
            f"{nivel30:.2f} m",
            f"{nivel30 - nivel_actual:+.2f} m",
        )

    else:

        m2.metric(
            "🟢 Día 30",
            "Sin dato",
        )

    if not seg60.empty:

        nivel60 = float(
            seg60["nivel"].iloc[-1]
        )

        m3.metric(
            "🔴 Día 60",
            f"{nivel60:.2f} m",
            f"{nivel60 - nivel_actual:+.2f} m",
        )

    else:

        m3.metric(
            "🔴 Día 60",
            "Sin dato",
        )

    # ========================================================
    # LLUVIA
    # ========================================================

    with st.expander(
        "🌧️ Precipitación prevista · 15 días",
        expanded=False,
    ):

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

            rain["datetime"] = pd.to_datetime(
                rain["datetime"],
                errors="coerce",
            )

            rain["precip_mm"] = pd.to_numeric(
                rain["precip_mm"],
                errors="coerce",
            ).fillna(0)

            rain = (
                rain
                .dropna(
                    subset=[
                        "datetime"
                    ]
                )
                .head(
                    FORECAST_DAYS
                )
            )

            rc1, rc2 = st.columns(2)

            rc1.metric(
                "Acumulado",
                f"{rain['precip_mm'].sum():.1f} mm",
            )

            rc2.metric(
                "Máximo diario",
                f"{rain['precip_mm'].max():.1f} mm",
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
                    name="Lluvia",
                    marker_color="#3B82F6",
                )
            )

            rain_fig.update_layout(
                height=280,
                yaxis_title="mm/día",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
            )

            st.plotly_chart(
                rain_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No hay precipitación futura disponible."
            )

    # ========================================================
    # CAUDAL
    # ========================================================

    with st.expander(
        "💧 Caudal",
        expanded=False,
    ):

        if (
            isinstance(
                exog_history,
                pd.DataFrame,
            )
            and not exog_history.empty
            and "caudal_m3s"
            in exog_history.columns
        ):

            qdf = (
                exog_history[
                    [
                        "datetime",
                        "caudal_m3s",
                    ]
                ]
                .copy()
            )

            qdf["datetime"] = pd.to_datetime(
                qdf["datetime"],
                errors="coerce",
            )

            qdf["caudal_m3s"] = pd.to_numeric(
                qdf["caudal_m3s"],
                errors="coerce",
            )

            qdf = qdf.dropna()

            if not qdf.empty:

                qfig = go.Figure()

                qfig.add_trace(
                    go.Scatter(
                        x=qdf[
                            "datetime"
                        ],
                        y=qdf[
                            "caudal_m3s"
                        ],
                        mode="lines",
                        name="Caudal",
                        line=dict(
                            color="#0891B2",
                            width=3,
                        ),
                    )
                )

                qfig.update_layout(
                    height=290,
                    yaxis_title="m³/s",
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10,
                    ),
                )

                st.plotly_chart(
                    qfig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No hay valores válidos."
                )

        else:

            st.info(
                "No hay caudal disponible."
            )

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    with st.expander(
        "🌊 Niveles aguas arriba",
        expanded=False,
    ):

        tabla_upstream = construir_tabla_upstream(
            upstream_history,
            upstream_meta,
        )

        st.dataframe(
            tabla_upstream,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # CORRIENTES -> SAN NICOLÁS
    # ========================================================

    st.subheader(
        "🌊 Corrientes → San Nicolás"
    )

    st.caption(
        "Relación entre niveles y propagación de crecidas."
    )

    lag_info = {}
    events = pd.DataFrame()
    hydro_stats = {}
    top_events = pd.DataFrame()

    if isinstance(
        hydro_analysis,
        dict,
    ):

        lag_info = hydro_analysis.get(
            "lag",
            {},
        )

        events = hydro_analysis.get(
            "events",
            pd.DataFrame(),
        )

        hydro_stats = hydro_analysis.get(
            "statistics",
            {},
        )

        top_events = hydro_analysis.get(
            "top_events",
            pd.DataFrame(),
        )

    best_lag = None
    correlation = np.nan

    if isinstance(
        lag_info,
        dict,
    ):

        best_lag = lag_info.get(
            "best_lag_days"
        )

        correlation = lag_info.get(
            "correlation",
            np.nan,
        )

    hc1, hc2, hc3 = st.columns(3)

    hc1.metric(
        "Retardo estimado",
        (
            f"{int(best_lag)} días"
            if best_lag is not None
            else "—"
        ),
    )

    hc2.metric(
        "Correlación",
        (
            f"{float(correlation):.2f}"
            if np.isfinite(
                correlation
            )
            else "—"
        ),
    )

    hc3.metric(
        "Eventos",
        (
            len(events)
            if isinstance(
                events,
                pd.DataFrame,
            )
            else 0
        ),
    )

    # ========================================================
    # GRÁFICO CORRIENTES
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

        corr = (
            upstream_history[
                [
                    "datetime",
                    "nivel_corrientes",
                ]
            ]
            .copy()
        )

        corr["datetime"] = pd.to_datetime(
            corr["datetime"],
            errors="coerce",
            utc=True,
        )

        corr["nivel_corrientes"] = pd.to_numeric(
            corr["nivel_corrientes"],
            errors="coerce",
        )

        corr = corr.dropna()

        if not corr.empty:

            relation_fig = go.Figure()

            relation_fig.add_trace(
                go.Scatter(
                    x=corr[
                        "datetime"
                    ],
                    y=corr[
                        "nivel_corrientes"
                    ],
                    mode="lines",
                    name="Corrientes",
                    line=dict(
                        color="#8B5CF6",
                        width=3,
                    ),
                )
            )

            relation_fig.add_trace(
                go.Scatter(
                    x=df[
                        "datetime"
                    ],
                    y=df[
                        "nivel"
                    ],
                    mode="lines",
                    name="San Nicolás",
                    line=dict(
                        color="#2563EB",
                        width=3,
                    ),
                )
            )

            relation_fig.update_layout(
                height=400,
                hovermode="x unified",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
                legend=dict(
                    orientation="h",
                    y=1.05,
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

            st.plotly_chart(
                relation_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "Corrientes no tiene valores válidos."
            )

    else:

        st.info(
            "Corrientes todavía no tiene suficientes "
            "datos disponibles."
        )

    # ========================================================
    # EVENTOS
    # ========================================================

    with st.expander(
        "📚 Máximos Corrientes → San Nicolás",
        expanded=False,
    ):

        tabla_eventos = preparar_tabla_eventos(
            top_events
        )

        if not tabla_eventos.empty:

            st.dataframe(
                tabla_eventos,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "Todavía no hay suficientes eventos históricos."
            )

    # ========================================================
    # DETALLE ESCENARIO
    # ========================================================

    with st.expander(
        "⚠️ Detalle escenario 31–60 días",
        expanded=False,
    ):

        if (
            isinstance(
                stress60,
                pd.DataFrame,
            )
            and not stress60.empty
        ):

            detail_cols = [
                col
                for col in [
                    "datetime",
                    "scenario_day",
                    "stress_level",
                    "daily_change",
                    "historical_level_max_m",
                    "rain_historical_max_mm",
                    "flow_historical_max_m3s",
                    "upstream_signal",
                    "current_flow_trend",
                ]
                if col in stress60.columns
            ]

            detail = (
                stress60[
                    detail_cols
                ]
                .copy()
            )

            if "datetime" in detail.columns:

                detail["datetime"] = pd.to_datetime(
                    detail["datetime"],
                    errors="coerce",
                ).dt.strftime(
                    "%d/%m/%Y"
                )

            detail = detail.rename(
                columns={
                    "datetime":
                        "Fecha",

                    "scenario_day":
                        "Día",

                    "stress_level":
                        "Nivel escenario (m)",

                    "daily_change":
                        "Variación diaria (m)",

                    "historical_level_max_m":
                        "Máx. nivel histórico (m)",

                    "rain_historical_max_mm":
                        "Máx. lluvia histórica (mm)",

                    "flow_historical_max_m3s":
                        "Máx. caudal histórico (m³/s)",

                    "upstream_signal":
                        "Señal aguas arriba",

                    "current_flow_trend":
                        "Tendencia caudal",
                }
            )

            st.dataframe(
                detail,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No hay escenario disponible."
            )

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🧠 Diagnóstico técnico",
        expanded=False,
    ):

        st.write(
            "**Versión app:**",
            APP_VERSION,
        )

        st.write(
            "**Motor escenario 60 días:**",
            stress_source,
        )

        st.write(
            "**Nivel ancla día 30:**",
            (
                f"{anchor_level:.2f} m"
                if anchor_level is not None
                else "Sin dato"
            ),
        )

        st.write(
            "**Fecha ancla día 30:**",
            (
                pd.Timestamp(
                    anchor_date
                ).strftime(
                    "%d/%m/%Y"
                )
                if anchor_date is not None
                else "Sin dato"
            ),
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

        if (
            isinstance(
                upstream_meta,
                dict,
            )
            and "Corrientes"
            in upstream_meta
        ):

            corr_meta = upstream_meta.get(
                "Corrientes",
                {},
            )

            if isinstance(
                corr_meta,
                dict,
            ):

                st.write(
                    "**Serie Corrientes:**",
                    corr_meta.get(
                        "series_id",
                        "Sin dato",
                    ),
                )

                st.write(
                    "**Primer registro Corrientes:**",
                    corr_meta.get(
                        "first_date",
                        "Sin dato",
                    ),
                )

                st.write(
                    "**Último registro Corrientes:**",
                    corr_meta.get(
                        "last_date",
                        "Sin dato",
                    ),
                )

                st.write(
                    "**Cantidad Corrientes:**",
                    corr_meta.get(
                        "records",
                        0,
                    ),
                )

        if isinstance(
            metrics,
            dict,
        ):

            rmse = (
                metrics.get(
                    "RMSE",
                    metrics.get(
                        "rmse"
                    ),
                )
            )

            if rmse is not None:

                try:

                    st.metric(
                        "RMSE",
                        f"{float(rmse):.3f} m",
                    )

                except Exception:
                    pass

        if isinstance(
            models,
            dict,
        ):

            st.write(
                "**Lluvia incluida:**",
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
                "**Caudal incluido:**",
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
                "**Aguas arriba:**",
                (
                    "Sí"
                    if models.get(
                        "uses_upstream",
                        False,
                    )
                    else "No"
                ),
            )

    if actualizado is not None:

        st.caption(
            "Actualizado: "
            + actualizado.strftime(
                "%d/%m/%Y %H:%M"
            )
        )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    f"Paraná · San Nicolás · {APP_VERSION}"
)

st.caption(
    "Nivel observado: INA A5 · "
    "Pronóstico experimental"
)

st.warning(
    "Las proyecciones son experimentales y no reemplazan "
    "mediciones, alertas ni pronósticos emitidos por "
    "organismos oficiales."
)
