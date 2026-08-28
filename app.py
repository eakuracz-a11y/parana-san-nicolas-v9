# ============================================================
# PARANÁ · SAN NICOLÁS
# app.py
# V11.9.6 COMPLETO
#
# COMPATIBLE CON:
#
# src/ina.py
# src/upstream.py V11.9.1
# src/exogenous.py V11.9.3
# src/hydrology.py V11.9.6
#
# OBJETIVOS V11.9.6
# ------------------------------------------------------------
# - Tabla completa del corredor Paraná
# - Corrientes -> San Nicolás
# - Tendencia creciente / estable / decreciente
# - Variación última
# - Variación 7 días
# - Comparativa histórica completa
# - Escalas automáticas
# - Demora histórica Corrientes -> San Nicolás
# - Eventos históricos de crecientes
# - Fecha probable de propagación
# - Caudal y lluvia
# - Pronóstico 15 días
# - Tendencia 30 días
# - Escenario 60 días compatible
# ============================================================


import streamlit as st

import pandas as pd
import numpy as np

import plotly.graph_objects as go

from datetime import (
    date,
    timedelta,
    datetime,
)


# ============================================================
# INA
# ============================================================

from src.ina import observed


# ============================================================
# MODELO
# ============================================================

from src.model import (
    train,
    predict,
)


# ============================================================
# EXÓGENAS
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

from src.hydrology import (
    analizar_corrientes_san_nicolas,
)


# ============================================================
# STRESS 60 DÍAS
# ============================================================

try:

    from src.stress_ui import (
        get_stress_scenario,
    )

except Exception:

    get_stress_scenario = None


# ============================================================
# VERSIÓN
# ============================================================

APP_VERSION = "V11.9.6"

FORECAST_DAYS = 15

TREND_DAYS = 30

STRESS_DAYS = 60


# ============================================================
# ESTACIONES
# ============================================================

UPSTREAM_STATIONS = [

    (
        "Corrientes",
        "nivel_corrientes",
    ),

    (
        "Goya",
        "nivel_goya",
    ),

    (
        "La Paz",
        "nivel_la_paz",
    ),

    (
        "Paraná",
        "nivel_parana",
    ),

    (
        "Diamante",
        "nivel_diamante",
    ),

    (
        "Rosario",
        "nivel_rosario",
    ),

    (
        "Villa Constitución",
        "nivel_villa_constitucion",
    ),
]


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(

    page_title=
        "Paraná · San Nicolás",

    page_icon=
        "🌊",

    layout=
        "wide",

    initial_sidebar_state=
        "collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {

        padding-top: 1.2rem;

        padding-bottom: 3rem;

        max-width: 1550px;
    }


    [data-testid="stMetric"] {

        border: 1px solid rgba(120,120,120,0.18);

        background: rgba(120,120,120,0.04);

        border-radius: 12px;

        padding: 10px 12px;
    }


    [data-testid="stMetricValue"] {

        font-size: 1.40rem;
    }


    [data-testid="stMetricLabel"] {

        font-size: 0.82rem;
    }


    div[data-testid="stExpander"] {

        border-radius: 10px;
    }


    h1 {

        font-size: 2rem !important;
    }


    h2 {

        font-size: 1.42rem !important;
    }


    h3 {

        font-size: 1.15rem !important;
    }


    @media (max-width: 700px) {

        .block-container {

            padding-left: 0.65rem;

            padding-right: 0.65rem;

            padding-top: 0.7rem;
        }


        h1 {

            font-size: 1.55rem !important;
        }


        h2 {

            font-size: 1.23rem !important;
        }


        h3 {

            font-size: 1.03rem !important;
        }


        [data-testid="stMetricValue"] {

            font-size: 1.12rem;
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
# FECHAS
# ============================================================

def datetime_naive(
    values,
):

    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
    )


# ============================================================
# NÚMERO SEGURO
# ============================================================

def safe_float(
    value,
    default=np.nan,
):

    try:

        result = float(
            value
        )

        if np.isfinite(
            result
        ):

            return result

    except Exception:

        pass

    return default


# ============================================================
# PREPARAR SAN NICOLÁS
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

    if (
        "datetime"
        not in x.columns
    ):

        return pd.DataFrame()

    x[
        "datetime"
    ] = datetime_naive(
        x[
            "datetime"
        ]
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
# NORMALIZAR FRAME TEMPORAL
# ============================================================

def normalizar_frame(
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

        return df

    x = df.copy()

    x[
        "datetime"
    ] = datetime_naive(
        x[
            "datetime"
        ]
    )

    return x


# ============================================================
# ESCALA AUTOMÁTICA
# ============================================================

def calcular_rango_y(
    *series,
    margen=0.08,
    minimo_margen=0.20,
):

    valores = []

    for serie in series:

        if serie is None:
            continue

        try:

            values = (
                pd.to_numeric(
                    serie,
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

        except Exception:

            continue

        if not values.empty:

            valores.extend(
                values.tolist()
            )

    if not valores:

        return None

    minimo = float(
        np.min(
            valores
        )
    )

    maximo = float(
        np.max(
            valores
        )
    )

    amplitud = max(
        maximo
        - minimo,
        minimo_margen,
    )

    expansion = max(
        amplitud
        * margen,
        0.05,
    )

    inferior = (
        minimo
        - expansion
    )

    superior = (
        maximo
        + expansion
    )

    return [
        inferior,
        superior,
    ]


# ============================================================
# CLASIFICAR TENDENCIA
# ============================================================

def clasificar_tendencia(
    delta,
    threshold=0.03,
):

    value = safe_float(
        delta
    )

    if not np.isfinite(
        value
    ):

        return "⚪ Sin datos"

    if value > threshold:

        return "🟢 ↑ Creciente"

    if value < -threshold:

        return "🔴 ↓ Decreciente"

    return "🟡 → Estable"


# ============================================================
# RESUMEN DE UNA SERIE
# ============================================================

def resumen_serie(
    datetime_series,
    level_series,
):

    temp = pd.DataFrame(
        {
            "datetime":
                datetime_naive(
                    datetime_series
                ),

            "value":
                pd.to_numeric(
                    level_series,
                    errors="coerce",
                ),
        }
    )

    temp = (
        temp
        .dropna()
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
    )

    result = {

        "actual":
            np.nan,

        "anterior":
            np.nan,

        "delta":
            np.nan,

        "delta_7d":
            np.nan,

        "fecha":
            None,

        "estado":
            "⚪ Sin datos",
    }

    if temp.empty:

        return result

    actual = float(
        temp[
            "value"
        ].iloc[-1]
    )

    fecha = temp[
        "datetime"
    ].iloc[-1]

    result[
        "actual"
    ] = actual

    result[
        "fecha"
    ] = fecha

    if len(
        temp
    ) >= 2:

        anterior = float(
            temp[
                "value"
            ].iloc[-2]
        )

        result[
            "anterior"
        ] = anterior

        result[
            "delta"
        ] = (
            actual
            - anterior
        )

    cutoff = (
        fecha
        - pd.Timedelta(
            days=7
        )
    )

    before = temp[
        temp[
            "datetime"
        ]
        <= cutoff
    ]

    if not before.empty:

        reference = float(
            before[
                "value"
            ].iloc[-1]
        )

        result[
            "delta_7d"
        ] = (
            actual
            - reference
        )

    # --------------------------------------------------------
    # ESTADO SE DEFINE MEJOR CON 7 DÍAS
    # --------------------------------------------------------

    if np.isfinite(
        safe_float(
            result[
                "delta_7d"
            ]
        )
    ):

        result[
            "estado"
        ] = clasificar_tendencia(
            result[
                "delta_7d"
            ],
            threshold=0.08,
        )

    else:

        result[
            "estado"
        ] = clasificar_tendencia(
            result[
                "delta"
            ],
            threshold=0.03,
        )

    return result


# ============================================================
# TABLA COMPLETA DEL CORREDOR
# ============================================================

def construir_tabla_corredor(
    upstream_history,
    df_local,
):

    rows = []

    # --------------------------------------------------------
    # UPSTREAM
    # --------------------------------------------------------

    if (
        isinstance(
            upstream_history,
            pd.DataFrame,
        )
        and not upstream_history.empty
        and "datetime"
        in upstream_history.columns
    ):

        up = normalizar_frame(
            upstream_history
        )

    else:

        up = pd.DataFrame()

    for (
        station,
        column,
    ) in UPSTREAM_STATIONS:

        if (
            not up.empty
            and column
            in up.columns
        ):

            result = resumen_serie(
                up[
                    "datetime"
                ],
                up[
                    column
                ],
            )

        else:

            result = {
                "actual":
                    np.nan,

                "anterior":
                    np.nan,

                "delta":
                    np.nan,

                "delta_7d":
                    np.nan,

                "fecha":
                    None,

                "estado":
                    "⚪ Sin datos",
            }

        rows.append(
            {
                "Estación":
                    station,

                "Nivel actual (m)":
                    result[
                        "actual"
                    ],

                "Anterior (m)":
                    result[
                        "anterior"
                    ],

                "Δ última (m)":
                    result[
                        "delta"
                    ],

                "Δ 7 días (m)":
                    result[
                        "delta_7d"
                    ],

                "Estado":
                    result[
                        "estado"
                    ],

                "Fecha":
                    (
                        result[
                            "fecha"
                        ].strftime(
                            "%d/%m/%Y"
                        )
                        if result[
                            "fecha"
                        ]
                        is not None
                        else "Sin dato"
                    ),
            }
        )

    # --------------------------------------------------------
    # SAN NICOLÁS
    # --------------------------------------------------------

    if (
        isinstance(
            df_local,
            pd.DataFrame,
        )
        and not df_local.empty
    ):

        local = resumen_serie(
            df_local[
                "datetime"
            ],
            df_local[
                "nivel"
            ],
        )

    else:

        local = {
            "actual":
                np.nan,

            "anterior":
                np.nan,

            "delta":
                np.nan,

            "delta_7d":
                np.nan,

            "fecha":
                None,

            "estado":
                "⚪ Sin datos",
        }

    rows.append(
        {
            "Estación":
                "San Nicolás",

            "Nivel actual (m)":
                local[
                    "actual"
                ],

            "Anterior (m)":
                local[
                    "anterior"
                ],

            "Δ última (m)":
                local[
                    "delta"
                ],

            "Δ 7 días (m)":
                local[
                    "delta_7d"
                ],

            "Estado":
                local[
                    "estado"
                ],

            "Fecha":
                (
                    local[
                        "fecha"
                    ].strftime(
                        "%d/%m/%Y"
                    )
                    if local[
                        "fecha"
                    ]
                    is not None
                    else "Sin dato"
                ),
        }
    )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# PERFIL DEL CORREDOR
# ============================================================

def construir_perfil_corredor(
    table,
):

    if (
        table is None
        or table.empty
    ):

        return pd.DataFrame()

    x = table[
        [
            "Estación",
            "Nivel actual (m)",
        ]
    ].copy()

    x[
        "Nivel actual (m)"
    ] = pd.to_numeric(
        x[
            "Nivel actual (m)"
        ],
        errors="coerce",
    )

    return x.dropna(
        subset=[
            "Nivel actual (m)"
        ]
    )


# ============================================================
# CAUDAL
# ============================================================

def calcular_tendencia_caudal(
    df,
):

    result = {

        "actual":
            None,

        "delta_3":
            None,

        "delta_7":
            None,

        "pct_7":
            None,

        "estado":
            "Sin datos",
    }

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "caudal_m3s"
        not in df.columns
    ):

        return result

    q = df.copy()

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

        return result

    values = q[
        "caudal_m3s"
    ].to_numpy(
        dtype=float
    )

    current = float(
        values[-1]
    )

    result[
        "actual"
    ] = current

    if len(
        values
    ) >= 4:

        result[
            "delta_3"
        ] = (
            current
            - float(
                values[-4]
            )
        )

    if len(
        values
    ) >= 8:

        q7 = float(
            values[-8]
        )

        result[
            "delta_7"
        ] = (
            current
            - q7
        )

        if q7 != 0:

            result[
                "pct_7"
            ] = (
                result[
                    "delta_7"
                ]
                / q7
                * 100.0
            )

    recent = values[
        -min(
            len(
                values
            ),
            7,
        ):
    ]

    if len(
        recent
    ) >= 3:

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

        threshold = max(
            current
            * 0.002,
            20.0,
        )

        if slope > threshold:

            result[
                "estado"
            ] = "↑ Creciente"

        elif slope < -threshold:

            result[
                "estado"
            ] = "↓ Decreciente"

        else:

            result[
                "estado"
            ] = "→ Estable"

    return result


# ============================================================
# EXTENDER 15 -> 30
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

    f[
        "datetime"
    ] = datetime_naive(
        f[
            "datetime"
        ]
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

    last_date = result[
        "datetime"
    ].iloc[-1]

    last_level = float(
        result[
            "prediction"
        ].iloc[-1]
    )

    recent = (
        result[
            "prediction"
        ]
        .tail(
            5
        )
        .dropna()
    )

    if len(
        recent
    ) >= 3:

        slope = float(
            np.polyfit(
                np.arange(
                    len(
                        recent
                    )
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
            local
        ) >= 3:

            slope = float(
                np.polyfit(
                    np.arange(
                        len(
                            local
                        )
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
            -0.10,
            0.10,
        )
    )

    extra = []

    for day in range(
        16,
        TREND_DAYS
        + 1,
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
                -0.05,
                0.07,
            )
        )

        last_level = (
            last_level
            + daily_change
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
# DÍA 30
# ============================================================

def obtener_ancla_30(
    forecast30,
):

    if (
        forecast30 is None
        or forecast30.empty
    ):

        return (
            None,
            None,
        )

    x = forecast30.copy()

    x[
        "datetime"
    ] = datetime_naive(
        x[
            "datetime"
        ]
    )

    x[
        "prediction"
    ] = pd.to_numeric(
        x[
            "prediction"
        ],
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

    return (

        x[
            "datetime"
        ].iloc[-1],

        float(
            x[
                "prediction"
            ].iloc[-1]
        ),
    )


# ============================================================
# STRESS COMPATIBLE
# ============================================================

def obtener_stress(
    df,
    models,
    exog_history,
    upstream_history,
    anchor_date,
    anchor_level,
):

    if (
        get_stress_scenario
        is None
        or anchor_date
        is None
        or anchor_level
        is None
    ):

        return pd.DataFrame()

    try:

        scenario = (
            get_stress_scenario(
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
                anchor_day=30,
            )
        )

        return normalizar_frame(
            scenario
        )

    except TypeError:

        try:

            old = (
                get_stress_scenario(
                    df=df,
                    models=models,
                    exog_history=
                        exog_history,
                    upstream_history=
                        upstream_history,
                    days=
                        STRESS_DAYS,
                )
            )

        except Exception:

            return pd.DataFrame()

        if (
            old is None
            or old.empty
            or "stress_level"
            not in old.columns
        ):

            return pd.DataFrame()

        old = old.copy()

        old[
            "stress_level"
        ] = pd.to_numeric(
            old[
                "stress_level"
            ],
            errors="coerce",
        )

        if (
            "scenario_day"
            in old.columns
        ):

            candidate = old[
                pd.to_numeric(
                    old[
                        "scenario_day"
                    ],
                    errors="coerce",
                )
                > 30
            ].copy()

        else:

            candidate = (
                old
                .iloc[
                    30:60
                ]
                .copy()
            )

        if candidate.empty:

            candidate = (
                old
                .iloc[
                    30:
                ]
                .copy()
            )

        candidate = (
            candidate
            .reset_index(
                drop=True
            )
        )

        first_valid = (
            candidate[
                "stress_level"
            ]
            .dropna()
        )

        if first_valid.empty:

            return pd.DataFrame()

        offset = (
            anchor_level
            - float(
                first_valid.iloc[0]
            )
        )

        candidate[
            "stress_level"
        ] = (
            candidate[
                "stress_level"
            ]
            + offset
        )

        candidate[
            "datetime"
        ] = pd.date_range(
            start=(
                anchor_date
                + pd.Timedelta(
                    days=1
                )
            ),
            periods=len(
                candidate
            ),
            freq="D",
        )

        candidate[
            "scenario_day"
        ] = np.arange(
            31,
            31
            + len(
                candidate
            ),
        )

        return candidate


# ============================================================
# FECHAS TEXTO
# ============================================================

def fecha_texto(
    value,
):

    if value is None:

        return "Sin dato"

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        dt
    ):

        return "Sin dato"

    return dt.strftime(
        "%d/%m/%Y"
    )


# ============================================================
# CABECERA
# ============================================================

st.title(
    "🌊 PARANÁ · SAN NICOLÁS"
)

st.caption(
    f"{APP_VERSION} · Monitoreo y propagación hidrológica"
)

st.markdown(
    """
    Análisis del río Paraná como sistema
    **Corrientes → San Nicolás** utilizando niveles hidrométricos,
    evolución aguas arriba, caudal, precipitación y antecedentes
    históricos de crecientes.
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

    value=
        default_start,

    format=
        "DD/MM/YYYY",
)

hasta = st.sidebar.date_input(

    "Hasta",

    value=
        today,

    format=
        "DD/MM/YYYY",
)

actualizar = st.sidebar.button(

    "🔄 Actualizar modelo",

    type=
        "primary",

    use_container_width=True,
)


st.sidebar.divider()

st.sidebar.write(
    "**Pronóstico principal:** 15 días"
)

st.sidebar.write(
    "**Tendencia:** 30 días"
)

st.sidebar.write(
    "**Escenario:** 60 días"
)

st.sidebar.caption(
    "Los gráficos utilizan escala automática."
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

        proceso_ok = True

        # ====================================================
        # SAN NICOLÁS
        # ====================================================

        with st.spinner(
            "Consultando nivel de San Nicolás..."
        ):

            df_raw, error_ina = observed(
                inicio,
                fin,
            )

        if error_ina:

            st.error(
                error_ina
            )

            proceso_ok = False

        else:

            df = preparar_datos(
                df_raw
            )

            if df.empty:

                st.error(
                    "No se obtuvieron observaciones válidas "
                    "de San Nicolás."
                )

                proceso_ok = False


        # ====================================================
        # CONTINUAR
        # ====================================================

        if proceso_ok:

            # ================================================
            # LLUVIA + CAUDAL
            # ================================================

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
                            TREND_DAYS,
                        )
                    )

                    exog_history = (
                        normalizar_frame(
                            exog_history
                        )
                    )

                    exog_future = (
                        normalizar_frame(
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
                            str(
                                exc
                            )
                    }

                    st.warning(
                        "No se pudieron recuperar completamente "
                        f"lluvia y caudal: {exc}"
                    )


            # ================================================
            # UPSTREAM
            # ================================================

            with st.spinner(
                "Consultando niveles aguas arriba..."
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

                    upstream_history = (
                        normalizar_frame(
                            upstream_history
                        )
                    )

                except Exception as exc:

                    upstream_history = (
                        pd.DataFrame()
                    )

                    upstream_meta = {}

                    st.warning(
                        "No fue posible recuperar completamente "
                        f"las estaciones aguas arriba: {exc}"
                    )


            # ================================================
            # HIDROLOGÍA
            # ================================================

            hydrology = {}

            with st.spinner(
                "Analizando propagación Corrientes → San Nicolás..."
            ):

                try:

                    hydrology = (
                        analizar_corrientes_san_nicolas(
                            san_nicolas=
                                df,

                            upstream_history=
                                upstream_history,

                            exog_history=
                                exog_history,

                            max_lag=20,

                            usar_historial_completo=True,
                        )
                    )

                except TypeError:

                    try:

                        hydrology = (
                            analizar_corrientes_san_nicolas(
                                san_nicolas=
                                    df,

                                upstream_history=
                                    upstream_history,

                                exog_history=
                                    exog_history,
                            )
                        )

                    except Exception as exc:

                        hydrology = {
                            "error":
                                str(
                                    exc
                                )
                        }

                except Exception as exc:

                    hydrology = {
                        "error":
                            str(
                                exc
                            )
                    }


            # ================================================
            # MODELO
            # ================================================

            models = {}

            metrics = {}

            forecast15 = pd.DataFrame()

            forecast30 = pd.DataFrame()

            stress60 = pd.DataFrame()

            model_error = None

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
                            train_result[
                                0
                            ]
                        )

                        if (
                            len(
                                train_result
                            )
                            > 1
                        ):

                            metrics = (
                                train_result[
                                    1
                                ]
                            )

                    elif isinstance(
                        train_result,
                        dict,
                    ):

                        models = (
                            train_result
                        )

                    else:

                        raise ValueError(
                            "train() no devolvió un modelo válido."
                        )


                    # ========================================
                    # PRONÓSTICO
                    # ========================================

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


                    if (
                        forecast15 is None
                        or not isinstance(
                            forecast15,
                            pd.DataFrame,
                        )
                        or forecast15.empty
                    ):

                        raise ValueError(
                            "El modelo no generó pronóstico."
                        )


                    forecast15 = (
                        normalizar_frame(
                            forecast15
                        )
                    )


                    # ========================================
                    # 30 DÍAS
                    # ========================================

                    forecast30 = (
                        extender_pronostico_30(
                            forecast15,
                            df,
                        )
                    )


                    # ========================================
                    # 60 DÍAS
                    # ========================================

                    (
                        anchor_date,
                        anchor_level,
                    ) = obtener_ancla_30(
                        forecast30
                    )

                    stress60 = obtener_stress(

                        df=
                            df,

                        models=
                            models,

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

                    model_error = str(
                        exc
                    )

                    forecast15 = (
                        pd.DataFrame()
                    )

                    forecast30 = (
                        pd.DataFrame()
                    )

                    stress60 = (
                        pd.DataFrame()
                    )


            # ================================================
            # SESIÓN
            # ================================================

            st.session_state[
                "datos"
            ] = df

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
                "forecast15"
            ] = forecast15

            st.session_state[
                "forecast30"
            ] = forecast30

            st.session_state[
                "stress60"
            ] = stress60

            st.session_state[
                "model_error"
            ] = model_error

            st.session_state[
                "fecha_inicio"
            ] = inicio

            st.session_state[
                "fecha_fin"
            ] = fin

            st.session_state[
                "actualizado"
            ] = datetime.now()


            # ================================================
            # MENSAJE CORRECTO
            # ================================================

            if model_error is None:

                st.success(
                    "✅ Datos y modelo actualizados correctamente."
                )

            else:

                st.warning(
                    "Los datos fueron actualizados, pero el pronóstico "
                    f"no pudo generarse: {model_error}"
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

    df = st.session_state[
        "datos"
    ]

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

    hydrology = (
        st.session_state.get(
            "hydrology",
            {},
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

    model_error = (
        st.session_state.get(
            "model_error"
        )
    )


    # ========================================================
    # RESUMEN SAN NICOLÁS
    # ========================================================

    local_summary = (
        resumen_serie(
            df[
                "datetime"
            ],
            df[
                "nivel"
            ],
        )
    )

    nivel_actual = (
        local_summary[
            "actual"
        ]
    )

    delta_local = (
        local_summary[
            "delta"
        ]
    )

    delta_local_7 = (
        local_summary[
            "delta_7d"
        ]
    )

    estado_local = (
        local_summary[
            "estado"
        ]
    )


    # ========================================================
    # CAUDAL
    # ========================================================

    flow_summary = (
        calcular_tendencia_caudal(
            exog_history
        )
    )


    # ========================================================
    # DEMORA
    # ========================================================

    demora = np.nan

    demora_min = np.nan

    demora_max = np.nan

    impacto = None

    impacto_desde = None

    impacto_hasta = None

    respuesta_probable = np.nan

    current_estimate = {}

    if isinstance(
        hydrology,
        dict,
    ):

        demora = safe_float(
            hydrology.get(
                "demora_probable_dias"
            )
        )

        demora_min = safe_float(
            hydrology.get(
                "demora_min_dias"
            )
        )

        demora_max = safe_float(
            hydrology.get(
                "demora_max_dias"
            )
        )

        respuesta_probable = safe_float(
            hydrology.get(
                "respuesta_probable_m"
            )
        )

        impacto = hydrology.get(
            "fecha_impacto_probable"
        )

        impacto_desde = hydrology.get(
            "fecha_impacto_desde"
        )

        impacto_hasta = hydrology.get(
            "fecha_impacto_hasta"
        )

        current_estimate = (
            hydrology.get(
                "current_estimate",
                {},
            )
        )


    # ========================================================
    # TARJETAS PRINCIPALES
    # ========================================================

    st.subheader(
        "📍 Estado actual"
    )

    a1, a2, a3, a4 = st.columns(
        4
    )

    a1.metric(

        "San Nicolás",

        (
            f"{nivel_actual:.2f} m"
            if np.isfinite(
                safe_float(
                    nivel_actual
                )
            )
            else "Sin dato"
        ),

        (
            f"{delta_local:+.2f} m"
            if np.isfinite(
                safe_float(
                    delta_local
                )
            )
            else None
        ),
    )


    a2.metric(

        "Tendencia 7 días",

        estado_local,

        (
            f"{delta_local_7:+.2f} m"
            if np.isfinite(
                safe_float(
                    delta_local_7
                )
            )
            else None
        ),
    )


    a3.metric(

        "Caudal",

        (
            f"{flow_summary['actual']:,.0f} m³/s"
            if flow_summary[
                "actual"
            ]
            is not None
            else "Sin dato"
        ),

        (
            flow_summary[
                "estado"
            ]
            if flow_summary[
                "actual"
            ]
            is not None
            else None
        ),
    )


    a4.metric(

        "Demora Corrientes → SN",

        (
            f"{demora:.0f} días"
            if np.isfinite(
                demora
            )
            else "Calculando"
        ),

        (
            (
                f"{demora_min:.0f}–"
                f"{demora_max:.0f} días"
            )
            if (
                np.isfinite(
                    demora_min
                )
                and np.isfinite(
                    demora_max
                )
            )
            else None
        ),
    )


    # ========================================================
    # PRONÓSTICO PRINCIPAL
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Nivel de San Nicolás · observado y proyección"
    )

    forecast_fig = go.Figure()


    # --------------------------------------------------------
    # OBSERVADO
    # --------------------------------------------------------

    observed_plot = (
        df
        .tail(
            180
        )
        .copy()
    )

    forecast_fig.add_trace(
        go.Scatter(

            x=
                observed_plot[
                    "datetime"
                ],

            y=
                observed_plot[
                    "nivel"
                ],

            mode=
                "lines",

            name=
                "Observado",

            line=dict(
                color="#2563EB",
                width=2.6,
            ),
        )
    )


    level_series_for_range = [
        observed_plot[
            "nivel"
        ]
    ]


    # --------------------------------------------------------
    # 1-15
    # --------------------------------------------------------

    if (
        isinstance(
            forecast15,
            pd.DataFrame,
        )
        and not forecast15.empty
        and "prediction"
        in forecast15.columns
    ):

        f15 = (
            forecast15
            .head(
                15
            )
            .copy()
        )

        connection = (
            pd.DataFrame(
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
        )

        f15_plot = pd.concat(
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

        forecast_fig.add_trace(
            go.Scatter(

                x=
                    f15_plot[
                        "datetime"
                    ],

                y=
                    f15_plot[
                        "prediction"
                    ],

                mode=
                    "lines+markers",

                name=
                    "Pronóstico 1–15 días",

                line=dict(
                    color="#F59E0B",
                    width=3,
                ),

                marker=dict(
                    size=5,
                ),
            )
        )

        level_series_for_range.append(
            f15_plot[
                "prediction"
            ]
        )


    # --------------------------------------------------------
    # 16-30
    # --------------------------------------------------------

    if (
        isinstance(
            forecast30,
            pd.DataFrame,
        )
        and len(
            forecast30
        ) > 15
    ):

        f30_plot = (
            forecast30
            .iloc[
                14:
            ]
            .copy()
        )

        forecast_fig.add_trace(
            go.Scatter(

                x=
                    f30_plot[
                        "datetime"
                    ],

                y=
                    f30_plot[
                        "prediction"
                    ],

                mode=
                    "lines+markers",

                name=
                    "Tendencia 16–30 días",

                line=dict(
                    color="#16A34A",
                    width=2.8,
                ),

                marker=dict(
                    size=4,
                ),
            )
        )

        level_series_for_range.append(
            f30_plot[
                "prediction"
            ]
        )


    # --------------------------------------------------------
    # 31-60
    # --------------------------------------------------------

    if (
        isinstance(
            stress60,
            pd.DataFrame,
        )
        and not stress60.empty
        and "stress_level"
        in stress60.columns
    ):

        forecast_fig.add_trace(
            go.Scatter(

                x=
                    stress60[
                        "datetime"
                    ],

                y=
                    stress60[
                        "stress_level"
                    ],

                mode=
                    "lines",

                name=
                    "Escenario 31–60 días",

                line=dict(
                    color="#DC2626",
                    width=2.7,
                    dash="dash",
                ),
            )
        )

        level_series_for_range.append(
            stress60[
                "stress_level"
            ]
        )


    range_y = calcular_rango_y(
        *level_series_for_range
    )

    forecast_fig.update_layout(

        height=500,

        hovermode=
            "x unified",

        margin=dict(
            l=10,
            r=10,
            t=20,
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

    forecast_fig.update_yaxes(

        title_text=
            "Nivel (m)",

        range=
            range_y,
    )

    forecast_fig.update_xaxes(
        tickformat=
            "%d/%m/%Y"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True,
    )


    if model_error:

        st.warning(
            "El modelo de pronóstico presentó un problema: "
            + str(
                model_error
            )
        )


    # ========================================================
    # CORREDOR COMPLETO
    # ========================================================

    st.divider()

    st.subheader(
        "🌊 Estado del corredor aguas arriba → San Nicolás"
    )

    corridor_table = (
        construir_tabla_corredor(
            upstream_history=
                upstream_history,
            df_local=
                df,
        )
    )


    # ========================================================
    # TABLA
    # ========================================================

    display_table = (
        corridor_table.copy()
    )

    for column in [
        "Nivel actual (m)",
        "Anterior (m)",
        "Δ última (m)",
        "Δ 7 días (m)",
    ]:

        display_table[
            column
        ] = display_table[
            column
        ].apply(
            lambda value:
                (
                    f"{value:.2f}"
                    if pd.notna(
                        value
                    )
                    else "—"
                )
        )


    st.dataframe(

        display_table,

        use_container_width=True,

        hide_index=True,

        column_config={

            "Estación":
                st.column_config.TextColumn(
                    "Estación",
                    width="medium",
                ),

            "Nivel actual (m)":
                st.column_config.TextColumn(
                    "Nivel actual",
                    width="small",
                ),

            "Anterior (m)":
                st.column_config.TextColumn(
                    "Anterior",
                    width="small",
                ),

            "Δ última (m)":
                st.column_config.TextColumn(
                    "Δ última",
                    width="small",
                ),

            "Δ 7 días (m)":
                st.column_config.TextColumn(
                    "Δ 7 días",
                    width="small",
                ),

            "Estado":
                st.column_config.TextColumn(
                    "Estado",
                    width="medium",
                ),

            "Fecha":
                st.column_config.TextColumn(
                    "Última medición",
                    width="medium",
                ),
        },
    )

    st.caption(
        "El estado creciente/decreciente se determina "
        "prioritariamente con la variación acumulada de 7 días."
    )


    # ========================================================
    # PERFIL ACTUAL
    # ========================================================

    profile = (
        construir_perfil_corredor(
            corridor_table
        )
    )

    if not profile.empty:

        st.subheader(
            "🧭 Perfil actual del corredor"
        )

        profile_fig = go.Figure()

        profile_fig.add_trace(
            go.Scatter(

                x=
                    profile[
                        "Estación"
                    ],

                y=
                    profile[
                        "Nivel actual (m)"
                    ],

                mode=
                    "lines+markers+text",

                text=[
                    f"{value:.2f} m"
                    for value in profile[
                        "Nivel actual (m)"
                    ]
                ],

                textposition=
                    "top center",

                marker=dict(
                    size=9,
                ),

                line=dict(
                    width=2.5,
                ),

                name=
                    "Nivel actual",
            )
        )

        profile_range = calcular_rango_y(
            profile[
                "Nivel actual (m)"
            ],
            margen=0.12,
        )

        profile_fig.update_layout(

            height=420,

            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),

            showlegend=False,
        )

        profile_fig.update_yaxes(

            title_text=
                "Nivel hidrométrico (m)",

            range=
                profile_range,
        )

        st.plotly_chart(
            profile_fig,
            use_container_width=True,
        )


    # ========================================================
    # PROPAGACIÓN CORRIENTES -> SAN NICOLÁS
    # ========================================================

    st.divider()

    st.subheader(
        "⏱️ Propagación Corrientes → San Nicolás"
    )


    # --------------------------------------------------------
    # ESTADO ACTUAL DE CORRIENTES
    # --------------------------------------------------------

    estado_corrientes = {}

    if isinstance(
        current_estimate,
        dict,
    ):

        estado_corrientes = (
            current_estimate.get(
                "estado_corrientes",
                {},
            )
        )

    nivel_corrientes_actual = (
        safe_float(
            estado_corrientes.get(
                "nivel_corrientes"
            )
        )
        if isinstance(
            estado_corrientes,
            dict,
        )
        else np.nan
    )

    variacion_corrientes_7 = (
        safe_float(
            estado_corrientes.get(
                "variacion_7d"
            )
        )
        if isinstance(
            estado_corrientes,
            dict,
        )
        else np.nan
    )

    estado_corrientes_texto = (
        estado_corrientes.get(
            "estado",
            "Sin datos",
        )
        if isinstance(
            estado_corrientes,
            dict,
        )
        else "Sin datos"
    )


    p1, p2, p3, p4 = st.columns(
        4
    )


    p1.metric(

        "Corrientes actual",

        (
            f"{nivel_corrientes_actual:.2f} m"
            if np.isfinite(
                nivel_corrientes_actual
            )
            else "Sin dato"
        ),
    )


    p2.metric(

        "Variación Corrientes 7 días",

        (
            f"{variacion_corrientes_7:+.2f} m"
            if np.isfinite(
                variacion_corrientes_7
            )
            else "Sin dato"
        ),

        estado_corrientes_texto,
    )


    p3.metric(

        "Demora más probable",

        (
            f"{demora:.0f} días"
            if np.isfinite(
                demora
            )
            else "Sin dato"
        ),

        (
            (
                f"rango {demora_min:.0f}–"
                f"{demora_max:.0f} días"
            )
            if (
                np.isfinite(
                    demora_min
                )
                and np.isfinite(
                    demora_max
                )
            )
            else None
        ),
    )


    p4.metric(

        "Impacto probable",

        fecha_texto(
            impacto
        ),

        (
            (
                fecha_texto(
                    impacto_desde
                )
                + " a "
                + fecha_texto(
                    impacto_hasta
                )
            )
            if (
                impacto_desde
                is not None
                and impacto_hasta
                is not None
            )
            else None
        ),
    )


    if np.isfinite(
        respuesta_probable
    ):

        st.info(
            "Para eventos históricos similares, la respuesta "
            "mediana observada en San Nicolás fue de aproximadamente "
            f"**{respuesta_probable:+.2f} m** respecto del nivel base "
            "del evento."
        )


    # ========================================================
    # ESTADÍSTICAS HISTÓRICAS
    # ========================================================

    statistics = {}

    if isinstance(
        hydrology,
        dict,
    ):

        statistics = (
            hydrology.get(
                "statistics",
                {},
            )
        )

    if isinstance(
        statistics,
        dict,
    ):

        eventos_count = statistics.get(
            "eventos",
            0,
        )

        lag_median = safe_float(
            statistics.get(
                "lag_mediana_dias"
            )
        )

        lag_average = safe_float(
            statistics.get(
                "lag_promedio_dias"
            )
        )

        corr_max = safe_float(
            statistics.get(
                "correlacion_maximos"
            )
        )


        s1, s2, s3, s4 = st.columns(
            4
        )


        s1.metric(
            "Eventos históricos",
            int(
                eventos_count
            ),
        )


        s2.metric(

            "Demora mediana",

            (
                f"{lag_median:.1f} días"
                if np.isfinite(
                    lag_median
                )
                else "Sin dato"
            ),
        )


        s3.metric(

            "Demora promedio",

            (
                f"{lag_average:.1f} días"
                if np.isfinite(
                    lag_average
                )
                else "Sin dato"
            ),
        )


        s4.metric(

            "Correlación máximos",

            (
                f"{corr_max:.2f}"
                if np.isfinite(
                    corr_max
                )
                else "Sin dato"
            ),
        )


    # ========================================================
    # HISTORIA COMPLETA CORRIENTES / SAN NICOLÁS
    # ========================================================

    st.divider()

    st.subheader(
        "🔗 Comparativa histórica completa Corrientes vs San Nicolás"
    )

    historical_features = pd.DataFrame()

    if isinstance(
        hydrology,
        dict,
    ):

        candidate = hydrology.get(
            "features"
        )

        if isinstance(
            candidate,
            pd.DataFrame,
        ):

            historical_features = (
                candidate.copy()
            )


    if (
        not historical_features.empty
        and "datetime"
        in historical_features.columns
        and "nivel"
        in historical_features.columns
        and "nivel_corrientes"
        in historical_features.columns
    ):

        hist = historical_features[
            [
                "datetime",
                "nivel_corrientes",
                "nivel",
            ]
        ].copy()

        hist[
            "datetime"
        ] = datetime_naive(
            hist[
                "datetime"
            ]
        )

        hist[
            "nivel_corrientes"
        ] = pd.to_numeric(
            hist[
                "nivel_corrientes"
            ],
            errors="coerce",
        )

        hist[
            "nivel"
        ] = pd.to_numeric(
            hist[
                "nivel"
            ],
            errors="coerce",
        )

        hist = hist.dropna(
            subset=[
                "datetime"
            ]
        )


        if not hist.empty:

            fecha_min = hist[
                "datetime"
            ].min()

            fecha_max = hist[
                "datetime"
            ].max()

            st.caption(
                "Historial coincidente utilizado: "
                f"**{fecha_texto(fecha_min)} → {fecha_texto(fecha_max)}**"
            )


            vista_hist = st.radio(

                "Visualización histórica",

                [
                    "Todo el historial",
                    "20 años",
                    "10 años",
                    "5 años",
                    "1 año",
                ],

                horizontal=True,

                index=0,
            )


            hist_view = hist.copy()


            if vista_hist != "Todo el historial":

                years = int(
                    vista_hist.split(
                        " "
                    )[0]
                )

                cutoff = (
                    fecha_max
                    - pd.DateOffset(
                        years=years
                    )
                )

                hist_view = hist_view[
                    hist_view[
                        "datetime"
                    ]
                    >= cutoff
                ].copy()


            # ------------------------------------------------
            # TODO HISTORIAL / 20 AÑOS -> PROMEDIO MENSUAL
            # SOLO PARA VISUALIZACIÓN
            # ------------------------------------------------

            if vista_hist in [
                "Todo el historial",
                "20 años",
            ]:

                monthly = (
                    hist_view
                    .set_index(
                        "datetime"
                    )[
                        [
                            "nivel_corrientes",
                            "nivel",
                        ]
                    ]
                    .resample(
                        "MS"
                    )
                    .mean()
                    .reset_index()
                )

                plot_hist = monthly

                subtitle = (
                    "Promedios mensuales para facilitar la lectura. "
                    "El algoritmo conserva la resolución diaria."
                )

            else:

                plot_hist = hist_view

                subtitle = (
                    "Datos diarios coincidentes."
                )


            st.caption(
                subtitle
            )


            historical_fig = go.Figure()


            historical_fig.add_trace(
                go.Scatter(

                    x=
                        plot_hist[
                            "datetime"
                        ],

                    y=
                        plot_hist[
                            "nivel_corrientes"
                        ],

                    mode=
                        "lines",

                    name=
                        "Corrientes",

                    line=dict(
                        color="#8B5CF6",
                        width=2,
                    ),
                )
            )


            historical_fig.add_trace(
                go.Scatter(

                    x=
                        plot_hist[
                            "datetime"
                        ],

                    y=
                        plot_hist[
                            "nivel"
                        ],

                    mode=
                        "lines",

                    name=
                        "San Nicolás",

                    line=dict(
                        color="#2563EB",
                        width=2,
                    ),
                )
            )


            historical_range = calcular_rango_y(

                plot_hist[
                    "nivel_corrientes"
                ],

                plot_hist[
                    "nivel"
                ],
            )


            historical_fig.update_layout(

                height=500,

                hovermode=
                    "x unified",

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
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


            historical_fig.update_yaxes(

                title_text=
                    "Nivel (m)",

                range=
                    historical_range,
            )


            historical_fig.update_xaxes(
                title_text=
                    "Fecha"
            )


            st.plotly_chart(
                historical_fig,
                use_container_width=True,
            )


    else:

        st.warning(
            "Todavía no existe suficiente historial coincidente "
            "entre Corrientes y San Nicolás para construir "
            "la comparación completa."
        )


    # ========================================================
    # EVENTOS HISTÓRICOS
    # ========================================================

    events = pd.DataFrame()

    if isinstance(
        hydrology,
        dict,
    ):

        candidate_events = (
            hydrology.get(
                "events"
            )
        )

        if isinstance(
            candidate_events,
            pd.DataFrame,
        ):

            events = (
                candidate_events.copy()
            )


    if not events.empty:

        st.subheader(
            "🌊 Eventos históricos Corrientes → San Nicolás"
        )


        # ====================================================
        # SCATTER MÁXIMOS
        # ====================================================

        if (
            "max_corrientes_m"
            in events.columns
            and "max_san_nicolas_m"
            in events.columns
        ):

            event_scatter = go.Figure()


            hover_text = []

            for _, row in events.iterrows():

                hover_text.append(

                    "Corrientes: "
                    + fecha_texto(
                        row.get(
                            "fecha_max_corrientes"
                        )
                    )
                    + "<br>San Nicolás: "
                    + fecha_texto(
                        row.get(
                            "fecha_max_san_nicolas"
                        )
                    )
                    + "<br>Demora: "
                    + str(
                        int(
                            safe_float(
                                row.get(
                                    "lag_real_dias"
                                ),
                                0,
                            )
                        )
                    )
                    + " días"
                )


            event_scatter.add_trace(
                go.Scatter(

                    x=
                        events[
                            "max_corrientes_m"
                        ],

                    y=
                        events[
                            "max_san_nicolas_m"
                        ],

                    mode=
                        "markers",

                    marker=dict(
                        size=8,
                        opacity=0.70,
                    ),

                    text=
                        hover_text,

                    hovertemplate=
                        "%{text}"
                        "<br>Máx Corrientes: %{x:.2f} m"
                        "<br>Máx San Nicolás: %{y:.2f} m"
                        "<extra></extra>",

                    name=
                        "Eventos históricos",
                )
            )


            scatter_x = pd.to_numeric(
                events[
                    "max_corrientes_m"
                ],
                errors="coerce",
            )

            scatter_y = pd.to_numeric(
                events[
                    "max_san_nicolas_m"
                ],
                errors="coerce",
            )


            valid_scatter = pd.DataFrame(
                {
                    "x":
                        scatter_x,

                    "y":
                        scatter_y,
                }
            ).dropna()


            if len(
                valid_scatter
            ) >= 3:

                slope, intercept = np.polyfit(

                    valid_scatter[
                        "x"
                    ],

                    valid_scatter[
                        "y"
                    ],

                    1,
                )

                x_line = np.linspace(

                    valid_scatter[
                        "x"
                    ].min(),

                    valid_scatter[
                        "x"
                    ].max(),

                    100,
                )

                y_line = (
                    slope
                    * x_line
                    + intercept
                )


                event_scatter.add_trace(
                    go.Scatter(

                        x=
                            x_line,

                        y=
                            y_line,

                        mode=
                            "lines",

                        name=
                            "Relación histórica",

                        line=dict(
                            dash="dash",
                            width=2,
                        ),
                    )
                )


            scatter_range_y = calcular_rango_y(
                events[
                    "max_san_nicolas_m"
                ]
            )


            event_scatter.update_layout(

                height=450,

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),

                xaxis_title=
                    "Máximo Corrientes (m)",

                yaxis_title=
                    "Máximo posterior San Nicolás (m)",

                hovermode=
                    "closest",
            )


            event_scatter.update_yaxes(
                range=
                    scatter_range_y
            )


            st.plotly_chart(
                event_scatter,
                use_container_width=True,
            )


        # ====================================================
        # DEMORA POR EVENTO
        # ====================================================

        if (
            "fecha_max_corrientes"
            in events.columns
            and "lag_real_dias"
            in events.columns
        ):

            delay_fig = go.Figure()


            delay_fig.add_trace(
                go.Scatter(

                    x=
                        events[
                            "fecha_max_corrientes"
                        ],

                    y=
                        events[
                            "lag_real_dias"
                        ],

                    mode=
                        "lines+markers",

                    name=
                        "Demora histórica",

                    marker=dict(
                        size=7,
                    ),

                    line=dict(
                        width=1.5,
                    ),
                )
            )


            if np.isfinite(
                safe_float(
                    statistics.get(
                        "lag_mediana_dias"
                    )
                )
            ):

                median_delay = safe_float(
                    statistics.get(
                        "lag_mediana_dias"
                    )
                )

                delay_fig.add_hline(

                    y=
                        median_delay,

                    line_dash=
                        "dash",

                    annotation_text=
                        (
                            "Mediana "
                            f"{median_delay:.1f} días"
                        ),
                )


            delay_range = calcular_rango_y(
                events[
                    "lag_real_dias"
                ],
                margen=0.15,
            )


            delay_fig.update_layout(

                height=400,

                xaxis_title=
                    "Fecha máximo en Corrientes",

                yaxis_title=
                    "Demora hasta máximo en San Nicolás (días)",

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
            )


            delay_fig.update_yaxes(
                range=
                    delay_range
            )


            st.plotly_chart(
                delay_fig,
                use_container_width=True,
            )


        # ====================================================
        # TABLA DE EVENTOS
        # ====================================================

        with st.expander(
            "📋 Ver eventos históricos analizados"
        ):

            event_table = events.copy()


            rename_map = {

                "fecha_max_corrientes":
                    "Fecha Corrientes",

                "max_corrientes_m":
                    "Máx. Corrientes (m)",

                "crecida_corrientes_m":
                    "Crecida Corrientes (m)",

                "velocidad_corrientes_m_dia":
                    "Velocidad (m/día)",

                "fecha_max_san_nicolas":
                    "Fecha San Nicolás",

                "max_san_nicolas_m":
                    "Máx. San Nicolás (m)",

                "respuesta_san_nicolas_m":
                    "Respuesta SN (m)",

                "lag_real_dias":
                    "Demora (días)",

                "caudal_medio_m3s":
                    "Caudal medio (m³/s)",

                "lluvia_previa_mm":
                    "Lluvia previa (mm)",
            }


            available_columns = [
                c
                for c
                in rename_map.keys()
                if c
                in event_table.columns
            ]


            event_table = (
                event_table[
                    available_columns
                ]
                .rename(
                    columns=
                        rename_map
                )
            )


            for date_column in [
                "Fecha Corrientes",
                "Fecha San Nicolás",
            ]:

                if (
                    date_column
                    in event_table.columns
                ):

                    event_table[
                        date_column
                    ] = pd.to_datetime(
                        event_table[
                            date_column
                        ],
                        errors="coerce",
                    ).dt.strftime(
                        "%d/%m/%Y"
                    )


            st.dataframe(

                event_table,

                use_container_width=True,

                hide_index=True,
            )


    # ========================================================
    # EVENTOS SIMILARES ACTUALES
    # ========================================================

    similar_events = pd.DataFrame()

    if isinstance(
        hydrology,
        dict,
    ):

        candidate_similar = (
            hydrology.get(
                "similar_events"
            )
        )

        if isinstance(
            candidate_similar,
            pd.DataFrame,
        ):

            similar_events = (
                candidate_similar.copy()
            )


    if not similar_events.empty:

        with st.expander(
            "🔎 Eventos históricos más similares a la situación actual"
        ):

            similar_table = (
                similar_events.copy()
            )


            columns = {

                "fecha_max_corrientes":
                    "Fecha Corrientes",

                "max_corrientes_m":
                    "Nivel Corrientes",

                "crecida_corrientes_m":
                    "Crecida",

                "fecha_max_san_nicolas":
                    "Fecha San Nicolás",

                "max_san_nicolas_m":
                    "Nivel San Nicolás",

                "respuesta_san_nicolas_m":
                    "Respuesta SN",

                "lag_real_dias":
                    "Demora",

                "similarity_score":
                    "Distancia estadística",
            }


            valid = [
                col
                for col
                in columns.keys()
                if col
                in similar_table.columns
            ]


            similar_table = (
                similar_table[
                    valid
                ]
                .rename(
                    columns=
                        columns
                )
            )


            for col in [
                "Fecha Corrientes",
                "Fecha San Nicolás",
            ]:

                if (
                    col
                    in similar_table.columns
                ):

                    similar_table[
                        col
                    ] = pd.to_datetime(
                        similar_table[
                            col
                        ],
                        errors="coerce",
                    ).dt.strftime(
                        "%d/%m/%Y"
                    )


            st.dataframe(

                similar_table,

                use_container_width=True,

                hide_index=True,
            )


    # ========================================================
    # CAUDAL
    # ========================================================

    st.divider()

    st.subheader(
        "💧 Caudal"
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


        q1, q2, q3, q4 = st.columns(
            4
        )


        q1.metric(

            "Caudal actual",

            (
                f"{flow_summary['actual']:,.0f} m³/s"
                if flow_summary[
                    "actual"
                ]
                is not None
                else "Sin dato"
            ),
        )


        q2.metric(

            "Variación 3 días",

            (
                f"{flow_summary['delta_3']:+,.0f} m³/s"
                if flow_summary[
                    "delta_3"
                ]
                is not None
                else "Sin dato"
            ),
        )


        q3_text = "Sin dato"

        if (
            flow_summary[
                "delta_7"
            ]
            is not None
        ):

            q3_text = (
                f"{flow_summary['delta_7']:+,.0f} m³/s"
            )

            if (
                flow_summary[
                    "pct_7"
                ]
                is not None
            ):

                q3_text += (
                    f" ({flow_summary['pct_7']:+.1f}%)"
                )


        q3.metric(
            "Variación 7 días",
            q3_text,
        )


        q4.metric(
            "Tendencia",
            flow_summary[
                "estado"
            ],
        )


        flow_fig = go.Figure()


        flow_fig.add_trace(
            go.Scatter(

                x=
                    q_hist[
                        "datetime"
                    ],

                y=
                    q_hist[
                        "caudal_m3s"
                    ],

                mode=
                    "lines",

                name=
                    "Caudal observado",

                line=dict(
                    color="#0284C7",
                    width=2.5,
                ),
            )
        )


        flow_range_series = [
            q_hist[
                "caudal_m3s"
            ]
        ]


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


            if not q_future.empty:

                connection = (
                    pd.DataFrame(
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


                flow_fig.add_trace(
                    go.Scatter(

                        x=
                            q_future_plot[
                                "datetime"
                            ],

                        y=
                            q_future_plot[
                                "caudal_m3s"
                            ],

                        mode=
                            "lines+markers",

                        name=
                            "Proyección de caudal",

                        line=dict(
                            color="#F59E0B",
                            width=2.5,
                            dash="dash",
                        ),
                    )
                )


                flow_range_series.append(
                    q_future_plot[
                        "caudal_m3s"
                    ]
                )


        flow_range = calcular_rango_y(
            *flow_range_series,
            margen=0.10,
        )


        flow_fig.update_layout(

            height=400,

            hovermode=
                "x unified",

            yaxis_title=
                "Caudal (m³/s)",

            margin=dict(
                l=10,
                r=10,
                t=20,
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


        flow_fig.update_yaxes(
            range=
                flow_range
        )


        st.plotly_chart(
            flow_fig,
            use_container_width=True,
        )


        if isinstance(
            exog_meta,
            dict,
        ):

            flow_info = (
                exog_meta.get(
                    "caudal_series"
                )
            )

            if isinstance(
                flow_info,
                dict,
            ):

                station_name = (
                    flow_info.get(
                        "station"
                    )
                    or flow_info.get(
                        "series_name"
                    )
                )

                series_id = (
                    flow_info.get(
                        "series_id"
                    )
                )

                if station_name:

                    text = (
                        "Serie INA utilizada: "
                        f"**{station_name}**"
                    )

                    if (
                        series_id
                        is not None
                    ):

                        text += (
                            f" · series_id **{series_id}**"
                        )

                    st.caption(
                        text
                    )


    else:

        st.warning(
            "No se obtuvo una serie de caudal utilizable."
        )


    # ========================================================
    # PRECIPITACIÓN
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
            exog_future
            .head(
                15
            )
            .copy()
        )

        rain[
            "precip_mm"
        ] = (
            pd.to_numeric(
                rain[
                    "precip_mm"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
        )


        r1, r2, r3 = st.columns(
            3
        )


        r1.metric(

            "Acumulado 15 días",

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

                x=
                    rain[
                        "datetime"
                    ],

                y=
                    rain[
                        "precip_mm"
                    ],

                name=
                    "Precipitación prevista",

                marker_color=
                    "#38BDF8",
            )
        )


        rain_range = calcular_rango_y(
            rain[
                "precip_mm"
            ],
            margen=0.15,
        )


        if (
            rain_range
            is not None
        ):

            rain_range[
                0
            ] = 0


        rain_fig.update_layout(

            height=320,

            yaxis_title=
                "Precipitación (mm/día)",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
        )


        rain_fig.update_yaxes(
            range=
                rain_range
        )


        rain_fig.update_xaxes(
            tickformat=
                "%d/%m"
        )


        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )


    else:

        st.info(
            "No se encuentra disponible precipitación prevista."
        )


    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    with st.expander(
        "🛠️ Diagnóstico técnico"
    ):

        st.write(
            "**App:**",
            APP_VERSION,
        )


        if isinstance(
            hydrology,
            dict,
        ):

            st.write(
                "**Hydrology:**",
                hydrology.get(
                    "version",
                    "—",
                ),
            )


            st.write(
                "**Fuente historial San Nicolás:**",
                hydrology.get(
                    "history_source",
                    "—",
                ),
            )


            st.write(
                "**Registros San Nicolás usados por hydrology:**",
                hydrology.get(
                    "san_nicolas_records",
                    0,
                ),
            )


            st.write(
                "**Registros Corrientes usados por hydrology:**",
                hydrology.get(
                    "corrientes_records",
                    0,
                ),
            )


            years = safe_float(
                hydrology.get(
                    "historical_years"
                )
            )

            if np.isfinite(
                years
            ):

                st.write(
                    "**Años históricos coincidentes:**",
                    f"{years:.1f}",
                )


            if (
                hydrology.get(
                    "error"
                )
            ):

                st.error(
                    hydrology[
                        "error"
                    ]
                )


        if isinstance(
            upstream_history,
            pd.DataFrame,
        ):

            st.write(
                "**Registros aguas arriba:**",
                len(
                    upstream_history
                ),
            )


        if isinstance(
            exog_meta,
            dict,
        ):

            st.write(
                "**Registros caudal:**",
                exog_meta.get(
                    "caudal_records",
                    0,
                ),
            )


            st.write(
                "**Fuente caudal:**",
                exog_meta.get(
                    "caudal_source",
                    "—",
                ),
            )


            if (
                exog_meta.get(
                    "caudal_series"
                )
            ):

                st.json(
                    exog_meta[
                        "caudal_series"
                    ]
                )


        if model_error:

            st.error(
                "Modelo: "
                + str(
                    model_error
                )
            )

        else:

            st.success(
                "Modelo entrenado y pronóstico generado."
            )


    # ========================================================
    # PIE
    # ========================================================

    st.divider()

    st.caption(
        "Nivel y caudal: Instituto Nacional del Agua (INA) · "
        "Precipitación: Open-Meteo · "
        "Propagación y pronóstico: modelo experimental propio."
    )


    actualizado = (
        st.session_state.get(
            "actualizado"
        )
    )


    if actualizado:

        st.caption(
            "Última actualización: "
            + actualizado.strftime(
                "%d/%m/%Y %H:%M"
            )
        )


    st.warning(
        "La estimación de propagación Corrientes → San Nicolás "
        "es estadística y experimental. No constituye un pronóstico "
        "hidráulico oficial ni reemplaza comunicaciones del INA "
        "u organismos competentes."
    )
