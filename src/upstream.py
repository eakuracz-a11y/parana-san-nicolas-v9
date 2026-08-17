import requests
import pandas as pd
import numpy as np
import unicodedata


# ============================================================
# CONFIGURACIÓN INA
# ============================================================

INA_SERIES_URL = (
    "https://alerta.ina.gob.ar/pub/datos/series"
)

INA_A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

MAX_FORECAST_DAYS = 30

TREND_WINDOW = 7

LEVEL_MIN = 0.0

LEVEL_MAX = 15.0


# ============================================================
# ESTACIONES AGUAS ARRIBA
# ============================================================

UPSTREAM_STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Rosario",
    "Villa Constitución",
]


# ============================================================
# VARIABLE
# ============================================================

VAR_ID_LEVEL = 2


# ============================================================
# CACHE
# ============================================================

_CATALOG_CACHE = None


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar_texto(
    texto,
):

    if texto is None:
        return ""

    texto = str(
        texto
    )

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(
            c
        )
    )

    return (
        texto
        .strip()
        .lower()
    )


# ============================================================
# NOMBRE DE COLUMNA
# ============================================================

def nombre_columna_estacion(
    estacion,
):

    return (
        "nivel_"
        + normalizar_texto(
            estacion
        )
        .replace(
            " ",
            "_",
        )
    )


# ============================================================
# REQUEST
# ============================================================

def request_json(
    url,
    params=None,
    timeout=40,
):

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent":
                "Parana-San-Nicolas-V11/1.0",

            "Accept":
                "application/json,text/plain,*/*",
        },
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# NORMALIZAR FECHAS
# ============================================================

def normalizar_fechas(
    serie,
):

    return (
        pd.to_datetime(
            serie,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(
            None
        )
        .dt
        .normalize()
    )


# ============================================================
# CATÁLOGO INA
# ============================================================

def obtener_catalogo():

    global _CATALOG_CACHE

    if (
        _CATALOG_CACHE
        is not None
    ):
        return _CATALOG_CACHE

    try:

        data = request_json(
            INA_SERIES_URL
        )

    except Exception:

        _CATALOG_CACHE = []

        return []

    if isinstance(
        data,
        list,
    ):

        catalogo = data

    elif isinstance(
        data,
        dict,
    ):

        catalogo = []

        for value in data.values():

            if isinstance(
                value,
                list,
            ):

                catalogo = value
                break

    else:

        catalogo = []

    _CATALOG_CACHE = catalogo

    return catalogo


# ============================================================
# BUSCAR MEJOR SERIE
# ============================================================

def buscar_serie_nivel(
    estacion,
):

    catalogo = obtener_catalogo()

    if not catalogo:
        return None

    objetivo = normalizar_texto(
        estacion
    )

    candidatos = []

    for row in catalogo:

        if not isinstance(
            row,
            dict,
        ):
            continue

        nombre = normalizar_texto(
            row.get(
                "estacion_nombre",
                "",
            )
        )

        if nombre != objetivo:
            continue

        try:

            varid = int(
                row.get(
                    "varid",
                    -1,
                )
            )

        except Exception:
            continue

        if varid != VAR_ID_LEVEL:
            continue

        series_id = (
            row.get(
                "seriesid"
            )
            or row.get(
                "seriesId"
            )
        )

        if series_id is None:
            continue

        try:

            series_id = int(
                series_id
            )

        except Exception:
            continue

        try:

            procid = int(
                row.get(
                    "procid",
                    -1,
                )
            )

        except Exception:

            procid = -1

        try:

            obs_count = int(
                row.get(
                    "obs_count",
                    0,
                )
                or 0
            )

        except Exception:

            obs_count = 0

        to_date = pd.to_datetime(
            row.get(
                "to_date"
            ),
            errors="coerce",
            utc=True,
        )

        score = 0

        if procid == 1:
            score += 100

        elif procid not in [
            4,
            8,
        ]:
            score += 40

        if procid == 4:
            score -= 100

        if procid == 8:
            score -= 80

        if obs_count > 100:
            score += 20

        if obs_count > 1000:
            score += 20

        if pd.notna(
            to_date
        ):

            ahora = pd.Timestamp.now(
                tz="UTC"
            )

            age = (
                ahora
                - to_date
            ).days

            if age <= 3:
                score += 60

            elif age <= 15:
                score += 50

            elif age <= 60:
                score += 30

            elif age <= 365:
                score += 10

        candidatos.append(
            {
                "station":
                    estacion,

                "series_id":
                    series_id,

                "procid":
                    procid,

                "proc_name":
                    row.get(
                        "proc_nombre"
                    ),

                "obs_count":
                    obs_count,

                "from_date":
                    row.get(
                        "from_date"
                    ),

                "to_date":
                    row.get(
                        "to_date"
                    ),

                "unit":
                    row.get(
                        "unit_nombre"
                    ),

                "score":
                    score,
            }
        )

    if not candidatos:
        return None

    candidatos = sorted(
        candidatos,
        key=lambda x:
            x[
                "score"
            ],
        reverse=True,
    )

    return candidatos[
        0
    ]


# ============================================================
# CONSULTAR UNA SERIE
# ============================================================

def consultar_serie(
    series_id,
    start,
    end,
):

    params = {
        "tipo":
            "puntual",

        "series_id":
            int(
                series_id
            ),

        "timestart":
            str(
                start
            ),

        "timeend":
            str(
                end
            ),
    }

    try:

        data = request_json(
            INA_A5_URL,
            params=params,
        )

    except Exception:

        return pd.DataFrame()

    if not isinstance(
        data,
        list,
    ):
        return pd.DataFrame()

    df = pd.DataFrame(
        data
    )

    if df.empty:
        return df

    if (
        "timestart"
        not in df.columns
        or "valor"
        not in df.columns
    ):
        return pd.DataFrame()

    df[
        "datetime"
    ] = normalizar_fechas(
        df[
            "timestart"
        ]
    )

    df[
        "value"
    ] = pd.to_numeric(
        df[
            "valor"
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    df = (
        df
        .groupby(
            "datetime",
            as_index=False,
        )[
            "value"
        ]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# DESCARGAR NIVELES AGUAS ARRIBA
# ============================================================

def get_upstream_history(
    start,
    end,
):

    resultado = None

    metadata = {}

    for estacion in UPSTREAM_STATIONS:

        info = buscar_serie_nivel(
            estacion
        )

        metadata[
            estacion
        ] = info

        if info is None:
            continue

        df = consultar_serie(
            info[
                "series_id"
            ],
            start,
            end,
        )

        if df.empty:
            continue

        nombre_columna = (
            nombre_columna_estacion(
                estacion
            )
        )

        df = df.rename(
            columns={
                "value":
                    nombre_columna
            }
        )

        if resultado is None:

            resultado = df

        else:

            resultado = resultado.merge(
                df,
                on="datetime",
                how="outer",
            )

    if resultado is None:

        resultado = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    resultado[
        "datetime"
    ] = normalizar_fechas(
        resultado[
            "datetime"
        ]
    )

    resultado = (
        resultado
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

    return (
        resultado,
        metadata,
    )


# ============================================================
# PREPARAR HISTÓRICO
# ============================================================

def preparar_upstream_history(
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

    out = df.copy()

    if "datetime" not in out.columns:
        return pd.DataFrame()

    out[
        "datetime"
    ] = normalizar_fechas(
        out[
            "datetime"
        ]
    )

    level_cols = [
        c
        for c in out.columns
        if (
            c.startswith(
                "nivel_"
            )
            and "_lag"
            not in c
            and "_diff"
            not in c
            and "_trend"
            not in c
            and "_mean"
            not in c
            and "_actual"
            not in c
            and "_next"
            not in c
        )
    ]

    for col in level_cols:

        out[
            col
        ] = pd.to_numeric(
            out[
                col
            ],
            errors="coerce",
        )

        if (
            out[
                col
            ]
            .notna()
            .sum()
            >= 2
        ):

            out[
                col
            ] = (
                out[
                    col
                ]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    return (
        out
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
# PENDIENTE RECIENTE
# ============================================================

def calcular_pendiente(
    serie,
    ventana=TREND_WINDOW,
):

    valores = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
        .tail(
            ventana
        )
        .to_numpy(
            dtype=float
        )
    )

    if len(
        valores
    ) < 3:
        return 0.0

    x = np.arange(
        len(
            valores
        ),
        dtype=float,
    )

    try:

        pendiente = float(
            np.polyfit(
                x,
                valores,
                1,
            )[0]
        )

    except Exception:

        pendiente = 0.0

    return pendiente


# ============================================================
# VOLATILIDAD RECIENTE
# ============================================================

def calcular_volatilidad(
    serie,
    ventana=14,
):

    valores = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
        .tail(
            ventana
        )
    )

    if len(
        valores
    ) < 3:
        return 0.03

    cambios = (
        valores
        .diff()
        .dropna()
    )

    if cambios.empty:
        return 0.03

    volatilidad = float(
        cambios.std()
    )

    if (
        not np.isfinite(
            volatilidad
        )
        or volatilidad <= 0
    ):
        volatilidad = 0.03

    return volatilidad


# ============================================================
# PROYECTAR UNA ESTACIÓN
# ============================================================

def proyectar_estacion(
    serie,
    days,
):

    valid = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
    )

    if valid.empty:

        return [
            np.nan
        ] * days

    actual = float(
        valid.iloc[
            -1
        ]
    )

    pendiente = calcular_pendiente(
        valid,
        ventana=TREND_WINDOW,
    )

    volatilidad = calcular_volatilidad(
        valid,
        ventana=14,
    )

    limite_diario = max(
        min(
            abs(
                actual
            )
            * 0.04,
            0.25,
        ),
        volatilidad
        * 1.5,
        0.03,
    )

    pendiente = float(
        np.clip(
            pendiente,
            -limite_diario,
            limite_diario,
        )
    )

    resultado = []

    nivel = actual

    for h in range(
        1,
        days + 1,
    ):

        amortiguacion = np.exp(
            -h
            / 12.0
        )

        incremento = (
            pendiente
            * amortiguacion
        )

        incremento = float(
            np.clip(
                incremento,
                -limite_diario,
                limite_diario,
            )
        )

        nivel = (
            nivel
            + incremento
        )

        nivel = float(
            np.clip(
                nivel,
                LEVEL_MIN,
                LEVEL_MAX,
            )
        )

        resultado.append(
            nivel
        )

    return resultado


# ============================================================
# PROYECCIÓN DIARIA AGUAS ARRIBA
# ============================================================

def project_upstream(
    upstream_history,
    future_dates=None,
    days=15,
):

    days = max(
        1,
        min(
            int(
                days
            ),
            MAX_FORECAST_DAYS,
        ),
    )

    history = preparar_upstream_history(
        upstream_history
    )

    if history.empty:

        return pd.DataFrame()

    dates = None

    if future_dates is not None:

        future_dates = pd.to_datetime(
            future_dates,
            errors="coerce",
        )

        future_dates = pd.DatetimeIndex(
            future_dates
        )

        future_dates = future_dates[
            ~future_dates.isna()
        ]

        if len(
            future_dates
        ) > 0:

            dates = pd.DatetimeIndex(
                future_dates[
                    :days
                ]
            )

    if dates is None:

        last_date = history[
            "datetime"
        ].max()

        dates = pd.date_range(
            last_date
            + pd.Timedelta(
                days=1
            ),
            periods=days,
            freq="D",
        )

    else:

        days = len(
            dates
        )

    result = pd.DataFrame(
        {
            "datetime":
                dates
        }
    )

    level_cols = [
        c
        for c in history.columns
        if (
            c.startswith(
                "nivel_"
            )
            and "_lag"
            not in c
            and "_diff"
            not in c
            and "_trend"
            not in c
            and "_mean"
            not in c
            and "_actual"
            not in c
            and "_next"
            not in c
        )
    ]

    for col in level_cols:

        projection = proyectar_estacion(
            history[
                col
            ],
            days=days,
        )

        result[
            col
        ] = projection

    return result


# ============================================================
# OBTENER HISTÓRICO + PROYECCIÓN
# ============================================================

def get_upstream_data(
    start,
    end,
    forecast_days=15,
    future_dates=None,
):

    forecast_days = max(
        1,
        min(
            int(
                forecast_days
            ),
            MAX_FORECAST_DAYS,
        ),
    )

    (
        history,
        metadata,
    ) = get_upstream_history(
        start,
        end,
    )

    history = preparar_upstream_history(
        history
    )

    future = project_upstream(
        upstream_history=history,
        future_dates=future_dates,
        days=forecast_days,
    )

    projection_meta = {}

    level_cols = [
        c
        for c in history.columns
        if (
            c.startswith(
                "nivel_"
            )
            and "_lag"
            not in c
            and "_diff"
            not in c
            and "_trend"
            not in c
            and "_mean"
            not in c
            and "_actual"
            not in c
            and "_next"
            not in c
        )
    ]

    for col in level_cols:

        serie = (
            pd.to_numeric(
                history[
                    col
                ],
                errors="coerce",
            )
            .dropna()
        )

        if serie.empty:

            projection_meta[
                col
            ] = {
                "actual":
                    None,

                "pendiente":
                    None,

                "estado":
                    "Sin datos",
            }

            continue

        actual = float(
            serie.iloc[
                -1
            ]
        )

        pendiente = calcular_pendiente(
            serie,
            ventana=TREND_WINDOW,
        )

        umbral = max(
            abs(
                actual
            )
            * 0.003,
            0.01,
        )

        if pendiente > umbral:

            estado = "Creciente"

        elif pendiente < -umbral:

            estado = "Bajante"

        else:

            estado = "Estable"

        projection_meta[
            col
        ] = {
            "actual":
                actual,

            "pendiente":
                pendiente,

            "estado":
                estado,
        }

    return (
        history,
        future,
        metadata,
        projection_meta,
    )


# ============================================================
# CREAR VARIABLES AGUAS ARRIBA
# ============================================================

def preparar_upstream_features(
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

    out = preparar_upstream_history(
        df
    )

    if out.empty:
        return out

    level_cols = [
        c
        for c in out.columns
        if (
            c.startswith(
                "nivel_"
            )
            and "_lag"
            not in c
            and "_diff"
            not in c
            and "_trend"
            not in c
            and "_mean"
            not in c
            and "_actual"
            not in c
            and "_next"
            not in c
        )
    ]

    for col in level_cols:

        out[
            f"{col}_actual"
        ] = out[
            col
        ]

        out[
            f"{col}_diff1"
        ] = (
            out[
                col
            ]
            .diff()
        )

        out[
            f"{col}_trend3"
        ] = (
            out[
                col
            ]
            - out[
                col
            ].shift(
                3
            )
        ) / 3.0

        out[
            f"{col}_trend7"
        ] = (
            out[
                col
            ]
            - out[
                col
            ].shift(
                7
            )
        ) / 7.0

        out[
            f"{col}_mean7"
        ] = (
            out[
                col
            ]
            .rolling(
                7
            )
            .mean()
        )

        for lag in [
            1,
            2,
            3,
            5,
            7,
            10,
            14,
        ]:

            out[
                f"{col}_lag{lag}"
            ] = out[
                col
            ].shift(
                lag
            )

    return out
