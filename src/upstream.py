import requests
import pandas as pd
import numpy as np
import unicodedata


# ============================================================
# CONFIGURACIÓN INA
# ============================================================

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"


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

# Altura hidrométrica
VAR_ID_LEVEL = 2


# ============================================================
# CACHE
# ============================================================

_CATALOG_CACHE = None


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================

def normalizar_texto(texto):

    if texto is None:
        return ""

    texto = str(texto)

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(c)
    )

    return (
        texto
        .strip()
        .lower()
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
            "User-Agent": (
                "Parana-San-Nicolas-V9/1.0"
            ),
            "Accept": (
                "application/json,"
                "text/plain,*/*"
            ),
        },
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CATÁLOGO INA
# ============================================================

def obtener_catalogo():

    global _CATALOG_CACHE

    if _CATALOG_CACHE is not None:
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
            row.get("seriesid")
            or row.get("seriesId")
        )

        if series_id is None:
            continue

        try:

            series_id = int(
                series_id
            )

        except Exception:
            continue

        # -----------------------------------------------
        # PROCEDIMIENTO
        # -----------------------------------------------

        try:

            procid = int(
                row.get(
                    "procid",
                    -1,
                )
            )

        except Exception:

            procid = -1

        # -----------------------------------------------
        # OBSERVACIONES
        # -----------------------------------------------

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

        # -----------------------------------------------
        # FECHA FINAL
        # -----------------------------------------------

        to_date = pd.to_datetime(
            row.get(
                "to_date"
            ),
            errors="coerce",
            utc=True,
        )

        # -----------------------------------------------
        # SCORE
        # -----------------------------------------------

        score = 0

        # Medición directa
        if procid == 1:
            score += 100

        # Otra observación real
        elif procid not in [
            4,
            8,
        ]:
            score += 40

        # Evitar simulados y trazas
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
                "station": estacion,
                "series_id": series_id,
                "procid": procid,
                "proc_name": row.get(
                    "proc_nombre"
                ),
                "obs_count": obs_count,
                "from_date": row.get(
                    "from_date"
                ),
                "to_date": row.get(
                    "to_date"
                ),
                "unit": row.get(
                    "unit_nombre"
                ),
                "score": score,
            }
        )

    if not candidatos:
        return None

    candidatos = sorted(
        candidatos,
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    return candidatos[0]


# ============================================================
# CONSULTAR UNA SERIE
# ============================================================

def consultar_serie(
    series_id,
    start,
    end,
):

    params = {
        "tipo": "puntual",
        "series_id": int(
            series_id
        ),
        "timestart": str(
            start
        ),
        "timeend": str(
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
    ] = pd.to_datetime(
        df[
            "timestart"
        ],
        errors="coerce",
        utc=True,
    )

    df[
        "datetime"
    ] = (
        df["datetime"]
        .dt.tz_localize(None)
        .dt.normalize()
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
        df.groupby(
            "datetime",
            as_index=False,
        )["value"]
        .mean()
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
            "nivel_"
            + normalizar_texto(
                estacion
            )
            .replace(
                " ",
                "_",
            )
        )

        df = df.rename(
            columns={
                "value": nombre_columna
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

    resultado = (
        resultado
        .sort_values(
            "datetime"
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
# CREAR VARIABLES AGUAS ARRIBA
# ============================================================

def preparar_upstream_features(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    out = df.copy()

    level_cols = [
        c
        for c in out.columns
        if c.startswith(
            "nivel_"
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

        out[
            col
        ] = (
            out[col]
            .interpolate(
                limit=3,
                limit_direction="both",
            )
        )

        # -----------------------------------------------
        # CAMBIO DIARIO
        # -----------------------------------------------

        out[
            f"{col}_diff1"
        ] = out[
            col
        ].diff()

        # -----------------------------------------------
        # TENDENCIA 3 DÍAS
        # -----------------------------------------------

        out[
            f"{col}_trend3"
        ] = (
            out[col]
            - out[
                col
            ].shift(
                3
            )
        ) / 3.0

        # -----------------------------------------------
        # LAGS
        # -----------------------------------------------

        for lag in [
            1,
            2,
            3,
            5,
            7,
            10,
        ]:

            out[
                f"{col}_lag{lag}"
            ] = out[
                col
            ].shift(
                lag
            )

    return out
