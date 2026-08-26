import unicodedata
import requests
import pandas as pd


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
# BASE ESTABLE V11
# ============================================================

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"
INA_DATA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

REQUEST_TIMEOUT = 60

VAR_ID_LEVEL = 2

TARGET_STATION = "San Nicolás"


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


_CATALOG_CACHE = None


# ============================================================
# TEXTO
# ============================================================

def normalizar_texto(texto):

    if texto is None:
        return ""

    texto = unicodedata.normalize(
        "NFKD",
        str(texto),
    )

    texto = "".join(
        c
        for c in texto
        if not unicodedata.combining(c)
    )

    return texto.strip().lower()


# ============================================================
# REQUEST
# ============================================================

def request_json(
    url,
    params=None,
    timeout=REQUEST_TIMEOUT,
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
# CATÁLOGO INA
# ============================================================

def obtener_catalogo():

    global _CATALOG_CACHE

    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    data = request_json(
        INA_SERIES_URL
    )

    if isinstance(data, list):

        catalogo = data

    elif isinstance(data, dict):

        catalogo = []

        for value in data.values():

            if isinstance(value, list):

                catalogo = value
                break

    else:

        catalogo = []

    _CATALOG_CACHE = catalogo

    return catalogo


# ============================================================
# BUSCAR SERIE DE NIVEL
# ============================================================

def buscar_serie_nivel(
    estacion=TARGET_STATION,
):

    catalogo = obtener_catalogo()

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

        # Serie observada/puntual preferida
        if procid == 1:
            score += 100

        elif procid == 2:
            score += 70

        if obs_count > 0:
            score += 40

        if pd.notna(to_date):

            age = (
                pd.Timestamp.now(
                    tz="UTC"
                )
                - to_date
            ).days

            if age <= 7:
                score += 80

            elif age <= 30:
                score += 60

            elif age <= 180:
                score += 40

            elif age <= 365:
                score += 20

        candidatos.append(
            {
                "station": estacion,
                "series_id": series_id,
                "varid": varid,
                "procid": procid,
                "proc_name":
                    row.get(
                        "proc_nombre"
                    ),
                "unit":
                    row.get(
                        "unit_nombre"
                    ),
                "from_date":
                    row.get(
                        "from_date"
                    ),
                "to_date":
                    row.get(
                        "to_date"
                    ),
                "obs_count":
                    obs_count,
                "score":
                    score,
            }
        )

    if not candidatos:
        return None

    candidatos = sorted(
        candidatos,
        key=lambda x:
            x["score"],
        reverse=True,
    )

    return candidatos[0]


# ============================================================
# CONSULTAR OBSERVACIONES
# ============================================================

def consultar_serie(
    series_id,
    start,
    end,
):

    params = {
        "timeStart": str(start),
        "timeEnd": str(end),
        "seriesId": int(
            series_id
        ),
        "format": "json",
    }

    data = request_json(
        INA_DATA_URL,
        params=params,
    )

    if isinstance(data, dict):

        if data.get("mensaje"):

            raise RuntimeError(
                str(
                    data.get(
                        "mensaje"
                    )
                )
            )

        if "data" in data:

            data = data[
                "data"
            ]

        elif "results" in data:

            data = data[
                "results"
            ]

        elif "observaciones" in data:

            data = data[
                "observaciones"
            ]

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

    # --------------------------------------------------------
    # FECHA
    # --------------------------------------------------------

    fecha_col = None

    for col in [
        "timestart",
        "timeStart",
        "datetime",
        "fecha",
        "date",
        "timestamp",
        "time",
    ]:

        if col in df.columns:

            fecha_col = col
            break

    # --------------------------------------------------------
    # VALOR
    # --------------------------------------------------------

    valor_col = None

    for col in [
        "valor",
        "value",
        "nivel",
        "altura",
        "level",
    ]:

        if col in df.columns:

            valor_col = col
            break

    if fecha_col is None:

        return pd.DataFrame()

    if valor_col is None:

        return pd.DataFrame()

    result = pd.DataFrame()

    result[
        "datetime"
    ] = pd.to_datetime(
        df[
            fecha_col
        ],
        errors="coerce",
        utc=True,
    )

    result[
        "value"
    ] = pd.to_numeric(
        df[
            valor_col
        ],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    result = (
        result
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

    return result


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def observed(
    start,
    end,
):

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        if info is None:

            return (
                pd.DataFrame(),
                (
                    "No fue posible localizar "
                    "una serie de nivel activa "
                    "para San Nicolás en el "
                    "catálogo del INA."
                ),
            )

        df = consultar_serie(
            info[
                "series_id"
            ],
            start,
            end,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA no devolvió "
                    "observaciones para "
                    f"San Nicolás · serie "
                    f"{info['series_id']}."
                ),
            )

        df[
            "station"
        ] = TARGET_STATION

        df[
            "series_id"
        ] = info[
            "series_id"
        ]

        df[
            "variable"
        ] = (
            "Nivel hidrométrico"
        )

        df[
            "unit"
        ] = "m"

        return (
            df,
            None,
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"INA: {exc}",
        )


# ============================================================
# META
# ============================================================

def forecast_meta():

    info = None

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

    except Exception:
        pass

    return {
        "fuente":
            "Instituto Nacional del Agua (INA)",
        "servicio":
            "Web API pública INA",
        "estacion":
            TARGET_STATION,
        "serie":
            (
                info.get(
                    "series_id"
                )
                if isinstance(
                    info,
                    dict,
                )
                else None
            ),
        "variable":
            "Nivel hidrométrico",
        "unidad":
            "metros",
        "estado":
            "Consulta online",
    }
