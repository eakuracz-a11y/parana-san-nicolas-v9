import unicodedata
import requests
import pandas as pd


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# BASE V11
# ============================================================

INA_SERIES_URL = (
    "https://alerta.ina.gob.ar/pub/datos/series"
)

INA_DATA_URL = (
    "https://alerta.ina.gob.ar/pub/datos/datos"
)


UPSTREAM_STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Rosario",
    "Villa Constitución",
]


VAR_ID_LEVEL = 2

_CATALOG_CACHE = None


# ============================================================
# TEXTO
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
# REQUEST
# ============================================================

def request_json(
    url,
    params=None,
    timeout=60,
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
# CATÁLOGO
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

    _CATALOG_CACHE = (
        catalogo
    )

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

        elif procid == 2:
            score += 70

        if obs_count > 0:
            score += 40

        if pd.notna(
            to_date
        ):

            age = (
                pd.Timestamp.now(
                    tz="UTC"
                )
                - to_date
            ).days

            if age <= 7:
                score += 60

            elif age <= 30:
                score += 50

            elif age <= 180:
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
            x["score"],
        reverse=True,
    )

    return candidatos[
        0
    ]


# ============================================================
# CONSULTAR SERIE
# ============================================================

def consultar_serie(
    series_id,
    start,
    end,
):

    params = {
        "timeStart":
            str(start),

        "timeEnd":
            str(end),

        "seriesId":
            int(
                series_id
            ),

        "format":
            "json",
    }

    try:

        data = request_json(
            INA_DATA_URL,
            params=params,
        )

    except Exception:

        return pd.DataFrame()

    if isinstance(
        data,
        dict,
    ):

        if (
            "data"
            in data
        ):

            data = data[
                "data"
            ]

        elif (
            "results"
            in data
        ):

            data = data[
                "results"
            ]

        elif (
            "observaciones"
            in data
        ):

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

    fecha_col = None
    valor_col = None

    for col in [
        "timestart",
        "timeStart",
        "datetime",
        "fecha",
        "date",
        "timestamp",
    ]:

        if col in df.columns:

            fecha_col = col
            break

    for col in [
        "valor",
        "value",
        "nivel",
        "altura",
    ]:

        if col in df.columns:

            valor_col = col
            break

    if (
        fecha_col is None
        or valor_col is None
    ):

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
        "datetime"
    ] = (
        result[
            "datetime"
        ]
        .dt.tz_localize(
            None
        )
        .dt.normalize()
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
        .groupby(
            "datetime",
            as_index=False,
        )["value"]
        .mean()
    )

    return result


# ============================================================
# DESCARGAR AGUAS ARRIBA
# ============================================================

def get_upstream_history(
    start,
    end,
):

    resultado = None

    metadata = {}

    for estacion in (
        UPSTREAM_STATIONS
    ):

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
                "value":
                    nombre_columna
            }
        )

        if resultado is None:

            resultado = df

        else:

            resultado = (
                resultado.merge(
                    df,
                    on="datetime",
                    how="outer",
                )
            )

    if resultado is None:

        resultado = (
            pd.DataFrame(
                columns=[
                    "datetime"
                ]
            )
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
