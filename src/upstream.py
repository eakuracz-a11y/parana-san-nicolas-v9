import unicodedata

import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.1 - ESTACIONES AGUAS ARRIBA
# ============================================================

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"
INA_DATA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

REQUEST_TIMEOUT = 60

VAR_ID_LEVEL = 2


# ============================================================
# ESTACIONES AGUAS ARRIBA
# ============================================================

UPSTREAM_STATIONS = [
    "Corrientes",
    "Goya",
    "La Paz",
    "Paraná",
    "Diamante",
    "Rosario",
    "Villa Constitución",
]


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

    texto = unicodedata.normalize(
        "NFKD",
        str(texto),
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(
            caracter
        )
    )

    return texto.strip().lower()


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
# REQUEST JSON
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
            "Accept":
                "application/json,text/plain,*/*",

            "User-Agent":
                "Parana-San-Nicolas-V11.1/1.0",
        },
    )

    response.raise_for_status()

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió sin contenido."
        )

    return response.json()


# ============================================================
# EXTRAER LISTA
# ============================================================

def extraer_lista(
    data,
):

    if data is None:
        return []

    if isinstance(
        data,
        list,
    ):
        return data

    if not isinstance(
        data,
        dict,
    ):
        return []

    for key in [
        "data",
        "datos",
        "observaciones",
        "results",
        "result",
        "items",
    ]:

        value = data.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            nested = extraer_lista(
                value
            )

            if nested:
                return nested

    for value in data.values():

        if isinstance(
            value,
            list,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            nested = extraer_lista(
                value
            )

            if nested:
                return nested

    return []


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

        catalogo = extraer_lista(
            data
        )

    except Exception:

        catalogo = []

    _CATALOG_CACHE = (
        catalogo
    )

    return catalogo


# ============================================================
# BUSCAR SERIE DE NIVEL
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
                row.get(
                    "station_name",
                    "",
                ),
            )
        )

        if nombre != objetivo:
            continue

        # ----------------------------------------------------
        # VARIABLE
        # ----------------------------------------------------

        raw_var = (
            row.get("varid")
            or row.get("varId")
            or row.get("var_id")
        )

        try:

            var_id = int(
                raw_var
            )

        except Exception:

            continue

        if var_id != VAR_ID_LEVEL:
            continue

        # ----------------------------------------------------
        # SERIES ID
        # ----------------------------------------------------

        raw_series = (
            row.get("seriesid")
            or row.get("seriesId")
            or row.get("series_id")
        )

        try:

            series_id = int(
                raw_series
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # PROCEDIMIENTO
        # ----------------------------------------------------

        try:

            procid = int(
                row.get(
                    "procid",
                    -1,
                )
            )

        except Exception:

            procid = -1

        # ----------------------------------------------------
        # CANTIDAD DE OBSERVACIONES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ÚLTIMA FECHA
        # ----------------------------------------------------

        to_date = pd.to_datetime(
            row.get(
                "to_date"
            ),
            errors="coerce",
            utc=True,
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = 0

        if procid == 1:

            score += 100

        elif procid == 2:

            score += 70

        if obs_count > 0:

            score += 30

        if pd.notna(
            to_date
        ):

            age_days = (
                pd.Timestamp.now(
                    tz="UTC"
                )
                - to_date
            ).days

            if age_days <= 7:

                score += 100

            elif age_days <= 30:

                score += 70

            elif age_days <= 180:

                score += 40

            elif age_days <= 365:

                score += 20

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
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )

    return candidatos[
        0
    ]


# ============================================================
# DETECTAR COLUMNA FECHA
# ============================================================

def detectar_columna_fecha(
    df,
):

    candidatos = [
        "timestart",
        "timeStart",
        "datetime",
        "dateTime",
        "timestamp",
        "fecha",
        "date",
        "time",
    ]

    for candidato in candidatos:

        for columna in df.columns:

            if (
                str(
                    columna
                ).lower()
                ==
                candidato.lower()
            ):

                return columna

    for columna in df.columns:

        nombre = str(
            columna
        ).lower()

        if (
            "fecha" in nombre
            or "date" in nombre
            or "time" in nombre
        ):

            return columna

    return None


# ============================================================
# DETECTAR COLUMNA VALOR
# ============================================================

def detectar_columna_valor(
    df,
):

    candidatos = [
        "valor",
        "value",
        "nivel",
        "altura",
        "level",
        "height",
    ]

    for candidato in candidatos:

        for columna in df.columns:

            if (
                str(
                    columna
                ).lower()
                ==
                candidato.lower()
            ):

                return columna

    for columna in df.columns:

        nombre = str(
            columna
        ).lower()

        if (
            "valor" in nombre
            or "value" in nombre
            or "nivel" in nombre
            or "altura" in nombre
        ):

            return columna

    return None


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

    registros = extraer_lista(
        data
    )

    if not registros:

        return pd.DataFrame()

    try:

        raw = pd.json_normalize(
            registros
        )

    except Exception:

        raw = pd.DataFrame(
            registros
        )

    if raw.empty:

        return pd.DataFrame()

    fecha_col = detectar_columna_fecha(
        raw
    )

    valor_col = detectar_columna_valor(
        raw
    )

    if (
        fecha_col is None
        or valor_col is None
    ):

        return pd.DataFrame()

    result = pd.DataFrame()

    result[
        "datetime"
    ] = pd.to_datetime(
        raw[
            fecha_col
        ],
        errors="coerce",
        utc=True,
    )

    result[
        "value"
    ] = pd.to_numeric(
        raw[
            valor_col
        ]
        .astype(str)
        .str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    if result.empty:

        return pd.DataFrame()

    # Pasar a día
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

    # Un valor medio diario
    result = (
        result
        .groupby(
            "datetime",
            as_index=False,
        )[
            "value"
        ]
        .mean()
    )

    result = (
        result
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# DESCARGAR UNA ESTACIÓN
# ============================================================

def get_station_history(
    estacion,
    start,
    end,
):

    info = buscar_serie_nivel(
        estacion
    )

    if info is None:

        return (
            pd.DataFrame(),
            None,
        )

    df = consultar_serie(
        series_id=
            info[
                "series_id"
            ],
        start=
            start,
        end=
            end,
    )

    if df.empty:

        return (
            df,
            info,
        )

    columna = nombre_columna_estacion(
        estacion
    )

    df = df.rename(
        columns={
            "value":
                columna
        }
    )

    return (
        df,
        info,
    )


# ============================================================
# HISTÓRICO COMPLETO AGUAS ARRIBA
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

        (
            station_df,
            info,
        ) = get_station_history(
            estacion=
                estacion,
            start=
                start,
            end=
                end,
        )

        metadata[
            estacion
        ] = info

        if (
            station_df is None
            or station_df.empty
        ):

            continue

        if resultado is None:

            resultado = (
                station_df.copy()
            )

        else:

            resultado = resultado.merge(
                station_df,
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
# RESUMEN DE ESTACIONES
# ============================================================

def resumen_upstream(
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

    for estacion in (
        UPSTREAM_STATIONS
    ):

        col = nombre_columna_estacion(
            estacion
        )

        if col not in upstream_history.columns:
            continue

        temp = upstream_history[
            [
                "datetime",
                col,
            ]
        ].copy()

        temp[
            col
        ] = pd.to_numeric(
            temp[
                col
            ],
            errors="coerce",
        )

        temp = temp.dropna(
            subset=[
                col
            ]
        )

        if temp.empty:
            continue

        nivel_actual = float(
            temp[
                col
            ].iloc[-1]
        )

        nivel_anterior = None
        variacion = None

        if len(
            temp
        ) >= 2:

            nivel_anterior = float(
                temp[
                    col
                ].iloc[-2]
            )

            variacion = (
                nivel_actual
                - nivel_anterior
            )

        if variacion is None:

            tendencia = (
                "Sin comparación"
            )

        elif variacion > 0.01:

            tendencia = (
                "↑ Creciendo"
            )

        elif variacion < -0.01:

            tendencia = (
                "↓ Bajando"
            )

        else:

            tendencia = (
                "→ Estable"
            )

        rows.append(
            {
                "Estación":
                    estacion,

                "Nivel actual":
                    round(
                        nivel_actual,
                        2,
                    ),

                "Nivel anterior":
                    (
                        round(
                            nivel_anterior,
                            2,
                        )
                        if nivel_anterior
                        is not None
                        else None
                    ),

                "Variación":
                    (
                        round(
                            variacion,
                            2,
                        )
                        if variacion
                        is not None
                        else None
                    ),

                "Tendencia":
                    tendencia,

                "Última fecha":
                    pd.to_datetime(
                        temp[
                            "datetime"
                        ].iloc[-1]
                    ).strftime(
                        "%d/%m/%Y"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )
