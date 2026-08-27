import unicodedata
from urllib.parse import quote

import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
# V11.0 - CONEXIÓN INA CORREGIDA
# ============================================================

INA_SERIES_BASE = "https://alerta.ina.gob.ar/pub/datos/series"
INA_DATA_BASE = "https://alerta.ina.gob.ar/pub/datos/datos"

# Respaldo A5
INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

REQUEST_TIMEOUT = 60

TARGET_STATION = "San Nicolás"

# Nivel hidrométrico
VAR_ID_LEVEL = 2

# Serie que ya utilizamos históricamente como respaldo.
SAN_NICOLAS_FALLBACK_SERIES_ID = 36


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
# NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(value):

    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    return text.strip().lower()


# ============================================================
# FECHA
# ============================================================

def normalizar_fecha(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(dt):

        raise ValueError(
            f"Fecha inválida: {value}"
        )

    return dt.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# REQUEST
# ============================================================

def _request_url(url):

    try:

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept":
                    "application/json,text/plain,*/*",

                "User-Agent":
                    "Parana-San-Nicolas-V11/1.0",
            },
        )

    except requests.Timeout as exc:

        raise RuntimeError(
            "El INA demoró demasiado en responder."
        ) from exc

    except requests.ConnectionError as exc:

        raise RuntimeError(
            "No fue posible conectarse con el INA."
        ) from exc

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Error de comunicación con INA: {exc}"
        ) from exc

    if response.status_code != 200:

        raise RuntimeError(
            f"INA respondió HTTP {response.status_code}."
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió sin contenido."
        )

    try:

        return response.json()

    except ValueError as exc:

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Inicio de respuesta: {response.text[:300]}"
        ) from exc


# ============================================================
# URL API PÚBLICA INA
# ============================================================

def _build_data_url(
    start,
    end,
    series_id,
):

    start_text = normalizar_fecha(
        start
    )

    end_text = normalizar_fecha(
        end
    )

    if (
        pd.to_datetime(start_text)
        >
        pd.to_datetime(end_text)
    ):

        raise ValueError(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    # IMPORTANTE:
    # El asistente del INA genera el identificador del recurso
    # agregando parámetros con & directamente luego de /datos.
    #
    # Ejemplo:
    # /pub/datos/datos&timeStart=...&timeEnd=...&seriesId=...
    #
    return (
        f"{INA_DATA_BASE}"
        f"&timeStart={quote(start_text)}"
        f"&timeEnd={quote(end_text)}"
        f"&seriesId={int(series_id)}"
        f"&format=json"
    )


# ============================================================
# URL CATÁLOGO
# ============================================================

def _build_series_url():

    # Pedimos catálogo de series observadas.
    return (
        f"{INA_SERIES_BASE}"
        f"&varId={VAR_ID_LEVEL}"
        f"&format=json"
    )


# ============================================================
# EXTRAER LISTAS
# ============================================================

def _extract_list(data):

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

    # --------------------------------------------------------
    # ERROR DEL INA
    # --------------------------------------------------------

    message = (
        data.get("mensaje")
        or data.get("message")
        or data.get("error")
    )

    possible_keys = [
        "data",
        "datos",
        "observaciones",
        "results",
        "result",
        "items",
    ]

    for key in possible_keys:

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

            nested = _extract_list(
                value
            )

            if nested:
                return nested

    # --------------------------------------------------------
    # BÚSQUEDA RECURSIVA
    # --------------------------------------------------------

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

            nested = _extract_list(
                value
            )

            if nested:
                return nested

    if message:

        raise RuntimeError(
            str(message)
        )

    return []


# ============================================================
# CATÁLOGO INA
# ============================================================

def obtener_catalogo():

    global _CATALOG_CACHE

    if _CATALOG_CACHE is not None:

        return _CATALOG_CACHE

    try:

        url = _build_series_url()

        data = _request_url(
            url
        )

        catalog = _extract_list(
            data
        )

        if catalog:

            _CATALOG_CACHE = catalog

            return catalog

    except Exception:

        pass

    # --------------------------------------------------------
    # SEGUNDO INTENTO:
    # catálogo sin filtros
    # --------------------------------------------------------

    try:

        data = _request_url(
            INA_SERIES_BASE
        )

        catalog = _extract_list(
            data
        )

        _CATALOG_CACHE = (
            catalog
            if catalog
            else []
        )

        return _CATALOG_CACHE

    except Exception:

        _CATALOG_CACHE = []

        return []


# ============================================================
# BUSCAR SERIE SAN NICOLÁS
# ============================================================

def buscar_serie_nivel(
    station=TARGET_STATION,
):

    catalog = obtener_catalogo()

    target = normalizar_texto(
        station
    )

    candidates = []

    for row in catalog:

        if not isinstance(
            row,
            dict,
        ):

            continue

        station_name = normalizar_texto(
            row.get(
                "estacion_nombre",
                row.get(
                    "station_name",
                    "",
                ),
            )
        )

        if station_name != target:

            continue

        # ----------------------------------------------------
        # VARIABLE
        # --------------------------------------------------------

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

            var_id = None

        if (
            var_id is not None
            and var_id != VAR_ID_LEVEL
        ):

            continue

        # ----------------------------------------------------
        # SERIES ID
        # --------------------------------------------------------

        raw_series = (
            row.get("seriesid")
            or row.get("seriesId")
            or row.get("series_id")
            or row.get("id")
        )

        try:

            series_id = int(
                raw_series
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # SCORE
        # --------------------------------------------------------

        score = 0

        try:

            procid = int(
                row.get(
                    "procid",
                    -1,
                )
            )

        except Exception:

            procid = -1

        if procid == 1:

            score += 100

        elif procid == 2:

            score += 70

        to_date = pd.to_datetime(
            row.get(
                "to_date"
            ),
            errors="coerce",
            utc=True,
        )

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
                score += 100

            elif age <= 30:
                score += 70

            elif age <= 180:
                score += 40

            elif age <= 365:
                score += 20

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

        if obs_count > 0:

            score += 30

        candidates.append(
            {
                "series_id":
                    series_id,

                "station":
                    station,

                "procid":
                    procid,

                "proc_name":
                    row.get(
                        "proc_nombre"
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

    if not candidates:

        # Respaldo conocido
        return {
            "series_id":
                SAN_NICOLAS_FALLBACK_SERIES_ID,

            "station":
                TARGET_STATION,

            "score":
                0,

            "source":
                "fallback",
        }

    candidates = sorted(
        candidates,
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )

    return candidates[
        0
    ]


# ============================================================
# DETECTAR FECHA
# ============================================================

def _find_datetime_column(df):

    candidates = [
        "timestart",
        "timeStart",
        "datetime",
        "dateTime",
        "timestamp",
        "fecha",
        "date",
        "time",
    ]

    columns = list(
        df.columns
    )

    for candidate in candidates:

        for column in columns:

            if (
                str(column).lower()
                ==
                candidate.lower()
            ):

                return column

    for column in columns:

        name = str(
            column
        ).lower()

        if (
            "fecha" in name
            or "date" in name
            or "time" in name
        ):

            return column

    return None


# ============================================================
# DETECTAR VALOR
# ============================================================

def _find_value_column(df):

    candidates = [
        "valor",
        "value",
        "nivel",
        "altura",
        "level",
        "height",
        "measurement",
    ]

    columns = list(
        df.columns
    )

    for candidate in candidates:

        for column in columns:

            if (
                str(column).lower()
                ==
                candidate.lower()
            ):

                return column

    for column in columns:

        name = str(
            column
        ).lower()

        if (
            "valor" in name
            or "value" in name
            or "nivel" in name
            or "altura" in name
        ):

            return column

    return None


# ============================================================
# NORMALIZAR OBSERVACIONES
# ============================================================

def _normalize_observations(
    records,
):

    if not records:

        return pd.DataFrame()

    try:

        raw = pd.json_normalize(
            records
        )

    except Exception:

        raw = pd.DataFrame(
            records
        )

    if raw.empty:

        return pd.DataFrame()

    date_col = (
        _find_datetime_column(
            raw
        )
    )

    value_col = (
        _find_value_column(
            raw
        )
    )

    if date_col is None:

        raise RuntimeError(
            "INA devolvió registros pero no "
            "se encontró la columna de fecha. "
            f"Columnas: {list(raw.columns)}"
        )

    if value_col is None:

        raise RuntimeError(
            "INA devolvió registros pero no "
            "se encontró la columna de nivel. "
            f"Columnas: {list(raw.columns)}"
        )

    result = pd.DataFrame()

    result[
        "datetime"
    ] = pd.to_datetime(
        raw[
            date_col
        ],
        errors="coerce",
        utc=True,
    )

    values = (
        raw[
            value_col
        ]
        .astype(str)
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    result[
        "value"
    ] = pd.to_numeric(
        values,
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    result = result[
        result[
            "value"
        ].between(
            -5.0,
            20.0,
        )
    ]

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
# CONSULTA API PÚBLICA
# ============================================================

def _query_public_api(
    series_id,
    start,
    end,
):

    url = _build_data_url(
        start=start,
        end=end,
        series_id=series_id,
    )

    data = _request_url(
        url
    )

    records = _extract_list(
        data
    )

    return _normalize_observations(
        records
    )


# ============================================================
# RESPALDO A5
# ============================================================

def _query_a5(
    series_id,
    start,
    end,
):

    start_text = normalizar_fecha(
        start
    )

    end_text = normalizar_fecha(
        end
    )

    # Esta es la sintaxis documentada por A5.
    params = {
        "tipo":
            "puntual",

        "series_id":
            int(
                series_id
            ),

        "timestart":
            start_text,

        "timeend":
            end_text,
    }

    response = requests.get(
        INA_A5_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "Accept":
                "application/json",

            "User-Agent":
                "Parana-San-Nicolas-V11/1.0",
        },
    )

    response.raise_for_status()

    data = response.json()

    records = _extract_list(
        data
    )

    return _normalize_observations(
        records
    )


# ============================================================
# GET SERIES
# ============================================================

def get_series(
    start,
    end,
    series_id=None,
):

    if series_id is None:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        series_id = info[
            "series_id"
        ]

    errors = []

    # ========================================================
    # INTENTO 1
    # API PÚBLICA CON FORMATO DEL ASISTENTE INA
    # ========================================================

    try:

        df = _query_public_api(
            series_id=
                series_id,
            start=
                start,
            end=
                end,
        )

        if not df.empty:

            return df

    except Exception as exc:

        errors.append(
            "API pública: "
            + str(exc)
        )

    # ========================================================
    # INTENTO 2
    # A5
    # ========================================================

    try:

        df = _query_a5(
            series_id=
                series_id,
            start=
                start,
            end=
                end,
        )

        if not df.empty:

            return df

    except Exception as exc:

        errors.append(
            "A5: "
            + str(exc)
        )

    raise RuntimeError(
        "No fue posible recuperar observaciones. "
        + " | ".join(
            errors
        )
    )


# ============================================================
# FUNCIÓN PRINCIPAL DE APP.PY
# ============================================================

def observed(
    start,
    end,
):

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        series_id = int(
            info[
                "series_id"
            ]
        )

        df = get_series(
            start=start,
            end=end,
            series_id=series_id,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "El INA no devolvió observaciones "
                    "para San Nicolás en el período seleccionado."
                ),
            )

        df[
            "station"
        ] = TARGET_STATION

        df[
            "series_id"
        ] = series_id

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

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        series_id = info.get(
            "series_id"
        )

    except Exception:

        series_id = None

    return {
        "fuente":
            "Instituto Nacional del Agua (INA)",

        "servicio":
            "Web API pública INA / respaldo A5",

        "estacion":
            TARGET_STATION,

        "serie":
            series_id,

        "variable":
            "Nivel hidrométrico",

        "unidad":
            "metros",

        "estado":
            "Consulta online",

        "observacion":
            (
                "La aplicación consulta primero la API "
                "pública del INA y utiliza A5 como respaldo."
            ),
    }
