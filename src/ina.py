import unicodedata

import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
# V11.0 - CONSULTA INA CORREGIDA
# ============================================================

INA_SERIES_URL = "https://alerta.ina.gob.ar/pub/datos/series"
INA_DATA_URL = "https://alerta.ina.gob.ar/pub/datos/datos"

REQUEST_TIMEOUT = 60

TARGET_STATION = "San Nicolás"
VAR_ID_LEVEL = 2

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
# UTILIDADES
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


def normalizar_fecha(value):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(dt):
        raise ValueError(
            f"Fecha inválida: {value}"
        )

    return dt.strftime("%Y-%m-%d")


# ============================================================
# REQUEST
# ============================================================

def request_json(
    url,
    params=None,
    timeout=REQUEST_TIMEOUT,
):

    try:

        response = requests.get(
            url,
            params=params,
            timeout=timeout,
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
            f"INA respondió HTTP {response.status_code}. "
            f"URL: {response.url}"
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió sin contenido."
        )

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"URL: {response.url} · "
            f"Respuesta: {response.text[:300]}"
        ) from exc

    return (
        data,
        response.url,
    )


# ============================================================
# EXTRAER LISTA
# ============================================================

def extraer_lista(data):

    if data is None:
        return []

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in [
        "data",
        "datos",
        "observaciones",
        "results",
        "result",
        "items",
    ]:

        value = data.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):

            nested = extraer_lista(
                value
            )

            if nested:
                return nested

    for value in data.values():

        if isinstance(value, list):
            return value

        if isinstance(value, dict):

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

    data, url_final = request_json(
        INA_SERIES_URL
    )

    catalogo = extraer_lista(
        data
    )

    if not catalogo:

        raise RuntimeError(
            "El catálogo del INA no devolvió series. "
            f"URL: {url_final}"
        )

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

        if not isinstance(row, dict):
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
            row.get("to_date"),
            errors="coerce",
            utc=True,
        )

        score = 0

        if procid == 1:
            score += 100

        elif procid == 2:
            score += 70

        if obs_count > 0:
            score += 30

        if pd.notna(to_date):

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
                "to_date":
                    row.get(
                        "to_date"
                    ),
                "score":
                    score,
            }
        )

    if not candidatos:

        raise RuntimeError(
            "No se encontró una serie activa "
            "de nivel para San Nicolás."
        )

    candidatos = sorted(
        candidatos,
        key=lambda item:
            item["score"],
        reverse=True,
    )

    return candidatos[0]


# ============================================================
# DETECTAR COLUMNAS
# ============================================================

def detectar_fecha(df):

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

        for col in df.columns:

            if (
                str(col).lower()
                == candidato.lower()
            ):
                return col

    for col in df.columns:

        name = str(col).lower()

        if (
            "fecha" in name
            or "date" in name
            or "time" in name
        ):
            return col

    return None


def detectar_valor(df):

    candidatos = [
        "valor",
        "value",
        "nivel",
        "altura",
        "level",
        "height",
    ]

    for candidato in candidatos:

        for col in df.columns:

            if (
                str(col).lower()
                == candidato.lower()
            ):
                return col

    for col in df.columns:

        name = str(col).lower()

        if (
            "valor" in name
            or "value" in name
            or "nivel" in name
            or "altura" in name
        ):
            return col

    return None


# ============================================================
# NORMALIZAR OBSERVACIONES
# ============================================================

def normalizar_observaciones(
    registros,
):

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

    fecha_col = detectar_fecha(
        raw
    )

    valor_col = detectar_valor(
        raw
    )

    if fecha_col is None:

        raise RuntimeError(
            "INA devolvió datos, pero no se encontró "
            f"la columna de fecha. Columnas: {list(raw.columns)}"
        )

    if valor_col is None:

        raise RuntimeError(
            "INA devolvió datos, pero no se encontró "
            f"la columna de nivel. Columnas: {list(raw.columns)}"
        )

    result = pd.DataFrame()

    result["datetime"] = pd.to_datetime(
        raw[fecha_col],
        errors="coerce",
        utc=True,
    )

    values = (
        raw[valor_col]
        .astype(str)
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    result["value"] = pd.to_numeric(
        values,
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
# CONSULTAR DATOS INA
# ============================================================

def consultar_serie(
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

    if (
        pd.to_datetime(start_text)
        >
        pd.to_datetime(end_text)
    ):

        raise ValueError(
            "La fecha Desde no puede ser posterior a Hasta."
        )

    # ========================================================
    # PARAMETROS EXACTOS
    # ========================================================

    params = {
        "timeStart":
            start_text,

        "timeEnd":
            end_text,

        "seriesId":
            int(series_id),

        "format":
            "json",
    }

    data, url_final = request_json(
        INA_DATA_URL,
        params=params,
    )

    # ========================================================
    # SI INA ENVIA ERROR
    # ========================================================

    if isinstance(data, dict):

        mensaje = (
            data.get("mensaje")
            or data.get("message")
            or data.get("error")
        )

        if mensaje:

            raise RuntimeError(
                f"{mensaje} · URL enviada: {url_final}"
            )

    registros = extraer_lista(
        data
    )

    if not registros:

        raise RuntimeError(
            "INA respondió correctamente pero "
            "no devolvió observaciones. "
            f"URL enviada: {url_final}"
        )

    df = normalizar_observaciones(
        registros
    )

    if df.empty:

        raise RuntimeError(
            "INA devolvió registros, pero no quedaron "
            "niveles válidos después de procesarlos. "
            f"URL enviada: {url_final}"
        )

    return (
        df,
        url_final,
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

    df, url_final = consultar_serie(
        series_id=series_id,
        start=start,
        end=end,
    )

    return df


# ============================================================
# OBSERVED
# ============================================================

def observed(
    start,
    end,
):

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        series_id = info[
            "series_id"
        ]

        df, url_final = consultar_serie(
            series_id=series_id,
            start=start,
            end=end,
        )

        df["station"] = (
            TARGET_STATION
        )

        df["series_id"] = (
            series_id
        )

        df["variable"] = (
            "Nivel hidrométrico"
        )

        df["unit"] = "m"

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

        series_id = (
            info.get(
                "series_id"
            )
        )

    except Exception:

        series_id = None

    return {
        "fuente":
            "Instituto Nacional del Agua (INA)",

        "servicio":
            "Web API pública INA",

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
    }
