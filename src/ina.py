import unicodedata

import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
# BASE V11.0 - CONSULTA INA ESTABLE
# ============================================================


INA_SERIES_URL = (
    "https://alerta.ina.gob.ar/pub/datos/series"
)

INA_DATA_URL = (
    "https://alerta.ina.gob.ar/pub/datos/datos"
)

REQUEST_TIMEOUT = 60

VAR_ID_LEVEL = 2

TARGET_STATION = "San Nicolás"


# ============================================================
# ESTACIONES UTILIZADAS
# ============================================================

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


# ============================================================
# CACHE CATÁLOGO
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
        caracter
        for caracter in texto
        if not unicodedata.combining(
            caracter
        )
    )

    return (
        texto
        .strip()
        .lower()
    )


# ============================================================
# NORMALIZACIÓN DE FECHA
# ============================================================

def normalizar_fecha(value):

    fecha = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(fecha):

        raise ValueError(
            f"Fecha inválida: {value}"
        )

    return fecha.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# REQUEST JSON
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
                "User-Agent":
                    "Parana-San-Nicolas-V11/1.0",

                "Accept":
                    "application/json,text/plain,*/*",
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

    try:

        response.raise_for_status()

    except requests.HTTPError as exc:

        raise RuntimeError(
            "INA respondió HTTP "
            f"{response.status_code}."
        ) from exc

    if not response.text.strip():

        raise RuntimeError(
            "INA respondió sin contenido."
        )

    try:

        return response.json()

    except ValueError as exc:

        raise RuntimeError(
            "INA no devolvió JSON válido. "
            f"Respuesta inicial: {response.text[:500]}"
        ) from exc


# ============================================================
# OBTENER CATÁLOGO INA
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

    except Exception as exc:

        raise RuntimeError(
            "No fue posible obtener el catálogo "
            f"de series del INA: {exc}"
        ) from exc

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

        # Buscar cualquier lista dentro
        # de la respuesta.
        for value in data.values():

            if isinstance(
                value,
                list,
            ):

                catalogo = value
                break

    else:

        catalogo = []

    if not catalogo:

        raise RuntimeError(
            "El catálogo del INA respondió "
            "pero no contiene series."
        )

    _CATALOG_CACHE = catalogo

    return catalogo


# ============================================================
# BUSCAR MEJOR SERIE DE NIVEL
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

        # ----------------------------------------------------
        # VARIABLE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SERIES ID
        # ----------------------------------------------------

        series_id = (
            row.get(
                "seriesid"
            )
            or row.get(
                "seriesId"
            )
            or row.get(
                "series_id"
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
        # ÚLTIMA FECHA DISPONIBLE
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

            score += 80

        if obs_count > 0:

            score += 40

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

                score += 80

            elif age_days <= 30:

                score += 60

            elif age_days <= 90:

                score += 40

            elif age_days <= 365:

                score += 20

        candidatos.append(
            {
                "station":
                    estacion,

                "series_id":
                    series_id,

                "varid":
                    varid,

                "procid":
                    procid,

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
# EXTRAER LISTA DE OBSERVACIONES
# ============================================================

def extraer_observaciones(data):

    if data is None:

        return []

    # --------------------------------------------------------
    # RESPUESTA DIRECTAMENTE COMO LISTA
    # --------------------------------------------------------

    if isinstance(
        data,
        list,
    ):

        return data

    # --------------------------------------------------------
    # RESPUESTA COMO DICT
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict,
    ):

        return []

    # Mensajes de API
    mensaje = (
        data.get(
            "mensaje"
        )
        or data.get(
            "message"
        )
        or data.get(
            "error"
        )
    )

    if mensaje:

        mensaje_texto = str(
            mensaje
        )

        # Si es solo un mensaje
        # informativo pero hay datos,
        # seguimos buscando.
        claves_datos = [
            "data",
            "datos",
            "observaciones",
            "results",
            "result",
            "items",
        ]

        hay_datos = any(
            isinstance(
                data.get(
                    clave
                ),
                list,
            )
            for clave
            in claves_datos
        )

        if not hay_datos:

            raise RuntimeError(
                f"INA: {mensaje_texto}"
            )

    # --------------------------------------------------------
    # CLAVES HABITUALES
    # --------------------------------------------------------

    for clave in [
        "data",
        "datos",
        "observaciones",
        "results",
        "result",
        "items",
    ]:

        value = data.get(
            clave
        )

        if isinstance(
            value,
            list,
        ):

            return value

    # --------------------------------------------------------
    # BÚSQUEDA RECURSIVA
    # --------------------------------------------------------

    for value in data.values():

        if isinstance(
            value,
            list,
        ):

            if value:

                return value

        if isinstance(
            value,
            dict,
        ):

            nested = extraer_observaciones(
                value
            )

            if nested:

                return nested

    return []


# ============================================================
# DETECTAR COLUMNA DE FECHA
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
        "timeStamp",
        "fecha_hora",
        "fechaHora",
        "fecha",
        "date",
        "time",
        "observedAt",
        "observed_at",
    ]

    columnas = list(
        df.columns
    )

    # coincidencia exacta
    for candidato in candidatos:

        for columna in columnas:

            if (
                str(
                    columna
                ).lower()
                ==
                candidato.lower()
            ):

                return columna

    # coincidencia parcial
    for columna in columnas:

        nombre = str(
            columna
        ).lower()

        if (
            "fecha"
            in nombre
            or "date"
            in nombre
            or "time"
            in nombre
        ):

            return columna

    return None


# ============================================================
# DETECTAR COLUMNA DE NIVEL
# ============================================================

def detectar_columna_valor(
    df,
):

    candidatos = [
        "valor",
        "value",
        "nivel",
        "level",
        "altura",
        "height",
        "dato",
        "measurement",
        "observed",
    ]

    columnas = list(
        df.columns
    )

    # coincidencia exacta
    for candidato in candidatos:

        for columna in columnas:

            if (
                str(
                    columna
                ).lower()
                ==
                candidato.lower()
            ):

                return columna

    # coincidencia parcial
    for columna in columnas:

        nombre = str(
            columna
        ).lower()

        if (
            "valor"
            in nombre
            or "value"
            in nombre
            or "nivel"
            in nombre
            or "altura"
            in nombre
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

    # --------------------------------------------------------
    # MUY IMPORTANTE:
    # nombres EXACTOS exigidos por /pub/datos/datos
    # --------------------------------------------------------

    start_text = normalizar_fecha(
        start
    )

    end_text = normalizar_fecha(
        end
    )

    if (
        pd.to_datetime(
            start_text
        )
        >
        pd.to_datetime(
            end_text
        )
    ):

        raise ValueError(
            "La fecha inicial no puede "
            "ser posterior a la final."
        )

    params = {
        "timeStart":
            start_text,

        "timeEnd":
            end_text,

        "seriesId":
            int(
                series_id
            ),

        "format":
            "json",
    }

    # ========================================================
    # CONSULTA
    # ========================================================

    data = request_json(
        INA_DATA_URL,
        params=params,
    )

    registros = (
        extraer_observaciones(
            data
        )
    )

    if not registros:

        return pd.DataFrame()

    # ========================================================
    # CONVERTIR
    # ========================================================

    try:

        df = pd.json_normalize(
            registros
        )

    except Exception:

        df = pd.DataFrame(
            registros
        )

    if df.empty:

        return pd.DataFrame()

    # ========================================================
    # DETECTAR COLUMNAS
    # ========================================================

    fecha_col = (
        detectar_columna_fecha(
            df
        )
    )

    valor_col = (
        detectar_columna_valor(
            df
        )
    )

    if fecha_col is None:

        raise RuntimeError(
            "INA devolvió registros pero "
            "no se encontró la fecha. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    if valor_col is None:

        raise RuntimeError(
            "INA devolvió registros pero "
            "no se encontró el nivel. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    # ========================================================
    # NORMALIZAR
    # ========================================================

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

    # Permitir coma decimal
    raw_values = (
        df[
            valor_col
        ]
        .astype(
            str
        )
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    result[
        "value"
    ] = pd.to_numeric(
        raw_values,
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

    # ========================================================
    # FILTRO DE NIVEL
    # ========================================================

    # Permitimos margen amplio porque
    # esta función debe recuperar datos,
    # no recortarlos a la escala visual.
    result = result[
        result[
            "value"
        ].between(
            -5.0,
            20.0,
        )
    ].copy()

    if result.empty:

        return pd.DataFrame()

    # ========================================================
    # ORDEN
    # ========================================================

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
# GET SERIES
# ============================================================

def get_series(
    start,
    end,
    series_id=None,
):

    # ========================================================
    # SI NO SE INDICA SERIE,
    # BUSCAR AUTOMÁTICAMENTE
    # ========================================================

    if series_id is None:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

        if info is None:

            raise RuntimeError(
                "No fue posible localizar "
                "una serie activa de nivel "
                "para San Nicolás."
            )

        series_id = info[
            "series_id"
        ]

    return consultar_serie(
        series_id=
            series_id,
        start=
            start,
        end=
            end,
    )


# ============================================================
# FUNCIÓN UTILIZADA POR APP.PY
# ============================================================

def observed(
    start,
    end,
):

    try:

        # ====================================================
        # BUSCAR SERIE REAL EN CATÁLOGO
        # ====================================================

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

        # ====================================================
        # CONSULTAR DATOS
        # ====================================================

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
                pd.DataFrame(),
                (
                    "El INA no devolvió "
                    "observaciones para "
                    "San Nicolás en el período "
                    "seleccionado. "
                    f"Serie consultada: "
                    f"{info['series_id']}."
                ),
            )

        # ====================================================
        # METADATOS
        # ====================================================

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
# METADATA
# ============================================================

def forecast_meta():

    info = None

    try:

        info = buscar_serie_nivel(
            TARGET_STATION
        )

    except Exception:

        info = None

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

        "observacion":
            (
                "La aplicación localiza automáticamente "
                "la serie hidrométrica vigente de "
                "San Nicolás en el catálogo del INA y "
                "consulta las observaciones mediante "
                "timeStart, timeEnd y seriesId."
            ),
    }
