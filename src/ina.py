import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/ina.py
# V11.0 - INA A5 ESTABLE
# ============================================================

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

REQUEST_TIMEOUT = 60

SAN_NICOLAS_SERIES_ID = 36

TARGET_STATION = "San Nicolás"


# ============================================================
# ESTACIONES
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
# FECHAS
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
# REQUEST A5
# ============================================================

def consultar_a5(
    start,
    end,
    series_id=SAN_NICOLAS_SERIES_ID,
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

    params = {
        "tipo":
            "puntual",

        "series_id":
            int(series_id),

        "timestart":
            start_text,

        "timeend":
            end_text,
    }

    try:

        response = requests.get(
            INA_A5_URL,
            params=params,
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
            f"INA A5 respondió HTTP {response.status_code}. "
            f"URL: {response.url}"
        )

    if not response.text.strip():

        raise RuntimeError(
            "INA A5 respondió sin contenido."
        )

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "INA A5 no devolvió JSON válido. "
            f"Respuesta inicial: {response.text[:300]}"
        ) from exc

    return (
        data,
        response.url,
        response.status_code,
    )


# ============================================================
# EXTRAER REGISTROS
# ============================================================

def extraer_registros(data):

    if data is None:
        return []

    # --------------------------------------------------------
    # RESPUESTA DIRECTA COMO LISTA
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

    # --------------------------------------------------------
    # MENSAJES DE ERROR
    # --------------------------------------------------------

    mensaje = (
        data.get("mensaje")
        or data.get("message")
        or data.get("error")
    )

    # --------------------------------------------------------
    # CLAVES HABITUALES
    # --------------------------------------------------------

    for key in [
        "observaciones",
        "data",
        "datos",
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

            nested = extraer_registros(
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

            if value:
                return value

        if isinstance(
            value,
            dict,
        ):

            nested = extraer_registros(
                value
            )

            if nested:
                return nested

    if mensaje:

        raise RuntimeError(
            str(mensaje)
        )

    return []


# ============================================================
# DETECTAR COLUMNA FECHA
# ============================================================

def detectar_columna_fecha(df):

    candidatos = [
        "timestart",
        "timeStart",
        "datetime",
        "dateTime",
        "timestamp",
        "fecha_hora",
        "fechaHora",
        "fecha",
        "date",
        "time",
    ]

    columnas = list(
        df.columns
    )

    # Coincidencia exacta
    for candidato in candidatos:

        for columna in columnas:

            if (
                str(columna).lower()
                ==
                candidato.lower()
            ):

                return columna

    # Coincidencia parcial
    for columna in columnas:

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
# DETECTAR COLUMNA NIVEL
# ============================================================

def detectar_columna_valor(df):

    candidatos = [
        "valor",
        "value",
        "nivel",
        "altura",
        "level",
        "height",
        "measurement",
    ]

    columnas = list(
        df.columns
    )

    # Coincidencia exacta
    for candidato in candidatos:

        for columna in columnas:

            if (
                str(columna).lower()
                ==
                candidato.lower()
            ):

                return columna

    # Coincidencia parcial
    for columna in columnas:

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

    fecha_col = detectar_columna_fecha(
        raw
    )

    valor_col = detectar_columna_valor(
        raw
    )

    if fecha_col is None:

        raise RuntimeError(
            "INA A5 devolvió registros, pero no se encontró "
            f"la columna de fecha. Columnas: {list(raw.columns)}"
        )

    if valor_col is None:

        raise RuntimeError(
            "INA A5 devolvió registros, pero no se encontró "
            f"la columna de nivel. Columnas: {list(raw.columns)}"
        )

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

    raw_values = (
        raw[
            valor_col
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
        raw_values,
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "datetime",
            "value",
        ]
    ).copy()

    if result.empty:
        return pd.DataFrame()

    # Rango amplio de seguridad.
    # La escala gráfica sigue siendo 0-7 m,
    # pero no cortamos prematuramente los datos.
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
    series_id=SAN_NICOLAS_SERIES_ID,
):

    data, url_final, http_status = consultar_a5(
        start=start,
        end=end,
        series_id=series_id,
    )

    registros = extraer_registros(
        data
    )

    if not registros:

        raise RuntimeError(
            "INA A5 respondió correctamente, pero no "
            "devolvió observaciones para el período seleccionado. "
            f"HTTP: {http_status} · URL: {url_final}"
        )

    df = normalizar_observaciones(
        registros
    )

    if df.empty:

        raise RuntimeError(
            "INA A5 devolvió registros, pero no quedaron "
            "niveles válidos después del procesamiento. "
            f"URL: {url_final}"
        )

    return df


# ============================================================
# FUNCIÓN UTILIZADA POR APP.PY
# ============================================================

def observed(
    start,
    end,
):

    try:

        df = get_series(
            start=start,
            end=end,
            series_id=
                SAN_NICOLAS_SERIES_ID,
        )

        if df.empty:

            return (
                pd.DataFrame(),
                (
                    "INA A5 no devolvió datos válidos "
                    "para San Nicolás."
                ),
            )

        df[
            "station"
        ] = TARGET_STATION

        df[
            "series_id"
        ] = (
            SAN_NICOLAS_SERIES_ID
        )

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
# DIAGNÓSTICO
# ============================================================

def diagnostic(
    start,
    end,
):

    info = {
        "endpoint":
            INA_A5_URL,

        "series_id":
            SAN_NICOLAS_SERIES_ID,

        "tipo":
            "puntual",

        "desde":
            normalizar_fecha(
                start
            ),

        "hasta":
            normalizar_fecha(
                end
            ),

        "http_status":
            None,

        "registros":
            0,

        "error":
            None,

        "url_final":
            None,

        "json_tipo":
            None,

        "json_claves":
            None,

        "columnas_detectadas":
            None,

        "primer_registro":
            None,

        "respuesta_preview":
            None,
    }

    try:

        (
            data,
            url_final,
            http_status,
        ) = consultar_a5(
            start=start,
            end=end,
            series_id=
                SAN_NICOLAS_SERIES_ID,
        )

        info[
            "http_status"
        ] = http_status

        info[
            "url_final"
        ] = url_final

        info[
            "json_tipo"
        ] = type(
            data
        ).__name__

        if isinstance(
            data,
            dict,
        ):

            info[
                "json_claves"
            ] = list(
                data.keys()
            )

        elif isinstance(
            data,
            list,
        ):

            info[
                "json_claves"
            ] = [
                "respuesta_raiz_lista"
            ]

        info[
            "respuesta_preview"
        ] = str(
            data
        )[:2000]

        registros = extraer_registros(
            data
        )

        info[
            "registros"
        ] = len(
            registros
        )

        if registros:

            info[
                "primer_registro"
            ] = str(
                registros[
                    0
                ]
            )[:1000]

            try:

                temp = pd.json_normalize(
                    registros
                )

                info[
                    "columnas_detectadas"
                ] = list(
                    temp.columns
                )

            except Exception:

                info[
                    "columnas_detectadas"
                ] = []

        return info

    except Exception as exc:

        info[
            "error"
        ] = str(
            exc
        )

        return info


# ============================================================
# META
# ============================================================

def forecast_meta():

    return {
        "fuente":
            "Instituto Nacional del Agua (INA)",

        "servicio":
            "INA A5 · getObservaciones",

        "estacion":
            TARGET_STATION,

        "serie":
            SAN_NICOLAS_SERIES_ID,

        "variable":
            "Nivel hidrométrico",

        "unidad":
            "metros",

        "estado":
            "Consulta online",

        "observacion":
            (
                "La plataforma consulta las observaciones "
                "hidrométricas de San Nicolás mediante "
                "INA A5 utilizando tipo puntual, "
                "series_id 36, timestart y timeend."
            ),
    }
