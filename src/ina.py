import time
import requests
import pandas as pd


# ============================================================
# CONFIGURACIÓN INA
# ============================================================

# Endpoint principal A5
INA_A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)

# Endpoint legado de respaldo
INA_LEGACY_URL = (
    "https://alerta.ina.gob.ar/pub/datos/datos"
)


# ============================================================
# SAN NICOLÁS
# ============================================================

# Serie utilizada actualmente para San Nicolás
SERIES_ID = 36

TIPO = "puntual"


# ============================================================
# CONFIGURACIÓN DE RED
# ============================================================

REQUEST_TIMEOUT = 45

MAX_RETRIES = 3

RETRY_WAIT_SECONDS = 2


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "User-Agent":
        "Parana-San-Nicolas-V11/1.0",

    "Accept":
        "application/json,text/plain,*/*",

    "Connection":
        "close",
}


# ============================================================
# NORMALIZAR FECHA
# ============================================================

def _normalizar_fecha(
    value,
):

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        dt
    ):

        return None

    return dt.strftime(
        "%Y-%m-%d"
    )


# ============================================================
# REQUEST CON REINTENTOS
# ============================================================

def _request_json(
    url,
    params,
):

    ultimo_error = None

    for intento in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )

            response.raise_for_status()

            data = response.json()

            return (
                data,
                None,
            )

        except requests.exceptions.Timeout:

            ultimo_error = (
                "Tiempo de espera agotado "
                "al consultar INA."
            )

        except requests.exceptions.ConnectionError:

            ultimo_error = (
                "No fue posible establecer conexión "
                "con el servidor del INA."
            )

        except requests.exceptions.HTTPError as exc:

            status = None

            try:

                status = (
                    exc.response.status_code
                )

            except Exception:

                pass

            if status is not None:

                ultimo_error = (
                    "INA respondió con error HTTP "
                    f"{status}."
                )

            else:

                ultimo_error = (
                    "INA respondió con un error HTTP."
                )

        except ValueError:

            ultimo_error = (
                "INA respondió, pero el contenido "
                "no pudo interpretarse como JSON."
            )

        except Exception as exc:

            ultimo_error = (
                "Error inesperado consultando INA: "
                f"{exc}"
            )

        if intento < MAX_RETRIES:

            time.sleep(
                RETRY_WAIT_SECONDS
            )

    return (
        None,
        ultimo_error,
    )


# ============================================================
# PARSEAR RESPUESTA A5
# ============================================================

def _parse_a5(
    data,
):

    if not isinstance(
        data,
        list,
    ):

        return pd.DataFrame()

    if len(
        data
    ) == 0:

        return pd.DataFrame()

    df = pd.DataFrame(
        data
    )

    if df.empty:

        return pd.DataFrame()

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

    if df.empty:

        return pd.DataFrame()

    # Orden cronológico
    df = (
        df
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

    return df[
        [
            "datetime",
            "value",
        ]
    ]


# ============================================================
# PARSEAR RESPUESTA LEGACY
# ============================================================

def _parse_legacy(
    data,
):

    if data is None:

        return pd.DataFrame()

    # ========================================================
    # CASO LISTA
    # ========================================================

    if isinstance(
        data,
        list,
    ):

        df = pd.DataFrame(
            data
        )

    # ========================================================
    # CASO DICCIONARIO
    # ========================================================

    elif isinstance(
        data,
        dict,
    ):

        lista = None

        # Buscar la primera lista disponible
        for value in data.values():

            if isinstance(
                value,
                list,
            ):

                lista = value
                break

        if lista is None:

            return pd.DataFrame()

        df = pd.DataFrame(
            lista
        )

    else:

        return pd.DataFrame()

    if df.empty:

        return pd.DataFrame()

    # ========================================================
    # DETECTAR FECHA
    # ========================================================

    date_col = None

    for candidate in [
        "timestart",
        "timeStart",
        "datetime",
        "fecha",
        "timestamp",
    ]:

        if candidate in df.columns:

            date_col = candidate
            break

    # ========================================================
    # DETECTAR VALOR
    # ========================================================

    value_col = None

    for candidate in [
        "valor",
        "value",
        "nivel",
    ]:

        if candidate in df.columns:

            value_col = candidate
            break

    if (
        date_col is None
        or value_col is None
    ):

        return pd.DataFrame()

    df[
        "datetime"
    ] = pd.to_datetime(
        df[
            date_col
        ],
        errors="coerce",
        utc=True,
    )

    df[
        "value"
    ] = pd.to_numeric(
        df[
            value_col
        ],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )

    if df.empty:

        return pd.DataFrame()

    df = (
        df
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

    return df[
        [
            "datetime",
            "value",
        ]
    ]


# ============================================================
# CONSULTAR ENDPOINT A5
# ============================================================

def _observed_a5(
    start,
    end,
):

    params = {
        "tipo":
            TIPO,

        "series_id":
            SERIES_ID,

        "timestart":
            start,

        "timeend":
            end,
    }

    (
        data,
        error,
    ) = _request_json(
        INA_A5_URL,
        params=params,
    )

    if error is not None:

        return (
            pd.DataFrame(),
            error,
        )

    df = _parse_a5(
        data
    )

    if df.empty:

        return (
            pd.DataFrame(),
            (
                "INA A5 respondió correctamente, "
                "pero no devolvió observaciones válidas."
            ),
        )

    return (
        df,
        None,
    )


# ============================================================
# CONSULTAR ENDPOINT LEGACY
# ============================================================

def _observed_legacy(
    start,
    end,
):

    params = {
        "seriesId":
            SERIES_ID,

        "timeStart":
            start,

        "timeEnd":
            end,
    }

    (
        data,
        error,
    ) = _request_json(
        INA_LEGACY_URL,
        params=params,
    )

    if error is not None:

        return (
            pd.DataFrame(),
            error,
        )

    df = _parse_legacy(
        data
    )

    if df.empty:

        return (
            pd.DataFrame(),
            (
                "El endpoint alternativo del INA "
                "no devolvió observaciones válidas."
            ),
        )

    return (
        df,
        None,
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def observed(
    start,
    end,
):

    # ========================================================
    # NORMALIZAR FECHAS
    # ========================================================

    start = _normalizar_fecha(
        start
    )

    end = _normalizar_fecha(
        end
    )

    if start is None:

        return (
            pd.DataFrame(),
            "La fecha inicial no es válida.",
        )

    if end is None:

        return (
            pd.DataFrame(),
            "La fecha final no es válida.",
        )

    if (
        pd.Timestamp(
            start
        )
        > pd.Timestamp(
            end
        )
    ):

        return (
            pd.DataFrame(),
            (
                "La fecha inicial no puede ser "
                "posterior a la fecha final."
            ),
        )

    # ========================================================
    # PRIMER INTENTO: A5
    # ========================================================

    (
        df,
        error_a5,
    ) = _observed_a5(
        start,
        end,
    )

    if not df.empty:

        return (
            df,
            None,
        )

    # ========================================================
    # SEGUNDO INTENTO: LEGACY
    # ========================================================

    (
        df_legacy,
        error_legacy,
    ) = _observed_legacy(
        start,
        end,
    )

    if not df_legacy.empty:

        return (
            df_legacy,
            None,
        )

    # ========================================================
    # ERROR FINAL
    # ========================================================

    mensaje = (
        "No fue posible obtener datos del INA."
    )

    if error_a5:

        mensaje += (
            " A5: "
            + str(
                error_a5
            )
        )

    if error_legacy:

        mensaje += (
            " Alternativo: "
            + str(
                error_legacy
            )
        )

    return (
        pd.DataFrame(),
        mensaje,
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostico(
    start,
    end,
):

    start = _normalizar_fecha(
        start
    )

    end = _normalizar_fecha(
        end
    )

    resultado = {
        "series_id":
            SERIES_ID,

        "tipo":
            TIPO,

        "inicio":
            start,

        "fin":
            end,

        "a5_ok":
            False,

        "legacy_ok":
            False,

        "a5_registros":
            0,

        "legacy_registros":
            0,

        "a5_error":
            None,

        "legacy_error":
            None,
    }

    if (
        start is None
        or end is None
    ):

        resultado[
            "a5_error"
        ] = "Fechas inválidas."

        return resultado

    # ========================================================
    # A5
    # ========================================================

    (
        df_a5,
        error_a5,
    ) = _observed_a5(
        start,
        end,
    )

    resultado[
        "a5_error"
    ] = error_a5

    resultado[
        "a5_registros"
    ] = len(
        df_a5
    )

    resultado[
        "a5_ok"
    ] = (
        not df_a5.empty
    )

    # ========================================================
    # LEGACY
    # ========================================================

    (
        df_legacy,
        error_legacy,
    ) = _observed_legacy(
        start,
        end,
    )

    resultado[
        "legacy_error"
    ] = error_legacy

    resultado[
        "legacy_registros"
    ] = len(
        df_legacy
    )

    resultado[
        "legacy_ok"
    ] = (
        not df_legacy.empty
    )

    return resultado
