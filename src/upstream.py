import unicodedata

import pandas as pd
import requests


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.3 - ESTACIONES AGUAS ARRIBA CON INA A5
# ============================================================

INA_A5_URL = "https://alerta.ina.gob.ar/a5/getObservaciones"

REQUEST_TIMEOUT = 60

VAR_ID_LEVEL = 2


# ============================================================
# ESTACIONES
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
# SERIES CONOCIDAS / RESPALDO
#
# IMPORTANTE:
# San Nicolás = 36 está confirmado en ina.py.
#
# Para las estaciones aguas arriba NO inventamos series.
# Primero intentamos obtenerlas dinámicamente del catálogo A5.
# ============================================================

SERIES_CACHE = {}

CATALOG_CACHE = None


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

    return (
        text
        .strip()
        .lower()
    )


# ============================================================
# NOMBRE DE COLUMNA
# ============================================================

def nombre_columna_estacion(estacion):

    name = normalizar_texto(
        estacion
    )

    name = (
        name
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"nivel_{name}"


# ============================================================
# NORMALIZAR FECHA
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
# REQUEST GENERAL
# ============================================================

def request_json(
    url,
    params=None,
):

    try:

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept":
                    "application/json,text/plain,*/*",

                "User-Agent":
                    "Parana-San-Nicolas-V11.3/1.0",
            },
        )

    except requests.Timeout as exc:

        raise RuntimeError(
            "INA demoró demasiado en responder."
        ) from exc

    except requests.ConnectionError as exc:

        raise RuntimeError(
            "No fue posible conectarse con INA."
        ) from exc

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Error de comunicación INA: {exc}"
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
            f"Respuesta: {response.text[:300]}"
        ) from exc

    return (
        data,
        response.url,
        response.status_code,
    )


# ============================================================
# EXTRAER LISTAS
# ============================================================

def extraer_lista(data):

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
    # CLAVES HABITUALES
    # --------------------------------------------------------

    for key in [
        "observaciones",
        "series",
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

            nested = extraer_lista(
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

        elif isinstance(
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
# EXTRAER MENSAJE DE ERROR
# ============================================================

def extraer_error(data):

    if not isinstance(
        data,
        dict,
    ):
        return None

    for key in [
        "error",
        "mensaje",
        "message",
        "detail",
    ]:

        value = data.get(
            key
        )

        if value:

            return str(
                value
            )

    return None


# ============================================================
# ENDPOINTS POSIBLES PARA CATÁLOGO A5
# ============================================================

def obtener_catalogo():

    global CATALOG_CACHE

    if CATALOG_CACHE is not None:
        return CATALOG_CACHE

    # --------------------------------------------------------
    # A5 puede exponer las series mediante distintos recursos.
    # Probamos solamente recursos de catálogo.
    # --------------------------------------------------------

    urls = [
        "https://alerta.ina.gob.ar/a5/series",
        "https://alerta.ina.gob.ar/a5/getSeries",
        "https://alerta.ina.gob.ar/a5/getSeriesPuntuales",
    ]

    for url in urls:

        try:

            data, _, _ = request_json(
                url
            )

            rows = extraer_lista(
                data
            )

            if rows:

                CATALOG_CACHE = rows

                return rows

        except Exception:

            continue

    # --------------------------------------------------------
    # Si A5 no ofrece catálogo utilizable,
    # devolvemos vacío.
    #
    # IMPORTANTE:
    # Esto NO afecta San Nicolás.
    # --------------------------------------------------------

    CATALOG_CACHE = []

    return []


# ============================================================
# BUSCAR VALOR EN DICT
# ============================================================

def buscar_valor_dict(
    row,
    keys,
):

    if not isinstance(
        row,
        dict,
    ):
        return None

    # --------------------------------------------------------
    # EXACTO
    # --------------------------------------------------------

    for key in keys:

        if key in row:

            value = row.get(
                key
            )

            if value not in [
                None,
                "",
            ]:

                return value

    # --------------------------------------------------------
    # IGNORANDO MAYÚSCULAS
    # --------------------------------------------------------

    lowered = {
        str(k).lower():
            v
        for k, v
        in row.items()
    }

    for key in keys:

        value = lowered.get(
            str(key).lower()
        )

        if value not in [
            None,
            "",
        ]:

            return value

    return None


# ============================================================
# DETECTAR NOMBRE DE ESTACIÓN
# ============================================================

def obtener_nombre_estacion(row):

    value = buscar_valor_dict(
        row,
        [
            "estacion_nombre",
            "estacion",
            "station_name",
            "station",
            "nombre",
            "sitio",
            "site_name",
            "siteName",
        ],
    )

    if isinstance(
        value,
        dict,
    ):

        value = (
            value.get("nombre")
            or value.get("name")
            or value.get("estacion")
        )

    return value


# ============================================================
# DETECTAR SERIES ID
# ============================================================

def obtener_series_id(row):

    value = buscar_valor_dict(
        row,
        [
            "series_id",
            "seriesId",
            "seriesid",
            "serie_id",
            "serieId",
            "id_serie",
            "id",
        ],
    )

    if isinstance(
        value,
        dict,
    ):

        value = (
            value.get("id")
            or value.get("series_id")
            or value.get("seriesId")
        )

    try:

        return int(
            value
        )

    except Exception:

        return None


# ============================================================
# DETECTAR VARIABLE
# ============================================================

def obtener_var_id(row):

    value = buscar_valor_dict(
        row,
        [
            "var_id",
            "varId",
            "varid",
            "variable_id",
            "variableId",
        ],
    )

    try:

        return int(
            value
        )

    except Exception:

        return None


# ============================================================
# DETECTAR FECHA FINAL
# ============================================================

def obtener_fecha_final(row):

    value = buscar_valor_dict(
        row,
        [
            "to_date",
            "fecha_hasta",
            "fecha_fin",
            "end_date",
            "timeend",
            "timeEnd",
            "ultima_fecha",
        ],
    )

    return pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )


# ============================================================
# BUSCAR SERIE DE NIVEL
# ============================================================

def buscar_serie_nivel(
    estacion,
):

    if estacion in SERIES_CACHE:

        return SERIES_CACHE[
            estacion
        ]

    catalogo = obtener_catalogo()

    if not catalogo:

        SERIES_CACHE[
            estacion
        ] = None

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

        nombre = obtener_nombre_estacion(
            row
        )

        nombre_norm = normalizar_texto(
            nombre
        )

        # ----------------------------------------------------
        # Coincidencia flexible
        # ----------------------------------------------------

        if (
            objetivo
            not in nombre_norm
            and nombre_norm
            not in objetivo
        ):
            continue

        series_id = obtener_series_id(
            row
        )

        if series_id is None:
            continue

        var_id = obtener_var_id(
            row
        )

        # Si conocemos la variable,
        # exigimos nivel = 2.
        if (
            var_id is not None
            and var_id != VAR_ID_LEVEL
        ):
            continue

        fecha_final = obtener_fecha_final(
            row
        )

        score = 0

        # Coincidencia exacta
        if nombre_norm == objetivo:
            score += 100

        # Variable confirmada nivel
        if var_id == VAR_ID_LEVEL:
            score += 100

        # Priorizar serie reciente
        if pd.notna(
            fecha_final
        ):

            ahora = pd.Timestamp.now(
                tz="UTC"
            )

            dias = (
                ahora
                - fecha_final
            ).days

            if dias <= 7:
                score += 100

            elif dias <= 30:
                score += 80

            elif dias <= 90:
                score += 60

            elif dias <= 365:
                score += 30

        candidatos.append(
            {
                "station":
                    estacion,

                "station_ina":
                    nombre,

                "series_id":
                    series_id,

                "var_id":
                    var_id,

                "to_date":
                    (
                        fecha_final.isoformat()
                        if pd.notna(
                            fecha_final
                        )
                        else None
                    ),

                "score":
                    score,

                "source":
                    "INA A5 catalog",
            }
        )

    if not candidatos:

        SERIES_CACHE[
            estacion
        ] = None

        return None

    candidatos = sorted(
        candidatos,
        key=lambda item:
            item["score"],
        reverse=True,
    )

    best = candidatos[
        0
    ]

    SERIES_CACHE[
        estacion
    ] = best

    return best


# ============================================================
# CONSULTAR OBSERVACIONES A5
# ============================================================

def consultar_a5(
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
        pd.to_datetime(
            start_text
        )
        >
        pd.to_datetime(
            end_text
        )
    ):

        raise ValueError(
            "La fecha inicial no puede ser posterior a la final."
        )

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

    data, url_final, status = (
        request_json(
            INA_A5_URL,
            params=params,
        )
    )

    error = extraer_error(
        data
    )

    # Solo considerar error si no hay observaciones.
    registros = extraer_lista(
        data
    )

    if error and not registros:

        raise RuntimeError(
            f"{error}"
        )

    return (
        registros,
        url_final,
        status,
    )


# ============================================================
# DETECTAR FECHA
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

    for candidato in candidatos:

        for columna in columnas:

            if (
                str(columna).lower()
                ==
                candidato.lower()
            ):

                return columna

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
# DETECTAR NIVEL
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

    for candidato in candidatos:

        for columna in columnas:

            if (
                str(columna).lower()
                ==
                candidato.lower()
            ):

                return columna

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

    # --------------------------------------------------------
    # FILTRO DE SEGURIDAD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convertir a fecha diaria
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Promedio diario
    # --------------------------------------------------------

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
# OBTENER HISTÓRICO DE UNA ESTACIÓN
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
            {
                "station":
                    estacion,

                "series_id":
                    None,

                "status":
                    "Serie no encontrada",

                "records":
                    0,

                "error":
                    None,
            },
        )

    series_id = info[
        "series_id"
    ]

    try:

        (
            registros,
            url_final,
            http_status,
        ) = consultar_a5(
            series_id=
                series_id,
            start=
                start,
            end=
                end,
        )

        df = normalizar_observaciones(
            registros
        )

        metadata = dict(
            info
        )

        metadata[
            "http_status"
        ] = http_status

        metadata[
            "url_final"
        ] = url_final

        metadata[
            "records"
        ] = len(
            df
        )

        metadata[
            "status"
        ] = (
            "OK"
            if not df.empty
            else "Sin observaciones"
        )

        metadata[
            "error"
        ] = None

        if df.empty:

            return (
                pd.DataFrame(),
                metadata,
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
            metadata,
        )

    except Exception as exc:

        metadata = dict(
            info
        )

        metadata[
            "status"
        ] = "Error"

        metadata[
            "records"
        ] = 0

        metadata[
            "error"
        ] = str(
            exc
        )

        return (
            pd.DataFrame(),
            metadata,
        )


# ============================================================
# HISTÓRICO AGUAS ARRIBA
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
            station_meta,
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
        ] = station_meta

        # ----------------------------------------------------
        # IMPORTANTE:
        # Una estación sin datos NO bloquea las demás.
        # ----------------------------------------------------

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

            resultado = (
                resultado.merge(
                    station_df,
                    on="datetime",
                    how="outer",
                )
            )

    if resultado is None:

        resultado = pd.DataFrame(
            columns=[
                "datetime"
            ]
        )

    resultado[
        "datetime"
    ] = pd.to_datetime(
        resultado[
            "datetime"
        ],
        errors="coerce",
    )

    resultado = (
        resultado
        .dropna(
            subset=[
                "datetime"
            ]
        )
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
# RESUMEN
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

        columna = nombre_columna_estacion(
            estacion
        )

        if (
            columna
            not in upstream_history.columns
        ):

            continue

        temp = upstream_history[
            [
                "datetime",
                columna,
            ]
        ].copy()

        temp[
            columna
        ] = pd.to_numeric(
            temp[
                columna
            ],
            errors="coerce",
        )

        temp = (
            temp
            .dropna(
                subset=[
                    columna
                ]
            )
            .sort_values(
                "datetime"
            )
        )

        if temp.empty:

            continue

        nivel_actual = float(
            temp[
                columna
            ].iloc[-1]
        )

        nivel_anterior = None
        variacion = None

        if len(temp) >= 2:

            nivel_anterior = float(
                temp[
                    columna
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


# ============================================================
# DIAGNÓSTICO
# ============================================================

def diagnostic_upstream(
    start,
    end,
):

    rows = []

    for estacion in (
        UPSTREAM_STATIONS
    ):

        (
            df,
            metadata,
        ) = get_station_history(
            estacion,
            start,
            end,
        )

        metadata = (
            metadata
            if isinstance(
                metadata,
                dict,
            )
            else {}
        )

        rows.append(
            {
                "Estación":
                    estacion,

                "Serie INA":
                    metadata.get(
                        "series_id"
                    ),

                "Estado":
                    metadata.get(
                        "status"
                    ),

                "HTTP":
                    metadata.get(
                        "http_status"
                    ),

                "Registros":
                    (
                        len(df)
                        if isinstance(
                            df,
                            pd.DataFrame,
                        )
                        else 0
                    ),

                "Error":
                    metadata.get(
                        "error"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )
