import requests
import pandas as pd
import numpy as np
import unicodedata

from datetime import (
    datetime,
    timedelta,
)


# ============================================================
# PARANÁ · SAN NICOLÁS
# src/upstream.py
# V11.8
#
# OBJETIVOS
#
# - Descubrir series INA automáticamente.
# - No inventar series_id.
# - Consultar observaciones por INA A5.
# - Recuperar Corrientes con histórico ampliado.
# - Mantener el resto de estaciones para el período elegido.
# - Evitar que el fallo de una estación detenga toda la app.
#
# IMPORTANTE:
# San Nicolás NO se consulta aquí.
# San Nicolás continúa usando src/ina.py serie 36.
# ============================================================


CATALOG_URL = (
    "https://alerta.ina.gob.ar/pub/datos/series"
)

A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)


# Variable nivel
VAR_ID_NIVEL = 2


# ============================================================
# HISTÓRICO CORRIENTES
#
# Se intenta analizar un período mucho mayor que el rango
# seleccionado por el usuario.
#
# 1990 es solamente un límite inicial de consulta.
# INA devolverá lo que realmente tenga disponible.
# ============================================================


CORRIENTES_HISTORY_START = (
    "1990-01-01"
)


# ============================================================
# ESTACIONES
# ============================================================


STATIONS = {
    "Corrientes": [
        "Corrientes",
    ],

    "Goya": [
        "Goya",
    ],

    "La Paz": [
        "La Paz",
        "La Paz Entre Rios",
        "La Paz Entre Ríos",
    ],

    "Paraná": [
        "Paraná",
        "Parana",
    ],

    "Diamante": [
        "Diamante",
    ],

    "Rosario": [
        "Rosario",
    ],

    "Villa Constitución": [
        "Villa Constitución",
        "Villa Constitucion",
    ],
}


OUTPUT_COLUMNS = {
    "Corrientes":
        "nivel_corrientes",

    "Goya":
        "nivel_goya",

    "La Paz":
        "nivel_la_paz",

    "Paraná":
        "nivel_parana",

    "Diamante":
        "nivel_diamante",

    "Rosario":
        "nivel_rosario",

    "Villa Constitución":
        "nivel_villa_constitucion",
}


# ============================================================
# SESSION HTTP
# ============================================================


SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Parana-San-Nicolas-App/11.8",

        "Accept":
            "application/json",
    }
)


# ============================================================
# NORMALIZACIÓN DE TEXTO
# ============================================================


def normalizar_texto(
    value,
):

    if value is None:
        return ""

    text = str(
        value
    ).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(
            ch
        )
    )

    text = (
        text
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# FECHAS
# ============================================================


def normalizar_fecha(
    value,
):

    try:

        return pd.Timestamp(
            value
        ).strftime(
            "%Y-%m-%d"
        )

    except Exception:

        return str(
            value
        )[:10]


# ============================================================
# UTILIDADES JSON
# ============================================================


def _recursive_records(
    data,
):

    """
    Busca listas de diccionarios dentro de una respuesta JSON.
    """

    if isinstance(
        data,
        list,
    ):

        if (
            data
            and all(
                isinstance(
                    item,
                    dict,
                )
                for item in data
            )
        ):

            return data

        for item in data:

            found = (
                _recursive_records(
                    item
                )
            )

            if found:
                return found


    if isinstance(
        data,
        dict,
    ):

        common_keys = [
            "rows",
            "data",
            "series",
            "observaciones",
            "observations",
            "result",
            "results",
            "items",
            "features",
        ]

        for key in common_keys:

            if key in data:

                found = (
                    _recursive_records(
                        data[
                            key
                        ]
                    )
                )

                if found:
                    return found

        for value in data.values():

            found = (
                _recursive_records(
                    value
                )
            )

            if found:
                return found

    return []


# ============================================================
# EXTRAER CAMPO
# ============================================================


def _first_value(
    obj,
    keys,
):

    if not isinstance(
        obj,
        dict,
    ):

        return None

    for key in keys:

        if key in obj:

            value = obj[
                key
            ]

            if value is not None:

                return value

    return None


# ============================================================
# EXTRAER NOMBRE ESTACIÓN
# ============================================================


def _extract_station_name(
    record,
):

    direct = _first_value(
        record,
        [
            "site_name",
            "siteName",
            "sitename",
            "station_name",
            "stationName",
            "nombre_estacion",
            "estacion",
            "nombre",
            "name",
        ],
    )

    if isinstance(
        direct,
        str,
    ):

        return direct


    # --------------------------------------------------------
    # Posibles estructuras anidadas
    # --------------------------------------------------------

    for key in [
        "site",
        "station",
        "estacion",
    ]:

        nested = record.get(
            key
        )

        if isinstance(
            nested,
            dict,
        ):

            value = _first_value(
                nested,
                [
                    "name",
                    "nombre",
                    "site_name",
                    "station_name",
                ],
            )

            if value is not None:

                return value

    return ""


# ============================================================
# EXTRAER SERIES ID
# ============================================================


def _extract_series_id(
    record,
):

    value = _first_value(
        record,
        [
            "series_id",
            "seriesId",
            "seriesid",
            "id",
            "serie_id",
            "serieId",
        ],
    )

    try:

        return int(
            value
        )

    except Exception:

        return None


# ============================================================
# EXTRAER VAR ID
# ============================================================


def _extract_var_id(
    record,
):

    value = _first_value(
        record,
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

        pass


    variable = record.get(
        "variable"
    )

    if isinstance(
        variable,
        dict,
    ):

        value = _first_value(
            variable,
            [
                "id",
                "var_id",
                "varId",
            ],
        )

        try:

            return int(
                value
            )

        except Exception:

            pass

    return None


# ============================================================
# EXTRAER PROCEDIMIENTO
# ============================================================


def _extract_proc_id(
    record,
):

    value = _first_value(
        record,
        [
            "proc_id",
            "procId",
            "procid",
            "procedure_id",
            "procedureId",
        ],
    )

    try:

        return int(
            value
        )

    except Exception:

        return None


# ============================================================
# EXTRAER FECHA FINAL
# ============================================================


def _extract_end_date(
    record,
):

    value = _first_value(
        record,
        [
            "date_end",
            "dateEnd",
            "to_date",
            "toDate",
            "time_end",
            "timeEnd",
            "end_date",
            "endDate",
            "max_date",
            "maxDate",
        ],
    )

    try:

        return pd.Timestamp(
            value
        )

    except Exception:

        return pd.NaT


# ============================================================
# EXTRAER CANTIDAD DE OBSERVACIONES
# ============================================================


def _extract_obs_count(
    record,
):

    value = _first_value(
        record,
        [
            "obs_count",
            "obsCount",
            "count",
            "nobs",
            "observations_count",
            "observaciones",
        ],
    )

    try:

        return int(
            value
        )

    except Exception:

        return 0


# ============================================================
# CONSULTAR CATÁLOGO
# ============================================================


def descargar_catalogo():

    params = {
        "format":
            "json",
    }

    response = SESSION.get(
        CATALOG_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    records = (
        _recursive_records(
            data
        )
    )

    if not records:

        raise RuntimeError(
            "INA no devolvió registros "
            "en el catálogo de series."
        )

    return records


# ============================================================
# COINCIDENCIA DE NOMBRE
# ============================================================


def _station_match_score(
    station_name,
    aliases,
):

    station_norm = (
        normalizar_texto(
            station_name
        )
    )

    if not station_norm:
        return 0

    alias_norms = [
        normalizar_texto(
            alias
        )
        for alias in aliases
    ]

    score = 0


    for alias in alias_norms:

        # Exacta
        if (
            station_norm
            == alias
        ):

            score = max(
                score,
                100,
            )


        # Empieza exactamente
        elif station_norm.startswith(
            alias + " "
        ):

            score = max(
                score,
                90,
            )


        # Alias contenido
        elif alias in station_norm:

            score = max(
                score,
                80,
            )


        # Nombre dentro del alias
        elif station_norm in alias:

            score = max(
                score,
                70,
            )

    return score


# ============================================================
# SELECCIONAR CANDIDATOS
# ============================================================


def buscar_series_estacion(
    catalog_records,
    station,
):

    aliases = STATIONS[
        station
    ]

    candidates = []


    for record in catalog_records:

        if not isinstance(
            record,
            dict,
        ):

            continue


        station_name = (
            _extract_station_name(
                record
            )
        )

        match_score = (
            _station_match_score(
                station_name,
                aliases,
            )
        )

        if match_score <= 0:

            continue


        series_id = (
            _extract_series_id(
                record
            )
        )

        if series_id is None:

            continue


        var_id = (
            _extract_var_id(
                record
            )
        )


        # ----------------------------------------------------
        # Si el catálogo informa varId y no es nivel,
        # se descarta.
        # ----------------------------------------------------

        if (
            var_id is not None
            and var_id
            != VAR_ID_NIVEL
        ):

            continue


        proc_id = (
            _extract_proc_id(
                record
            )
        )

        end_date = (
            _extract_end_date(
                record
            )
        )

        obs_count = (
            _extract_obs_count(
                record
            )
        )


        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        score = float(
            match_score
        )


        # Preferencia procedimiento observado
        if proc_id == 1:

            score += 20

        elif proc_id == 2:

            score += 10


        # Preferir series activas/recientes
        if pd.notna(
            end_date
        ):

            days_old = (
                pd.Timestamp.today()
                .normalize()
                - end_date.normalize()
            ).days

            if days_old <= 30:

                score += 20

            elif days_old <= 365:

                score += 10

            elif days_old <= 3650:

                score += 3


        # Preferir series con observaciones
        if obs_count > 0:

            score += min(
                np.log10(
                    obs_count + 1
                )
                * 3,
                15,
            )


        candidates.append(
            {
                "series_id":
                    series_id,

                "station_name":
                    station_name,

                "var_id":
                    var_id,

                "proc_id":
                    proc_id,

                "end_date":
                    end_date,

                "obs_count":
                    obs_count,

                "score":
                    score,

                "record":
                    record,
            }
        )


    candidates = sorted(
        candidates,
        key=lambda x:
            x[
                "score"
            ],
        reverse=True,
    )


    # Evitar ids repetidos
    unique = []

    seen = set()

    for candidate in candidates:

        sid = candidate[
            "series_id"
        ]

        if sid in seen:
            continue

        seen.add(
            sid
        )

        unique.append(
            candidate
        )


    return unique


# ============================================================
# NORMALIZAR OBSERVACIONES A5
# ============================================================


def normalizar_observaciones(
    data,
):

    records = (
        _recursive_records(
            data
        )
    )

    if not records:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )


    rows = []


    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue


        dt = _first_value(
            record,
            [
                "timestart",
                "timeStart",
                "timestamp",
                "datetime",
                "date",
                "fecha",
                "time",
            ],
        )


        value = _first_value(
            record,
            [
                "valor",
                "value",
                "obs_value",
                "observation",
                "nivel",
            ],
        )


        # A veces viene:
        # {"timestart": ..., "valor": ...}
        # o estructuras similares.
        if (
            dt is None
            or value is None
        ):

            continue


        rows.append(
            {
                "datetime":
                    dt,

                "value":
                    value,
            }
        )


    if not rows:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )


    df = pd.DataFrame(
        rows
    )


    df[
        "datetime"
    ] = pd.to_datetime(
        df[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )


    df[
        "value"
    ] = pd.to_numeric(
        df[
            "value"
        ],
        errors="coerce",
    )


    df = df.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )


    # Niveles plausibles
    df = df[
        (
            df[
                "value"
            ]
            >= -5
        )
        &
        (
            df[
                "value"
            ]
            <= 20
        )
    ]


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


    return df


# ============================================================
# CONSULTAR A5
# ============================================================


def consultar_a5(
    series_id,
    start,
    end,
):

    params = {
        "tipo":
            "puntual",

        "series_id":
            int(
                series_id
            ),

        "timestart":
            normalizar_fecha(
                start
            ),

        "timeend":
            normalizar_fecha(
                end
            ),
    }


    response = SESSION.get(
        A5_URL,
        params=params,
        timeout=45,
    )


    if response.status_code != 200:

        raise RuntimeError(
            "INA A5 HTTP "
            f"{response.status_code}"
        )


    data = response.json()


    return normalizar_observaciones(
        data
    )


# ============================================================
# CONSULTA EN BLOQUES
# ============================================================


def consultar_a5_en_bloques(
    series_id,
    start,
    end,
    block_years=5,
):

    """
    Para históricos grandes divide la consulta en bloques.

    Esto evita pedir décadas completas en una sola llamada.
    """

    start_ts = pd.Timestamp(
        start
    ).normalize()

    end_ts = pd.Timestamp(
        end
    ).normalize()


    if start_ts > end_ts:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )


    frames = []

    current = start_ts


    while current <= end_ts:

        block_end = min(
            current
            + pd.DateOffset(
                years=block_years
            )
            - pd.Timedelta(
                days=1
            ),
            end_ts,
        )


        try:

            part = consultar_a5(
                series_id=
                    series_id,

                start=
                    current,

                end=
                    block_end,
            )


            if not part.empty:

                frames.append(
                    part
                )


        except Exception:

            # Una ventana fallida no invalida
            # automáticamente todo el histórico.
            pass


        current = (
            block_end
            + pd.Timedelta(
                days=1
            )
        )


    if not frames:

        return pd.DataFrame(
            columns=[
                "datetime",
                "value",
            ]
        )


    df = pd.concat(
        frames,
        ignore_index=True,
    )


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


    return df


# ============================================================
# NORMALIZACIÓN DIARIA
# ============================================================


def convertir_diario(
    df,
    output_column,
):

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )


    x = df[
        [
            "datetime",
            "value",
        ]
    ].copy()


    x[
        "datetime"
    ] = pd.to_datetime(
        x[
            "datetime"
        ],
        errors="coerce",
        utc=True,
    )


    x[
        "value"
    ] = pd.to_numeric(
        x[
            "value"
        ],
        errors="coerce",
    )


    x = x.dropna(
        subset=[
            "datetime",
            "value",
        ]
    )


    if x.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                output_column,
            ]
        )


    x[
        "date"
    ] = (
        x[
            "datetime"
        ]
        .dt.floor(
            "D"
        )
    )


    # Para nivel, usamos mediana diaria.
    daily = (
        x.groupby(
            "date",
            as_index=False,
        )[
            "value"
        ]
        .median()
        .rename(
            columns={
                "date":
                    "datetime",

                "value":
                    output_column,
            }
        )
    )


    daily = (
        daily
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    return daily


# ============================================================
# VERIFICAR CANDIDATO
# ============================================================


def verificar_candidato(
    candidate,
    start,
    end,
):

    """
    Hace una consulta corta para verificar que la serie
    realmente devuelve observaciones.
    """

    series_id = candidate[
        "series_id"
    ]


    end_ts = pd.Timestamp(
        end
    )


    # Ventana de validación.
    test_start = max(
        pd.Timestamp(
            start
        ),
        end_ts
        - pd.Timedelta(
            days=120
        ),
    )


    try:

        df = consultar_a5(
            series_id=
                series_id,

            start=
                test_start,

            end=
                end_ts,
        )

        return (
            not df.empty,
            len(
                df
            ),
        )


    except Exception:

        return (
            False,
            0,
        )


# ============================================================
# ELEGIR SERIE
# ============================================================


def seleccionar_serie(
    catalog_records,
    station,
    start,
    end,
):

    candidates = (
        buscar_series_estacion(
            catalog_records,
            station,
        )
    )


    if not candidates:

        return (
            None,
            []
        )


    checked = []


    # Solo probamos los mejores candidatos
    # para no hacer demasiadas llamadas.
    for candidate in candidates[
        :5
    ]:

        ok, test_records = (
            verificar_candidato(
                candidate,
                start,
                end,
            )
        )


        candidate = (
            candidate.copy()
        )

        candidate[
            "verified"
        ] = ok

        candidate[
            "test_records"
        ] = test_records


        checked.append(
            candidate
        )


        if ok:

            return (
                candidate,
                checked,
            )


    return (
        None,
        checked,
    )


# ============================================================
# OBTENER UNA ESTACIÓN
# ============================================================


def obtener_estacion(
    catalog_records,
    station,
    start,
    end,
):

    output_column = (
        OUTPUT_COLUMNS[
            station
        ]
    )


    selected, checked = (
        seleccionar_serie(
            catalog_records=
                catalog_records,

            station=
                station,

            start=
                start,

            end=
                end,
        )
    )


    if selected is None:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    output_column,
                ]
            ),
            {
                "status":
                    "sin_serie",

                "series_id":
                    None,

                "records":
                    0,

                "candidates_checked":
                    len(
                        checked
                    ),
            },
        )


    series_id = (
        selected[
            "series_id"
        ]
    )


    # ========================================================
    # CORRIENTES
    #
    # Histórico extendido independiente del rango elegido.
    # ========================================================

    if station == "Corrientes":

        query_start = (
            CORRIENTES_HISTORY_START
        )

        query_end = (
            end
        )


        df_obs = (
            consultar_a5_en_bloques(
                series_id=
                    series_id,

                start=
                    query_start,

                end=
                    query_end,

                block_years=5,
            )
        )


    else:

        query_start = (
            start
        )

        query_end = (
            end
        )


        try:

            df_obs = (
                consultar_a5(
                    series_id=
                        series_id,

                    start=
                        query_start,

                    end=
                        query_end,
                )
            )

        except Exception:

            df_obs = (
                pd.DataFrame(
                    columns=[
                        "datetime",
                        "value",
                    ]
                )
            )


    daily = (
        convertir_diario(
            df_obs,
            output_column,
        )
    )


    meta = {
        "status":
            (
                "ok"
                if not daily.empty
                else "sin_datos"
            ),

        "series_id":
            series_id,

        "station_name":
            selected.get(
                "station_name",
                station,
            ),

        "var_id":
            selected.get(
                "var_id"
            ),

        "proc_id":
            selected.get(
                "proc_id"
            ),

        "records":
            len(
                daily
            ),

        "query_start":
            normalizar_fecha(
                query_start
            ),

        "query_end":
            normalizar_fecha(
                query_end
            ),

        "catalog_score":
            selected.get(
                "score"
            ),

        "candidates_checked":
            len(
                checked
            ),
    }


    if not daily.empty:

        meta[
            "first_date"
        ] = (
            daily[
                "datetime"
            ]
            .min()
            .strftime(
                "%Y-%m-%d"
            )
        )

        meta[
            "last_date"
        ] = (
            daily[
                "datetime"
            ]
            .max()
            .strftime(
                "%Y-%m-%d"
            )
        )


    return (
        daily,
        meta,
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================


def get_upstream_history(
    start,
    end,
):

    """
    Devuelve:

    (
        upstream_history,
        metadata
    )

    upstream_history contiene:

    datetime
    nivel_corrientes
    nivel_goya
    nivel_la_paz
    nivel_parana
    nivel_diamante
    nivel_rosario
    nivel_villa_constitucion

    Corrientes puede contener un historial mucho más largo
    que las demás estaciones.
    """


    start = normalizar_fecha(
        start
    )

    end = normalizar_fecha(
        end
    )


    # ========================================================
    # CATÁLOGO
    # ========================================================

    try:

        catalog_records = (
            descargar_catalogo()
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            {
                "_error":
                    (
                        "No fue posible descargar "
                        "el catálogo INA: "
                        + str(
                            exc
                        )
                    )
            },
        )


    station_frames = []

    metadata = {}


    # ========================================================
    # CADA ESTACIÓN
    # ========================================================

    for station in STATIONS:

        try:

            df_station, meta = (
                obtener_estacion(
                    catalog_records=
                        catalog_records,

                    station=
                        station,

                    start=
                        start,

                    end=
                        end,
                )
            )


        except Exception as exc:

            df_station = (
                pd.DataFrame()
            )

            meta = {
                "status":
                    "error",

                "series_id":
                    None,

                "records":
                    0,

                "error":
                    str(
                        exc
                    ),
            }


        metadata[
            station
        ] = meta


        if (
            isinstance(
                df_station,
                pd.DataFrame,
            )
            and not df_station.empty
        ):

            station_frames.append(
                df_station
            )


    # ========================================================
    # UNIR ESTACIONES
    # ========================================================

    if not station_frames:

        return (
            pd.DataFrame(),
            metadata,
        )


    result = (
        station_frames[0]
        .copy()
    )


    for frame in station_frames[
        1:
    ]:

        result = result.merge(
            frame,
            on="datetime",
            how="outer",
        )


    result = (
        result
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # ASEGURAR TODAS LAS COLUMNAS
    # ========================================================

    for col in OUTPUT_COLUMNS.values():

        if col not in result.columns:

            result[
                col
            ] = np.nan


    ordered_cols = [
        "datetime",
        "nivel_corrientes",
        "nivel_goya",
        "nivel_la_paz",
        "nivel_parana",
        "nivel_diamante",
        "nivel_rosario",
        "nivel_villa_constitucion",
    ]


    result = result[
        ordered_cols
    ]


    return (
        result,
        metadata,
    )


# ============================================================
# DIAGNÓSTICO OPCIONAL
# ============================================================


def diagnostic(
    start,
    end,
):

    df, meta = (
        get_upstream_history(
            start,
            end,
        )
    )


    summary = []


    for station in STATIONS:

        info = (
            meta.get(
                station,
                {}
            )
        )


        summary.append(
            {
                "station":
                    station,

                "status":
                    info.get(
                        "status"
                    ),

                "series_id":
                    info.get(
                        "series_id"
                    ),

                "records":
                    info.get(
                        "records",
                        0,
                    ),

                "first_date":
                    info.get(
                        "first_date"
                    ),

                "last_date":
                    info.get(
                        "last_date"
                    ),

                "query_start":
                    info.get(
                        "query_start"
                    ),

                "query_end":
                    info.get(
                        "query_end"
                    ),
            }
        )


    return {
        "rows":
            len(
                df
            ),

        "stations":
            pd.DataFrame(
                summary
            ),

        "metadata":
            meta,
    }
