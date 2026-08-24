import requests
import numpy as np
import pandas as pd

from datetime import timedelta


# ============================================================
# PARANÁ · SAN NICOLÁS
# UPSTREAM V11.9
# ============================================================

UPSTREAM_VERSION = "V11.9"


# ============================================================
# INA A5
# ============================================================

INA_A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)


# ============================================================
# ESTACIONES AGUAS ARRIBA
#
# IMPORTANTE:
# Mantener acá los series_id de NIVEL que ya estén confirmados
# en tu instalación.
#
# Si alguno de estos ID en tu versión anterior era distinto,
# conservar el ID que ya te funcionaba.
# ============================================================

UPSTREAM_STATIONS = {

    "Corrientes": {
        "key": "corrientes",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -27.4692,
        "lon": -58.8306,
    },

    "Goya": {
        "key": "goya",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -29.1400,
        "lon": -59.2634,
    },

    "La Paz": {
        "key": "la_paz",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -30.7449,
        "lon": -59.6457,
    },

    "Paraná": {
        "key": "parana",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -31.7413,
        "lon": -60.5115,
    },

    "Diamante": {
        "key": "diamante",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -32.0664,
        "lon": -60.6380,
    },

    "Rosario": {
        "key": "rosario",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -32.9442,
        "lon": -60.6505,
    },

    "Villa Constitución": {
        "key": "villa_constitucion",
        "nivel_series_id": None,
        "caudal_series_id": None,
        "lat": -33.2272,
        "lon": -60.3294,
    },
}


# ============================================================
# CONFIGURACIÓN OPEN-METEO
# ============================================================

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# TIMEOUT
# ============================================================

HTTP_TIMEOUT = 30


# ============================================================
# UTILIDADES
# ============================================================

def _to_datetime(
    serie,
):

    return pd.to_datetime(
        serie,
        errors="coerce",
        utc=True,
    )


def _to_numeric(
    serie,
):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _normalizar_diario(
    df,
    value_col,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime" not in df.columns
        or value_col not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                value_col,
            ]
        )

    x = df[
        [
            "datetime",
            value_col,
        ]
    ].copy()

    x["datetime"] = _to_datetime(
        x["datetime"]
    )

    x[value_col] = _to_numeric(
        x[value_col]
    )

    x = x.dropna(
        subset=[
            "datetime",
            value_col,
        ]
    )

    if x.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                value_col,
            ]
        )

    # Convertir a día calendario.
    x["datetime"] = (
        x["datetime"]
        .dt
        .tz_convert(
            "America/Argentina/Buenos_Aires"
        )
        .dt
        .tz_localize(None)
        .dt
        .normalize()
    )

    x = (
        x
        .groupby(
            "datetime",
            as_index=False,
        )[value_col]
        .mean()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return x


# ============================================================
# INTERPRETAR RESPUESTA INA
# ============================================================

def _parse_ina_response(
    payload,
):

    if payload is None:

        return pd.DataFrame()

    observations = None

    # ========================================================
    # FORMATO DIRECTO LISTA
    # ========================================================

    if isinstance(
        payload,
        list,
    ):

        observations = payload

    # ========================================================
    # FORMATOS DICT
    # ========================================================

    elif isinstance(
        payload,
        dict,
    ):

        for key in [
            "observaciones",
            "observations",
            "data",
            "datos",
        ]:

            if (
                key in payload
                and isinstance(
                    payload[key],
                    list,
                )
            ):

                observations = (
                    payload[key]
                )

                break

        # Algunos endpoints devuelven la lista
        # dentro de una estructura adicional.
        if observations is None:

            for value in payload.values():

                if isinstance(
                    value,
                    list,
                ):

                    if len(value) == 0:

                        continue

                    if isinstance(
                        value[0],
                        dict,
                    ):

                        observations = (
                            value
                        )

                        break

    if not observations:

        return pd.DataFrame()

    raw = pd.DataFrame(
        observations
    )

    if raw.empty:

        return pd.DataFrame()

    # ========================================================
    # ENCONTRAR FECHA
    # ========================================================

    datetime_col = None

    for candidate in [
        "timestart",
        "datetime",
        "fecha",
        "time",
        "date",
    ]:

        if candidate in raw.columns:

            datetime_col = (
                candidate
            )

            break

    # ========================================================
    # ENCONTRAR VALOR
    # ========================================================

    value_col = None

    for candidate in [
        "valor",
        "value",
        "nivel",
        "caudal",
    ]:

        if candidate in raw.columns:

            value_col = (
                candidate
            )

            break

    if (
        datetime_col is None
        or value_col is None
    ):

        return pd.DataFrame()

    return pd.DataFrame(
        {
            "datetime":
                raw[
                    datetime_col
                ],

            "value":
                raw[
                    value_col
                ],
        }
    )


# ============================================================
# CONSULTAR UNA SERIE INA
# ============================================================

def _get_ina_series(
    series_id,
    start_date,
    end_date,
):

    if series_id in [
        None,
        "",
        0,
    ]:

        return pd.DataFrame()

    params = {

        "tipo":
            "puntual",

        "series_id":
            int(
                series_id
            ),

        "timestart":
            str(
                start_date
            ),

        "timeend":
            str(
                end_date
            ),
    }

    try:

        response = requests.get(
            INA_A5_URL,
            params=params,
            timeout=HTTP_TIMEOUT,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

        return _parse_ina_response(
            payload
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# CONSULTAR NIVEL INA
# ============================================================

def _get_nivel_station(
    station,
    start_date,
    end_date,
):

    series_id = station.get(
        "nivel_series_id"
    )

    raw = _get_ina_series(
        series_id=
            series_id,

        start_date=
            start_date,

        end_date=
            end_date,
    )

    if raw.empty:

        return pd.DataFrame()

    key = station[
        "key"
    ]

    col = (
        f"nivel_{key}"
    )

    raw = raw.rename(
        columns={
            "value":
                col,
        }
    )

    return _normalizar_diario(
        raw,
        col,
    )


# ============================================================
# CONSULTAR CAUDAL INA
# ============================================================

def _get_caudal_station(
    station,
    start_date,
    end_date,
):

    series_id = station.get(
        "caudal_series_id"
    )

    raw = _get_ina_series(
        series_id=
            series_id,

        start_date=
            start_date,

        end_date=
            end_date,
    )

    if raw.empty:

        return pd.DataFrame()

    key = station[
        "key"
    ]

    col = (
        f"caudal_{key}"
    )

    raw = raw.rename(
        columns={
            "value":
                col,
        }
    )

    return _normalizar_diario(
        raw,
        col,
    )


# ============================================================
# LLUVIA HISTÓRICA POR ESTACIÓN
# OPEN-METEO
# ============================================================

def _get_rain_history_station(
    station,
    start_date,
    end_date,
):

    lat = station.get(
        "lat"
    )

    lon = station.get(
        "lon"
    )

    if (
        lat is None
        or lon is None
    ):

        return pd.DataFrame()

    params = {

        "latitude":
            lat,

        "longitude":
            lon,

        "start_date":
            str(
                start_date
            ),

        "end_date":
            str(
                end_date
            ),

        "daily":
            "precipitation_sum",

        "timezone":
            "America/Argentina/Buenos_Aires",
    }

    try:

        response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=HTTP_TIMEOUT,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

        daily = payload.get(
            "daily",
            {}
        )

        dates = daily.get(
            "time",
            []
        )

        rain = daily.get(
            "precipitation_sum",
            []
        )

        if (
            not dates
            or len(
                dates
            )
            != len(
                rain
            )
        ):

            return pd.DataFrame()

        key = station[
            "key"
        ]

        col = (
            f"lluvia_{key}"
        )

        x = pd.DataFrame(
            {
                "datetime":
                    dates,

                col:
                    rain,
            }
        )

        x[
            "datetime"
        ] = pd.to_datetime(
            x[
                "datetime"
            ],
            errors="coerce",
        )

        x[
            col
        ] = (
            pd.to_numeric(
                x[
                    col
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
        )

        return (
            x
            .dropna(
                subset=[
                    "datetime"
                ]
            )
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# LLUVIA FUTURA POR ESTACIÓN
# ============================================================

def _get_rain_forecast_station(
    station,
    forecast_days=15,
):

    lat = station.get(
        "lat"
    )

    lon = station.get(
        "lon"
    )

    if (
        lat is None
        or lon is None
    ):

        return pd.DataFrame()

    forecast_days = int(
        np.clip(
            forecast_days,
            1,
            16,
        )
    )

    params = {

        "latitude":
            lat,

        "longitude":
            lon,

        "daily":
            "precipitation_sum",

        "forecast_days":
            forecast_days,

        "timezone":
            "America/Argentina/Buenos_Aires",
    }

    try:

        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=HTTP_TIMEOUT,
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

        daily = payload.get(
            "daily",
            {}
        )

        dates = daily.get(
            "time",
            []
        )

        rain = daily.get(
            "precipitation_sum",
            []
        )

        if (
            not dates
            or len(
                dates
            )
            != len(
                rain
            )
        ):

            return pd.DataFrame()

        key = station[
            "key"
        ]

        col = (
            f"lluvia_{key}"
        )

        x = pd.DataFrame(
            {
                "datetime":
                    dates,

                col:
                    rain,
            }
        )

        x[
            "datetime"
        ] = pd.to_datetime(
            x[
                "datetime"
            ],
            errors="coerce",
        )

        x[
            col
        ] = (
            pd.to_numeric(
                x[
                    col
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .clip(
                lower=0.0
            )
        )

        return (
            x
            .dropna(
                subset=[
                    "datetime"
                ]
            )
            .sort_values(
                "datetime"
            )
            .reset_index(
                drop=True
            )
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# MERGE SEGURO
# ============================================================

def _merge_series(
    base,
    new_df,
):

    if (
        new_df is None
        or not isinstance(
            new_df,
            pd.DataFrame,
        )
        or new_df.empty
    ):

        return base

    if base is None:

        base = pd.DataFrame()

    if base.empty:

        return (
            new_df
            .copy()
        )

    return base.merge(
        new_df,
        on="datetime",
        how="outer",
    )


# ============================================================
# COMPLETAR HUECOS CORTOS
# ============================================================

def _interpolate_short_gaps(
    df,
):

    if (
        df is None
        or df.empty
    ):

        return df

    x = (
        df
        .copy()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # NIVEL
    # Máximo 3 días
    # ========================================================

    for col in [
        c
        for c in x.columns
        if c.startswith(
            "nivel_"
        )
    ]:

        x[
            col
        ] = pd.to_numeric(
            x[
                col
            ],
            errors="coerce",
        )

        x[
            col
        ] = (
            x[
                col
            ]
            .interpolate(
                limit=3,
                limit_direction="both",
            )
        )

    # ========================================================
    # CAUDAL
    # Máximo 3 días
    # ========================================================

    for col in [
        c
        for c in x.columns
        if c.startswith(
            "caudal_"
        )
    ]:

        x[
            col
        ] = pd.to_numeric(
            x[
                col
            ],
            errors="coerce",
        )

        x[
            col
        ] = (
            x[
                col
            ]
            .interpolate(
                limit=3,
                limit_direction="both",
            )
        )

    # ========================================================
    # LLUVIA
    # NO INTERPOLAR.
    #
    # Si Open-Meteo entrega NaN se conserva NaN.
    # No queremos inventar lluvia.
    # ========================================================

    for col in [
        c
        for c in x.columns
        if c.startswith(
            "lluvia_"
        )
    ]:

        x[
            col
        ] = pd.to_numeric(
            x[
                col
            ],
            errors="coerce",
        )

    return x


# ============================================================
# CREAR VARIABLES DERIVADAS POR ESTACIÓN
# ============================================================

def add_upstream_features(
    df,
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return pd.DataFrame()

    x = (
        df
        .copy()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    for station_name, station in (
        UPSTREAM_STATIONS.items()
    ):

        key = station[
            "key"
        ]

        nivel_col = (
            f"nivel_{key}"
        )

        caudal_col = (
            f"caudal_{key}"
        )

        lluvia_col = (
            f"lluvia_{key}"
        )

        # ====================================================
        # NIVEL
        # ====================================================

        if nivel_col in x.columns:

            x[
                f"{nivel_col}_diff1"
            ] = (
                x[
                    nivel_col
                ]
                .diff(
                    1
                )
            )

            x[
                f"{nivel_col}_diff3"
            ] = (
                x[
                    nivel_col
                ]
                .diff(
                    3
                )
            )

            x[
                f"{nivel_col}_trend7"
            ] = (
                x[
                    nivel_col
                ]
                - x[
                    nivel_col
                ].shift(
                    7
                )
            ) / 7.0

        # ====================================================
        # CAUDAL
        # ====================================================

        if caudal_col in x.columns:

            x[
                f"{caudal_col}_diff1"
            ] = (
                x[
                    caudal_col
                ]
                .diff(
                    1
                )
            )

            x[
                f"{caudal_col}_diff3"
            ] = (
                x[
                    caudal_col
                ]
                .diff(
                    3
                )
            )

            x[
                f"{caudal_col}_trend7"
            ] = (
                x[
                    caudal_col
                ]
                - x[
                    caudal_col
                ].shift(
                    7
                )
            ) / 7.0

        # ====================================================
        # LLUVIA ACUMULADA
        # ====================================================

        if lluvia_col in x.columns:

            rain = (
                pd.to_numeric(
                    x[
                        lluvia_col
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .clip(
                    lower=0.0
                )
            )

            x[
                f"{lluvia_col}_3d"
            ] = (
                rain
                .rolling(
                    3,
                    min_periods=1,
                )
                .sum()
            )

            x[
                f"{lluvia_col}_7d"
            ] = (
                rain
                .rolling(
                    7,
                    min_periods=1,
                )
                .sum()
            )

            x[
                f"{lluvia_col}_14d"
            ] = (
                rain
                .rolling(
                    14,
                    min_periods=1,
                )
                .sum()
            )

            x[
                f"{lluvia_col}_21d"
            ] = (
                rain
                .rolling(
                    21,
                    min_periods=1,
                )
                .sum()
            )

    return x


# ============================================================
# HISTÓRICO AGUAS ARRIBA
#
# Esta función conserva el nombre que ya usa app.py:
#
# get_upstream_history(...)
#
# ============================================================

def get_upstream_history(
    start_date,
    end_date,
):

    start_date = pd.Timestamp(
        start_date
    ).date()

    end_date = pd.Timestamp(
        end_date
    ).date()

    # ========================================================
    # CALENDARIO DIARIO
    # ========================================================

    history = pd.DataFrame(
        {
            "datetime":
                pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq="D",
                )
        }
    )

    meta = {
        "version":
            UPSTREAM_VERSION,

        "stations":
            {},

        "start_date":
            str(
                start_date
            ),

        "end_date":
            str(
                end_date
            ),
    }

    # ========================================================
    # CADA ESTACIÓN
    # ========================================================

    for station_name, station in (
        UPSTREAM_STATIONS.items()
    ):

        key = station[
            "key"
        ]

        # ====================================================
        # NIVEL
        # ====================================================

        nivel = _get_nivel_station(
            station=
                station,

            start_date=
                start_date,

            end_date=
                end_date,
        )

        history = _merge_series(
            history,
            nivel,
        )

        # ====================================================
        # CAUDAL
        # ====================================================

        caudal = _get_caudal_station(
            station=
                station,

            start_date=
                start_date,

            end_date=
                end_date,
        )

        history = _merge_series(
            history,
            caudal,
        )

        # ====================================================
        # LLUVIA HISTÓRICA
        # ====================================================

        lluvia = (
            _get_rain_history_station(
                station=
                    station,

                start_date=
                    start_date,

                end_date=
                    end_date,
            )
        )

        history = _merge_series(
            history,
            lluvia,
        )

        nivel_col = (
            f"nivel_{key}"
        )

        caudal_col = (
            f"caudal_{key}"
        )

        lluvia_col = (
            f"lluvia_{key}"
        )

        meta[
            "stations"
        ][station_name] = {

            "key":
                key,

            "nivel_series_id":
                station.get(
                    "nivel_series_id"
                ),

            "caudal_series_id":
                station.get(
                    "caudal_series_id"
                ),

            "lat":
                station.get(
                    "lat"
                ),

            "lon":
                station.get(
                    "lon"
                ),

            "nivel_disponible":
                (
                    nivel_col
                    in history.columns
                    and history[
                        nivel_col
                    ]
                    .notna()
                    .any()
                ),

            "caudal_disponible":
                (
                    caudal_col
                    in history.columns
                    and history[
                        caudal_col
                    ]
                    .notna()
                    .any()
                ),

            "lluvia_disponible":
                (
                    lluvia_col
                    in history.columns
                    and history[
                        lluvia_col
                    ]
                    .notna()
                    .any()
                ),
        }

    # ========================================================
    # LIMPIEZA FINAL
    # ========================================================

    history = (
        history
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

    history = _interpolate_short_gaps(
        history
    )

    return (
        history,
        meta,
    )


# ============================================================
# LLUVIA FUTURA AGUAS ARRIBA
# ============================================================

def get_upstream_forecast(
    forecast_days=15,
):

    forecast_days = int(
        np.clip(
            forecast_days,
            1,
            16,
        )
    )

    future = pd.DataFrame()

    meta = {
        "version":
            UPSTREAM_VERSION,

        "forecast_days":
            forecast_days,

        "stations":
            {},
    }

    for station_name, station in (
        UPSTREAM_STATIONS.items()
    ):

        rain = (
            _get_rain_forecast_station(
                station=
                    station,

                forecast_days=
                    forecast_days,
            )
        )

        future = _merge_series(
            future,
            rain,
        )

        key = station[
            "key"
        ]

        col = (
            f"lluvia_{key}"
        )

        meta[
            "stations"
        ][station_name] = {

            "key":
                key,

            "lluvia_disponible":
                (
                    col
                    in future.columns
                    and future[
                        col
                    ]
                    .notna()
                    .any()
                ),
        }

    if not future.empty:

        future = (
            future
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
        future,
        meta,
    )


# ============================================================
# TABLA RESUMEN PARA APP
# ============================================================

def build_upstream_summary(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
        or "datetime"
        not in upstream_history.columns
    ):

        return pd.DataFrame()

    x = (
        upstream_history
        .copy()
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    for station_name, station in (
        UPSTREAM_STATIONS.items()
    ):

        key = station[
            "key"
        ]

        nivel_col = (
            f"nivel_{key}"
        )

        caudal_col = (
            f"caudal_{key}"
        )

        lluvia_col = (
            f"lluvia_{key}"
        )

        # ====================================================
        # NIVEL
        # ====================================================

        nivel_actual = np.nan
        delta_nivel_1 = np.nan
        delta_nivel_3 = np.nan
        fecha_nivel = pd.NaT

        if nivel_col in x.columns:

            s = (
                x[
                    [
                        "datetime",
                        nivel_col,
                    ]
                ]
                .dropna(
                    subset=[
                        nivel_col
                    ]
                )
                .copy()
            )

            if not s.empty:

                nivel_actual = float(
                    s[
                        nivel_col
                    ].iloc[
                        -1
                    ]
                )

                fecha_nivel = (
                    s[
                        "datetime"
                    ].iloc[
                        -1
                    ]
                )

                if len(
                    s
                ) >= 2:

                    delta_nivel_1 = (
                        nivel_actual
                        - float(
                            s[
                                nivel_col
                            ].iloc[
                                -2
                            ]
                        )
                    )

                if len(
                    s
                ) >= 4:

                    delta_nivel_3 = (
                        nivel_actual
                        - float(
                            s[
                                nivel_col
                            ].iloc[
                                -4
                            ]
                        )
                    )

        # ====================================================
        # CAUDAL
        # ====================================================

        caudal_actual = np.nan
        delta_caudal_1 = np.nan

        if caudal_col in x.columns:

            q = (
                x[
                    caudal_col
                ]
                .dropna()
            )

            if not q.empty:

                caudal_actual = float(
                    q.iloc[
                        -1
                    ]
                )

                if len(
                    q
                ) >= 2:

                    delta_caudal_1 = (
                        caudal_actual
                        - float(
                            q.iloc[
                                -2
                            ]
                        )
                    )

        # ====================================================
        # LLUVIA
        # ====================================================

        lluvia_24h = np.nan
        lluvia_7d = np.nan

        if lluvia_col in x.columns:

            rain = (
                pd.to_numeric(
                    x[
                        lluvia_col
                    ],
                    errors="coerce",
                )
                .fillna(
                    0.0
                )
                .clip(
                    lower=0.0
                )
            )

            if not rain.empty:

                lluvia_24h = float(
                    rain.iloc[
                        -1
                    ]
                )

                lluvia_7d = float(
                    rain.tail(
                        7
                    )
                    .sum()
                )

        # ====================================================
        # TENDENCIA
        # ====================================================

        if pd.notna(
            delta_nivel_3
        ):

            if (
                delta_nivel_3
                > 0.05
            ):

                tendencia = (
                    "↑ Creciente"
                )

            elif (
                delta_nivel_3
                < -0.05
            ):

                tendencia = (
                    "↓ Bajante"
                )

            else:

                tendencia = (
                    "→ Estable"
                )

        else:

            tendencia = (
                "Sin datos"
            )

        rows.append(
            {
                "Estación":
                    station_name,

                "Fecha":
                    (
                        pd.to_datetime(
                            fecha_nivel
                        )
                        .strftime(
                            "%d/%m/%Y"
                        )
                        if pd.notna(
                            fecha_nivel
                        )
                        else "--"
                    ),

                "Nivel (m)":
                    (
                        round(
                            nivel_actual,
                            2,
                        )
                        if pd.notna(
                            nivel_actual
                        )
                        else np.nan
                    ),

                "Δ nivel 1d":
                    (
                        round(
                            delta_nivel_1,
                            2,
                        )
                        if pd.notna(
                            delta_nivel_1
                        )
                        else np.nan
                    ),

                "Δ nivel 3d":
                    (
                        round(
                            delta_nivel_3,
                            2,
                        )
                        if pd.notna(
                            delta_nivel_3
                        )
                        else np.nan
                    ),

                "Caudal (m³/s)":
                    (
                        round(
                            caudal_actual,
                            0,
                        )
                        if pd.notna(
                            caudal_actual
                        )
                        else np.nan
                    ),

                "Δ caudal 1d":
                    (
                        round(
                            delta_caudal_1,
                            0,
                        )
                        if pd.notna(
                            delta_caudal_1
                        )
                        else np.nan
                    ),

                "Lluvia 24h (mm)":
                    (
                        round(
                            lluvia_24h,
                            1,
                        )
                        if pd.notna(
                            lluvia_24h
                        )
                        else np.nan
                    ),

                "Lluvia 7d (mm)":
                    (
                        round(
                            lluvia_7d,
                            1,
                        )
                        if pd.notna(
                            lluvia_7d
                        )
                        else np.nan
                    ),

                "Tendencia":
                    tendencia,
            }
        )

    return pd.DataFrame(
        rows
    )
