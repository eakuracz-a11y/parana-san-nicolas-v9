import requests
import numpy as np
import pandas as pd

from functools import lru_cache

from src.exogenous import (
    RAIN_POINTS,
    find_best_caudal_series,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

OPEN_METEO_ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

INA_A5_URL = (
    "https://alerta.ina.gob.ar/a5/getObservaciones"
)

# Serie confirmada de nivel San Nicolás
SAN_NICOLAS_LEVEL_SERIES_ID = 36

# Se consulta desde 1800.
# El resultado real comienza donde la serie INA
# tenga sus primeros registros disponibles.
INA_SEARCH_START = pd.Timestamp(
    "1800-01-01"
)

# Open-Meteo ERA5 dispone de información
# desde 1940.
RAIN_SEARCH_START = pd.Timestamp(
    "1940-01-01"
)

# Ventana estacional para buscar valores
# correspondientes a una misma época del año.
CALENDAR_WINDOW_DAYS = 3

# Tamaño de los bloques de consulta INA.
INA_CHUNK_YEARS = 10


# ============================================================
# REQUEST
# ============================================================

def _request_json(
    url,
    params=None,
    timeout=120,
):

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent":
                "Parana-San-Nicolas-HistoricalScenario/1.0",

            "Accept":
                "application/json,text/plain,*/*",
        },
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# NORMALIZAR FECHA
# ============================================================

def _normalize_datetime(
    values,
):

    result = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )

    if isinstance(
        result,
        pd.Series,
    ):

        return (
            result
            .dt
            .tz_localize(None)
            .dt
            .normalize()
        )

    if pd.isna(
        result
    ):

        return pd.NaT

    return (
        result
        .tz_localize(None)
        .normalize()
    )


# ============================================================
# CLAVE CALENDARIO
# ============================================================

def _calendar_key(
    dt,
):

    dt = pd.Timestamp(
        dt
    )

    return (
        int(dt.month),
        int(dt.day),
    )


# ============================================================
# VENTANA DE DÍA/MES
# ============================================================

def _calendar_window_keys(
    month,
    day,
    window_days=3,
):

    # Año bisiesto artificial para
    # permitir correctamente 29/02.
    base = pd.Timestamp(
        year=2000,
        month=int(month),
        day=int(day),
    )

    keys = set()

    for offset in range(
        -window_days,
        window_days + 1,
    ):

        dt = (
            base
            + pd.Timedelta(
                days=offset
            )
        )

        keys.add(
            (
                int(dt.month),
                int(dt.day),
            )
        )

    return keys


# ============================================================
# DESCARGAR TODA UNA SERIE INA
# ============================================================

@lru_cache(
    maxsize=16
)
def fetch_full_ina_series(
    series_id,
):

    """
    Recupera toda la información que el servicio INA
    devuelve para una serie.

    Se consulta en bloques de 10 años para evitar una
    única petición extremadamente grande.
    """

    series_id = int(
        series_id
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    start = INA_SEARCH_START

    frames = []

    while start <= today:

        end = min(
            start
            + pd.DateOffset(
                years=INA_CHUNK_YEARS
            )
            - pd.Timedelta(
                days=1
            ),
            today,
        )

        params = {
            "tipo":
                "puntual",

            "series_id":
                series_id,

            "timestart":
                start.strftime(
                    "%Y-%m-%d"
                ),

            "timeend":
                end.strftime(
                    "%Y-%m-%d"
                ),
        }

        try:

            data = _request_json(
                INA_A5_URL,
                params=params,
            )

        except Exception:

            # No abortamos el histórico completo
            # por un bloque problemático.
            start = (
                end
                + pd.Timedelta(
                    days=1
                )
            )

            continue

        if isinstance(
            data,
            list,
        ) and data:

            df = pd.DataFrame(
                data
            )

            if (
                "timestart"
                in df.columns
                and "valor"
                in df.columns
            ):

                part = pd.DataFrame(
                    {
                        "datetime":
                            _normalize_datetime(
                                df[
                                    "timestart"
                                ]
                            ),

                        "value":
                            pd.to_numeric(
                                df[
                                    "valor"
                                ],
                                errors="coerce",
                            ),
                    }
                )

                part = part.dropna(
                    subset=[
                        "datetime",
                        "value",
                    ]
                )

                if not part.empty:

                    frames.append(
                        part
                    )

        start = (
            end
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

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result
        .dropna(
            subset=[
                "datetime",
                "value",
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

    return result


# ============================================================
# TODO EL HISTÓRICO DEL NIVEL SAN NICOLÁS
# ============================================================

def get_full_level_history():

    df = fetch_full_ina_series(
        SAN_NICOLAS_LEVEL_SERIES_ID
    ).copy()

    if df.empty:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    df = df.rename(
        columns={
            "value":
                "nivel",
        }
    )

    df[
        "nivel"
    ] = pd.to_numeric(
        df[
            "nivel"
        ],
        errors="coerce",
    )

    return (
        df
        .dropna(
            subset=[
                "datetime",
                "nivel",
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TODO EL HISTÓRICO DE CAUDAL INA
# ============================================================

def get_full_flow_history():

    info = find_best_caudal_series()

    if not info:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            None,
        )

    series_id = info.get(
        "series_id"
    )

    if series_id is None:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    df = fetch_full_ina_series(
        int(series_id)
    ).copy()

    if df.empty:

        return (
            pd.DataFrame(
                columns=[
                    "datetime",
                    "caudal_m3s",
                ]
            ),
            info,
        )

    df = df.rename(
        columns={
            "value":
                "caudal_m3s",
        }
    )

    df[
        "caudal_m3s"
    ] = pd.to_numeric(
        df[
            "caudal_m3s"
        ],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "datetime",
                "caudal_m3s",
            ]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        df,
        info,
    )


# ============================================================
# TODO EL HISTÓRICO DE PRECIPITACIÓN
# ============================================================

@lru_cache(
    maxsize=1
)
def get_full_rain_history():

    """
    Utiliza ERA5 desde 1940 hasta el último
    período razonablemente disponible.

    Se calcula, para cada fecha, el máximo de
    precipitación entre los puntos del corredor.
    """

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    # ERA5 tiene algunos días de retraso.
    historical_end = (
        today
        - pd.Timedelta(
            days=5
        )
    )

    frames = []

    for (
        station,
        coordinates,
    ) in RAIN_POINTS.items():

        latitude = float(
            coordinates[0]
        )

        longitude = float(
            coordinates[1]
        )

        params = {
            "latitude":
                latitude,

            "longitude":
                longitude,

            "start_date":
                RAIN_SEARCH_START.strftime(
                    "%Y-%m-%d"
                ),

            "end_date":
                historical_end.strftime(
                    "%Y-%m-%d"
                ),

            "daily":
                "precipitation_sum",

            "timezone":
                "America/Argentina/Buenos_Aires",

            # Mismo producto durante décadas.
            "models":
                "era5",
        }

        try:

            data = _request_json(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=180,
            )

        except Exception:

            continue

        daily = data.get(
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
            or not rain
        ):

            continue

        column = (
            "rain_"
            + station
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        frame = pd.DataFrame(
            {
                "datetime":
                    pd.to_datetime(
                        dates,
                        errors="coerce",
                    ),

                column:
                    pd.to_numeric(
                        rain,
                        errors="coerce",
                    ),
            }
        )

        frame[
            column
        ] = (
            frame[
                column
            ]
            .clip(
                lower=0
            )
        )

        frames.append(
            frame
        )

    if not frames:

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
            ]
        )

    merged = frames[0]

    for frame in frames[1:]:

        merged = merged.merge(
            frame,
            on="datetime",
            how="outer",
        )

    rain_columns = [
        c
        for c in merged.columns
        if c.startswith(
            "rain_"
        )
    ]

    # Para el escenario extremo nos interesa
    # la mayor precipitación del corredor
    # registrada cada día.
    merged[
        "precip_mm"
    ] = (
        merged[
            rain_columns
        ]
        .max(
            axis=1,
            skipna=True,
        )
    )

    merged[
        "precip_mm"
    ] = (
        pd.to_numeric(
            merged[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0
        )
    )

    result = (
        merged[
            [
                "datetime",
                "precip_mm",
            ]
        ]
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

    return result


# ============================================================
# OBTENER TODOS LOS HISTÓRICOS
# ============================================================

def get_full_historical_bundle():

    level = get_full_level_history()

    rain = get_full_rain_history().copy()

    flow, flow_info = (
        get_full_flow_history()
    )

    return {
        "level":
            level,

        "rain":
            rain,

        "flow":
            flow,

        "flow_info":
            flow_info,
    }


# ============================================================
# FILTRAR POR VENTANA DEL CALENDARIO
# ============================================================

def _calendar_subset(
    df,
    target_date,
    column,
    window_days=3,
    positive_only=False,
):

    if (
        df is None
        or df.empty
        or "datetime"
        not in df.columns
        or column not in df.columns
    ):

        return pd.DataFrame()

    work = df.copy()

    work[
        "datetime"
    ] = pd.to_datetime(
        work[
            "datetime"
        ],
        errors="coerce",
    )

    work[
        column
    ] = pd.to_numeric(
        work[
            column
        ],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "datetime",
            column,
        ]
    )

    if positive_only:

        work = work[
            work[
                column
            ] > 0
        ]

    if work.empty:

        return work

    keys = _calendar_window_keys(
        target_date.month,
        target_date.day,
        window_days=window_days,
    )

    mask = [
        (
            int(dt.month),
            int(dt.day),
        )
        in keys
        for dt
        in work[
            "datetime"
        ]
    ]

    return work.loc[
        mask
    ].copy()


# ============================================================
# FALLBACK ESTACIONAL
# ============================================================

def _fallback_month(
    df,
    target_date,
    column,
    positive_only=False,
):

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    work = df.copy()

    work[
        "datetime"
    ] = pd.to_datetime(
        work[
            "datetime"
        ],
        errors="coerce",
    )

    work[
        column
    ] = pd.to_numeric(
        work[
            column
        ],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "datetime",
            column,
        ]
    )

    work = work[
        work[
            "datetime"
        ].dt.month
        == target_date.month
    ]

    if positive_only:

        work = work[
            work[
                column
            ] > 0
        ]

    return work


# ============================================================
# ESTADÍSTICA PARA UNA FECHA FUTURA
# ============================================================

def _date_statistics(
    df,
    target_date,
    column,
    positive_only=False,
):

    subset = _calendar_subset(
        df=df,
        target_date=target_date,
        column=column,
        window_days=CALENDAR_WINDOW_DAYS,
        positive_only=positive_only,
    )

    source = "±3 días"

    # Si no existen registros válidos
    # usamos el mes correspondiente.
    if subset.empty:

        subset = _fallback_month(
            df=df,
            target_date=target_date,
            column=column,
            positive_only=positive_only,
        )

        source = "mes"

    # Último fallback: todo el historial.
    if subset.empty:

        subset = df.copy()

        subset[
            column
        ] = pd.to_numeric(
            subset[
                column
            ],
            errors="coerce",
        )

        subset = subset.dropna(
            subset=[
                "datetime",
                column,
            ]
        )

        if positive_only:

            subset = subset[
                subset[
                    column
                ] > 0
            ]

        source = "histórico completo"

    if subset.empty:

        return {
            "p90":
                np.nan,

            "p95":
                np.nan,

            "maximum":
                np.nan,

            "max_date":
                None,

            "records":
                0,

            "source":
                source,
        }

    values = pd.to_numeric(
        subset[
            column
        ],
        errors="coerce",
    ).dropna()

    if values.empty:

        return {
            "p90":
                np.nan,

            "p95":
                np.nan,

            "maximum":
                np.nan,

            "max_date":
                None,

            "records":
                0,

            "source":
                source,
        }

    idx = values.idxmax()

    return {
        "p90":
            float(
                values.quantile(
                    0.90
                )
            ),

        "p95":
            float(
                values.quantile(
                    0.95
                )
            ),

        "maximum":
            float(
                values.max()
            ),

        "max_date":
            pd.to_datetime(
                subset.loc[
                    idx,
                    "datetime",
                ]
            ),

        "records":
            int(
                len(
                    values
                )
            ),

        "source":
            source,
    }


# ============================================================
# ENVOLVENTE HISTÓRICA 60 DÍAS
# ============================================================

def build_daily_historical_envelope(
    start_date,
    days=60,
):

    bundle = get_full_historical_bundle()

    rain = bundle[
        "rain"
    ].copy()

    flow = bundle[
        "flow"
    ].copy()

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    future_dates = pd.date_range(
        start=(
            start_date
            + pd.Timedelta(
                days=1
            )
        ),
        periods=int(days),
        freq="D",
    )

    rows = []

    for target_date in future_dates:

        # ----------------------------------------------------
        # PRECIPITACIÓN
        #
        # positive_only=True evita que P90/P95 se conviertan
        # en cero solamente porque históricamente hubo
        # muchos días secos.
        # ----------------------------------------------------

        rain_stats = _date_statistics(
            df=rain,
            target_date=target_date,
            column="precip_mm",
            positive_only=True,
        )

        # ----------------------------------------------------
        # CAUDAL
        # ----------------------------------------------------

        flow_stats = _date_statistics(
            df=flow,
            target_date=target_date,
            column="caudal_m3s",
            positive_only=False,
        )

        rows.append(
            {
                "datetime":
                    target_date,

                "rain_p90_mm":
                    rain_stats[
                        "p90"
                    ],

                "rain_p95_mm":
                    rain_stats[
                        "p95"
                    ],

                "rain_max_mm":
                    rain_stats[
                        "maximum"
                    ],

                "rain_source_date":
                    rain_stats[
                        "max_date"
                    ],

                "rain_records":
                    rain_stats[
                        "records"
                    ],

                "rain_search":
                    rain_stats[
                        "source"
                    ],

                "flow_p90_m3s":
                    flow_stats[
                        "p90"
                    ],

                "flow_p95_m3s":
                    flow_stats[
                        "p95"
                    ],

                "flow_max_m3s":
                    flow_stats[
                        "maximum"
                    ],

                "flow_source_date":
                    flow_stats[
                        "max_date"
                    ],

                "flow_records":
                    flow_stats[
                        "records"
                    ],

                "flow_search":
                    flow_stats[
                        "source"
                    ],
            }
        )

    result = pd.DataFrame(
        rows
    )

    return (
        result,
        bundle,
    )
