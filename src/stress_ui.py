import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# VERSIÓN
# ============================================================

STRESS_VERSION = "V11.6.1"


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRESS_DAYS = 60

# Escala visual fija
Y_MIN = 0.0
Y_MAX = 7.0
Y_STEP = 0.5

MIN_HISTORY_DAYS = 120


# ============================================================
# UTILIDADES
# ============================================================

def _normalizar_datetime(serie):

    return (
        pd.to_datetime(
            serie,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
        .dt
        .normalize()
    )


def _numeric(serie):

    return pd.to_numeric(
        serie,
        errors="coerce",
    )


def _safe_float(
    value,
    default=np.nan,
):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return default


def _safe_last(
    serie,
    default=np.nan,
):

    s = (
        pd.to_numeric(
            serie,
            errors="coerce",
        )
        .dropna()
    )

    if s.empty:

        return default

    return float(
        s.iloc[-1]
    )


def _format_number(
    value,
    decimals=1,
):

    if value is None:

        return "--"

    try:

        if pd.isna(value):

            return "--"

    except Exception:

        pass

    text = (
        f"{float(value):,.{decimals}f}"
    )

    return (
        text
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def _upstream_cols(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
    ):

        return []

    return [
        c
        for c in df.columns
        if (
            c.startswith("nivel_")
            and c != "nivel"
            and "_lag" not in c
            and "_diff" not in c
            and "_trend" not in c
            and "_mean" not in c
            and "_actual" not in c
            and "_next" not in c
        )
    ]


# ============================================================
# PREPARAR NIVEL SAN NICOLÁS
# ============================================================

def _preparar_nivel(df):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame,
        )
        or df.empty
        or "datetime" not in df.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x = df.copy()

    if "nivel" in x.columns:

        x["nivel"] = _numeric(
            x["nivel"]
        )

    elif "value" in x.columns:

        x["nivel"] = _numeric(
            x["value"]
        )

    else:

        return pd.DataFrame(
            columns=[
                "datetime",
                "nivel",
            ]
        )

    x["datetime"] = _normalizar_datetime(
        x["datetime"]
    )

    x = x.dropna(
        subset=[
            "datetime",
            "nivel",
        ]
    )

    x = (
        x
        .groupby(
            "datetime",
            as_index=False,
        )["nivel"]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # ========================================================
    # HACER SERIE DIARIA CONTINUA
    # ========================================================

    if len(x) >= 2:

        fecha_min = x["datetime"].min()
        fecha_max = x["datetime"].max()

        calendario = pd.DataFrame(
            {
                "datetime":
                    pd.date_range(
                        fecha_min,
                        fecha_max,
                        freq="D",
                    )
            }
        )

        x = calendario.merge(
            x,
            on="datetime",
            how="left",
        )

        # Sólo completar huecos relativamente cortos.
        x["nivel"] = (
            x["nivel"]
            .interpolate(
                limit=7,
                limit_direction="both",
            )
        )

    x = x.dropna(
        subset=[
            "nivel"
        ]
    )

    return (
        x
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# ============================================================
# PREPARAR LLUVIA Y CAUDAL
# ============================================================

def _preparar_exog(
    exog_history,
):

    if (
        exog_history is None
        or not isinstance(
            exog_history,
            pd.DataFrame,
        )
        or exog_history.empty
        or "datetime" not in exog_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "precip_mm",
                "caudal_m3s",
            ]
        )

    x = exog_history.copy()

    x["datetime"] = _normalizar_datetime(
        x["datetime"]
    )

    if "precip_mm" not in x.columns:

        x["precip_mm"] = np.nan

    if "caudal_m3s" not in x.columns:

        x["caudal_m3s"] = np.nan

    x["precip_mm"] = (
        _numeric(
            x["precip_mm"]
        )
        .clip(lower=0.0)
    )

    x["caudal_m3s"] = _numeric(
        x["caudal_m3s"]
    )

    x = x.dropna(
        subset=[
            "datetime"
        ]
    )

    x = (
        x
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            precip_mm=(
                "precip_mm",
                "mean",
            ),
            caudal_m3s=(
                "caudal_m3s",
                "mean",
            ),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return x


# ============================================================
# PREPARAR AGUAS ARRIBA
# ============================================================

def _preparar_upstream(
    upstream_history,
):

    if (
        upstream_history is None
        or not isinstance(
            upstream_history,
            pd.DataFrame,
        )
        or upstream_history.empty
        or "datetime" not in upstream_history.columns
    ):

        return pd.DataFrame(
            columns=[
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        )

    x = upstream_history.copy()

    x["datetime"] = _normalizar_datetime(
        x["datetime"]
    )

    cols = _upstream_cols(x)

    if not cols:

        return pd.DataFrame(
            columns=[
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        )

    for col in cols:

        x[col] = _numeric(
            x[col]
        )

        if (
            x[col]
            .notna()
            .sum()
            >= 2
        ):

            x[col] = (
                x[col]
                .interpolate(
                    limit=3,
                    limit_direction="both",
                )
            )

    x["upstream_mean"] = (
        x[cols]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    x["upstream_max"] = (
        x[cols]
        .max(
            axis=1,
            skipna=True,
        )
    )

    return (
        x[
            [
                "datetime",
                "upstream_mean",
                "upstream_max",
            ]
        ]
        .dropna(
            subset=[
                "datetime"
            ]
        )
        .groupby(
            "datetime",
            as_index=False,
        )
        .agg(
            upstream_mean=(
                "upstream_mean",
                "mean",
            ),
            upstream_max=(
                "upstream_max",
                "max",
            ),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# ============================================================
# HISTÓRICO INTEGRADO
# ============================================================

def _armar_historico(
    df,
    exog_history,
    upstream_history,
):

    nivel = _preparar_nivel(df)

    if nivel.empty:

        return pd.DataFrame()

    hist = nivel.copy()

    exog = _preparar_exog(
        exog_history
    )

    upstream = _preparar_upstream(
        upstream_history
    )

    if not exog.empty:

        hist = hist.merge(
            exog,
            on="datetime",
            how="left",
        )

    if not upstream.empty:

        hist = hist.merge(
            upstream,
            on="datetime",
            how="left",
        )

    for col in [
        "precip_mm",
        "caudal_m3s",
        "upstream_mean",
        "upstream_max",
    ]:

        if col not in hist.columns:

            hist[col] = np.nan

        hist[col] = _numeric(
            hist[col]
        )

    # ========================================================
    # CAUDAL
    # ========================================================

    if (
        hist["caudal_m3s"]
        .notna()
        .sum()
        >= 7
    ):

        hist["caudal_m3s"] = (
            hist["caudal_m3s"]
            .interpolate(
                limit=7,
                limit_direction="both",
            )
        )

    # ========================================================
    # AGUAS ARRIBA
    # ========================================================

    for col in [
        "upstream_mean",
        "upstream_max",
    ]:

        if (
            hist[col]
            .notna()
            .sum()
            >= 7
        ):

            hist[col] = (
                hist[col]
                .interpolate(
                    limit=5,
                    limit_direction="both",
                )
            )

    return (
        hist
        .sort_values("datetime")
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# BUSCAR EVENTOS HISTÓRICOS DE 60 DÍAS
# ============================================================

def _buscar_eventos_crecida(
    hist,
    days=STRESS_DAYS,
):

    eventos = []

    if (
        hist is None
        or not isinstance(
            hist,
            pd.DataFrame,
        )
        or hist.empty
        or len(hist) < days
    ):

        return eventos

    # ========================================================
    # VENTANAS CONTINUAS
    # ========================================================

    for inicio in range(
        0,
        len(hist) - days + 1,
    ):

        block = (
            hist
            .iloc[
                inicio:
                inicio + days
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(block) != days:

            continue

        # Verificar que sean realmente 60 días consecutivos.
        fecha_inicio = block[
            "datetime"
        ].iloc[0]

        fecha_fin = block[
            "datetime"
        ].iloc[-1]

        diferencia = (
            fecha_fin
            - fecha_inicio
        ).days

        if diferencia != days - 1:

            continue

        niveles = (
            pd.to_numeric(
                block["nivel"],
                errors="coerce",
            )
        )

        if niveles.notna().sum() < days:

            continue

        valores = niveles.to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(valores)
        ):

            continue

        nivel_inicial = float(
            valores[0]
        )

        # ====================================================
        # CRECIDA MÁXIMA DESDE EL NIVEL INICIAL
        # ====================================================

        nivel_maximo = float(
            np.max(valores)
        )

        peak_idx = int(
            np.argmax(valores)
        )

        crecimiento = (
            nivel_maximo
            - nivel_inicial
        )

        if crecimiento <= 0.05:

            continue

        # Queremos que el pico no esté exactamente en el
        # primer día porque no representaría una creciente.
        if peak_idx < 2:

            continue

        eventos.append(
            {
                "growth":
                    float(crecimiento),

                "initial":
                    nivel_inicial,

                "max":
                    nivel_maximo,

                "peak_idx":
                    peak_idx,

                "start_date":
                    fecha_inicio,

                "end_date":
                    fecha_fin,

                "block":
                    block,
            }
        )

    eventos = sorted(
        eventos,
        key=lambda evento:
            evento["growth"],
    )

    return eventos


# ============================================================
# ELEGIR EVENTO SEGÚN PERCENTIL
# ============================================================

def _evento_percentil(
    eventos,
    percentile,
):

    if not eventos:

        return None

    percentile = float(
        np.clip(
            percentile,
            0.0,
            1.0,
        )
    )

    growths = np.array(
        [
            evento["growth"]
            for evento in eventos
        ],
        dtype=float,
    )

    target = float(
        np.quantile(
            growths,
            percentile,
        )
    )

    diferencias = np.abs(
        growths - target
    )

    idx = int(
        np.argmin(
            diferencias
        )
    )

    return eventos[idx]


# ============================================================
# PEOR EVENTO
# ============================================================

def _peor_evento(eventos):

    if not eventos:

        return None

    return max(
        eventos,
        key=lambda evento:
            evento["growth"],
    )


# ============================================================
# CONSTRUIR FORMA DEL EVENTO
# ============================================================

def _event_shape(
    event,
):

    if (
        event is None
        or "block" not in event
    ):

        return np.zeros(
            STRESS_DAYS,
            dtype=float,
        )

    block = event["block"]

    niveles = (
        pd.to_numeric(
            block["nivel"],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # GARANTIZAR LONGITUD
    # ========================================================

    if len(niveles) < STRESS_DAYS:

        if len(niveles) == 0:

            return np.zeros(
                STRESS_DAYS,
                dtype=float,
            )

        niveles = np.pad(
            niveles,
            (
                0,
                STRESS_DAYS
                - len(niveles),
            ),
            mode="edge",
        )

    elif len(niveles) > STRESS_DAYS:

        niveles = niveles[
            :STRESS_DAYS
        ]

    # ========================================================
    # COMPLETAR CUALQUIER NAN RESIDUAL
    # ========================================================

    niveles = (
        pd.Series(niveles)
        .interpolate(
            limit_direction="both"
        )
        .ffill()
        .bfill()
        .to_numpy(
            dtype=float
        )
    )

    if not np.all(
        np.isfinite(niveles)
    ):

        return np.zeros(
            STRESS_DAYS,
            dtype=float,
        )

    nivel_inicial = float(
        niveles[0]
    )

    # Diferencia respecto del inicio histórico
    shape = (
        niveles
        - nivel_inicial
    )

    # ========================================================
    # SUAVIZACIÓN
    # ========================================================

    shape = (
        pd.Series(shape)
        .rolling(
            window=3,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # IDENTIFICAR PICO
    # ========================================================

    peak_idx = int(
        np.argmax(shape)
    )

    # ========================================================
    # ETAPA DE CRECIDA
    # ========================================================

    subida = shape[
        :peak_idx + 1
    ].copy()

    # Evitar pequeñas oscilaciones negativas antes del pico.
    subida = np.maximum.accumulate(
        subida
    )

    # ========================================================
    # ETAPA DESPUÉS DEL PICO
    # ========================================================

    bajada = shape[
        peak_idx + 1:
    ].copy()

    if len(bajada) > 0:

        bajada = (
            pd.Series(bajada)
            .rolling(
                window=5,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy(
                dtype=float
            )
        )

    # ========================================================
    # UNIR
    # ========================================================

    shape_final = np.concatenate(
        [
            subida,
            bajada,
        ]
    )

    # ========================================================
    # EN ESCENARIO SEVERO NO BAJAR POR DEBAJO DEL NIVEL
    # DE PARTIDA
    # ========================================================

    shape_final = np.maximum(
        shape_final,
        0.0,
    )

    # ========================================================
    # LONGITUD EXACTA = 60
    # ========================================================

    if len(shape_final) < STRESS_DAYS:

        shape_final = np.pad(
            shape_final,
            (
                0,
                STRESS_DAYS
                - len(shape_final),
            ),
            mode="edge",
        )

    elif len(shape_final) > STRESS_DAYS:

        shape_final = shape_final[
            :STRESS_DAYS
        ]

    return shape_final.astype(
        float
    )


# ============================================================
# CONSTRUIR ESCENARIO
# ============================================================

def _crear_escenario(
    hist,
    event,
):

    if (
        hist is None
        or not isinstance(
            hist,
            pd.DataFrame,
        )
        or hist.empty
        or event is None
    ):

        return pd.DataFrame()

    nivel_actual = _safe_last(
        hist["nivel"]
    )

    if pd.isna(
        nivel_actual
    ):

        return pd.DataFrame()

    ultima_fecha = pd.to_datetime(
        hist["datetime"].max()
    )

    future_dates = pd.date_range(
        start=(
            ultima_fecha
            + pd.Timedelta(days=1)
        ),
        periods=STRESS_DAYS,
        freq="D",
    )

    # ========================================================
    # FORMA DEL EVENTO
    # ========================================================

    shape = _event_shape(
        event
    )

    if len(shape) != STRESS_DAYS:

        return pd.DataFrame()

    # ========================================================
    # TRASLADAR EVENTO AL NIVEL ACTUAL
    # ========================================================

    levels = (
        float(nivel_actual)
        + shape
    )

    # Límite técnico.
    # La visualización queda siempre 0–7 m.
    levels = np.clip(
        levels,
        0.0,
        12.0,
    )

    # ========================================================
    # ASEGURAR ARRANQUE EXACTO
    # ========================================================

    levels[0] = float(
        nivel_actual
    )

    block = (
        event["block"]
        .copy()
        .reset_index(drop=True)
    )

    # ========================================================
    # CREAR SALIDA
    # ========================================================

    registros = []

    nivel_anterior = float(
        nivel_actual
    )

    for i in range(
        STRESS_DAYS
    ):

        nivel = float(
            levels[i]
        )

        if i < len(block):

            source = block.iloc[i]

            lluvia = _safe_float(
                source.get(
                    "precip_mm",
                    np.nan,
                ),
                default=0.0,
            )

            caudal = _safe_float(
                source.get(
                    "caudal_m3s",
                    np.nan,
                ),
                default=np.nan,
            )

            upstream = _safe_float(
                source.get(
                    "upstream_mean",
                    np.nan,
                ),
                default=np.nan,
            )

            fecha_origen = source.get(
                "datetime",
                pd.NaT,
            )

        else:

            lluvia = 0.0
            caudal = np.nan
            upstream = np.nan
            fecha_origen = pd.NaT

        registros.append(
            {
                "datetime":
                    future_dates[i],

                "prediction":
                    nivel,

                "nivel_base":
                    nivel_anterior,

                "variacion_dia":
                    nivel
                    - nivel_anterior,

                "crecimiento_desde_actual":
                    nivel
                    - nivel_actual,

                "precip_mm":
                    max(
                        lluvia,
                        0.0,
                    )
                    if pd.notna(lluvia)
                    else 0.0,

                "caudal_m3s":
                    caudal,

                "upstream_mean":
                    upstream,

                "source_date":
                    fecha_origen,
            }
        )

        nivel_anterior = nivel

    return pd.DataFrame(
        registros
    )


# ============================================================
# ENVOLVENTE HISTÓRICA
# ============================================================

def _envolvente_historica_diaria(
    df,
    future_dates,
):

    nivel = _preparar_nivel(
        df
    )

    if nivel.empty:

        return pd.DataFrame()

    nivel["month"] = (
        nivel["datetime"]
        .dt
        .month
    )

    nivel["day"] = (
        nivel["datetime"]
        .dt
        .day
    )

    resumen = (
        nivel
        .groupby(
            [
                "month",
                "day",
            ],
            as_index=False,
        )
        .agg(
            nivel_min_historico=(
                "nivel",
                "min",
            ),
            nivel_max_historico=(
                "nivel",
                "max",
            ),
            nivel_promedio_historico=(
                "nivel",
                "mean",
            ),
            registros=(
                "nivel",
                "count",
            ),
        )
    )

    future = pd.DataFrame(
        {
            "datetime":
                pd.to_datetime(
                    future_dates,
                    errors="coerce",
                )
        }
    )

    future["month"] = (
        future["datetime"]
        .dt
        .month
    )

    future["day"] = (
        future["datetime"]
        .dt
        .day
    )

    return future.merge(
        resumen,
        on=[
            "month",
            "day",
        ],
        how="left",
    )


# ============================================================
# MÉTRICAS ESCENARIO
# ============================================================

def _scenario_metrics(
    scenario,
    nivel_actual,
):

    if (
        scenario is None
        or not isinstance(
            scenario,
            pd.DataFrame,
        )
        or scenario.empty
    ):

        return {}

    valores = pd.to_numeric(
        scenario["prediction"],
        errors="coerce",
    )

    if valores.notna().sum() == 0:

        return {}

    idx = valores.idxmax()

    maximo = float(
        valores.loc[idx]
    )

    fecha = pd.to_datetime(
        scenario.loc[
            idx,
            "datetime",
        ]
    )

    final = float(
        valores.iloc[-1]
    )

    posicion = int(
        scenario.index.get_loc(idx)
    )

    return {
        "max":
            maximo,

        "date":
            fecha,

        "growth":
            maximo
            - nivel_actual,

        "final":
            final,

        "days_to_peak":
            posicion + 1,
    }


# ============================================================
# GRÁFICO BASE 0–7 m
# ============================================================

def _aplicar_escala_nivel(
    fig,
    height=520,
):

    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10,
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(
        title_text="Nivel hidrométrico (m)",
        range=[
            Y_MIN,
            Y_MAX,
        ],
        dtick=
            Y_STEP,
        autorange=False,
    )

    return fig


# ============================================================
# RENDER PRINCIPAL
# ============================================================

def render_stress_scenario(
    df,
    models=None,
    exog_history=None,
    upstream_history=None,
):

    st.subheader(
        "⚠️ Escenarios históricos severos · 60 días"
    )

    st.caption(
        f"{STRESS_VERSION} · "
        "Simulación desde el nivel real actual. "
        "Escenarios P90, P95 y peor creciente histórica."
    )

    # ========================================================
    # PREPARAR HISTÓRICO
    # ========================================================

    hist = _armar_historico(
        df=df,
        exog_history=exog_history,
        upstream_history=upstream_history,
    )

    if hist.empty:

        st.info(
            "No existen datos históricos suficientes."
        )

        return

    if len(hist) < MIN_HISTORY_DAYS:

        st.info(
            "Se requieren al menos "
            f"{MIN_HISTORY_DAYS} días históricos. "
            f"Disponibles: {len(hist)}."
        )

        return

    nivel_actual = _safe_last(
        hist["nivel"]
    )

    if pd.isna(
        nivel_actual
    ):

        st.info(
            "No existe un nivel actual válido."
        )

        return

    # ========================================================
    # BUSCAR CRECIDAS HISTÓRICAS
    # ========================================================

    eventos = _buscar_eventos_crecida(
        hist,
        days=STRESS_DAYS,
    )

    if not eventos:

        st.info(
            "No fue posible identificar eventos "
            "históricos continuos de creciente."
        )

        return

    evento_p90 = _evento_percentil(
        eventos,
        0.90,
    )

    evento_p95 = _evento_percentil(
        eventos,
        0.95,
    )

    peor_evento = _peor_evento(
        eventos
    )

    # ========================================================
    # CONSTRUIR ESCENARIOS
    # ========================================================

    scenario_p90 = _crear_escenario(
        hist,
        evento_p90,
    )

    scenario_p95 = _crear_escenario(
        hist,
        evento_p95,
    )

    scenario_worst = _crear_escenario(
        hist,
        peor_evento,
    )

    if scenario_worst.empty:

        st.error(
            "No fue posible construir el peor escenario histórico."
        )

        return

    # ========================================================
    # ENVOLVENTE HISTÓRICA
    # ========================================================

    envelope = _envolvente_historica_diaria(
        df,
        scenario_worst[
            "datetime"
        ],
    )

    if not envelope.empty:

        scenario_worst = (
            scenario_worst
            .merge(
                envelope[
                    [
                        "datetime",
                        "nivel_min_historico",
                        "nivel_max_historico",
                        "nivel_promedio_historico",
                        "registros",
                    ]
                ],
                on="datetime",
                how="left",
            )
        )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    m90 = _scenario_metrics(
        scenario_p90,
        nivel_actual,
    )

    m95 = _scenario_metrics(
        scenario_p95,
        nivel_actual,
    )

    mw = _scenario_metrics(
        scenario_worst,
        nivel_actual,
    )

    st.markdown(
        "### Resumen de escenarios"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel real actual",
        f"{nivel_actual:.2f} m",
    )

    c2.metric(
        "P90 · máximo",
        (
            f"{m90['max']:.2f} m"
            if m90
            else "--"
        ),
        (
            f"{m90['growth']:+.2f} m"
            if m90
            else None
        ),
    )

    c3.metric(
        "P95 · máximo",
        (
            f"{m95['max']:.2f} m"
            if m95
            else "--"
        ),
        (
            f"{m95['growth']:+.2f} m"
            if m95
            else None
        ),
    )

    c4.metric(
        "Peor caso · máximo",
        f"{mw['max']:.2f} m",
        f"{mw['growth']:+.2f} m",
    )

    # ========================================================
    # SEGUNDA FILA
    # ========================================================

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Días hasta pico",
        mw[
            "days_to_peak"
        ],
    )

    d2.metric(
        "Fecha del pico",
        mw[
            "date"
        ].strftime(
            "%d/%m/%Y"
        ),
    )

    d3.metric(
        "Nivel día 60",
        f"{mw['final']:.2f} m",
        f"{mw['final'] - nivel_actual:+.2f} m",
    )

    d4.metric(
        "Crecida histórica patrón",
        f"{peor_evento['growth']:+.2f} m",
    )

    # ========================================================
    # MÁXIMO HISTÓRICO DEL PERÍODO
    # ========================================================

    max_hist_periodo = np.nan

    if (
        "nivel_max_historico"
        in scenario_worst.columns
        and scenario_worst[
            "nivel_max_historico"
        ]
        .notna()
        .any()
    ):

        max_hist_periodo = float(
            scenario_worst[
                "nivel_max_historico"
            ]
            .max()
        )

        st.caption(
            "Máximo histórico para las fechas del horizonte: "
            f"**{max_hist_periodo:.2f} m**."
        )

    # ========================================================
    # GRÁFICO PRINCIPAL
    # ========================================================

    fig = go.Figure()

    # ========================================================
    # MÁXIMO HISTÓRICO
    # ========================================================

    if (
        "nivel_max_historico"
        in scenario_worst.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario_worst[
                    "datetime"
                ],
                y=scenario_worst[
                    "nivel_max_historico"
                ],
                mode="lines",
                name="Máximo histórico del día",
                line=dict(
                    color="#d62728",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Máximo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # ========================================================
    # MÍNIMO HISTÓRICO
    # ========================================================

    if (
        "nivel_min_historico"
        in scenario_worst.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=scenario_worst[
                    "datetime"
                ],
                y=scenario_worst[
                    "nivel_min_historico"
                ],
                mode="lines",
                name="Mínimo histórico del día",
                line=dict(
                    color="#2ca02c",
                    width=2,
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Mínimo histórico: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # ========================================================
    # P90
    # ========================================================

    if not scenario_p90.empty:

        fig.add_trace(
            go.Scatter(
                x=scenario_p90[
                    "datetime"
                ],
                y=scenario_p90[
                    "prediction"
                ],
                mode="lines",
                name="P90",
                line=dict(
                    color="#17becf",
                    width=2,
                    dash="dot",
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "P90: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # ========================================================
    # P95
    # ========================================================

    if not scenario_p95.empty:

        fig.add_trace(
            go.Scatter(
                x=scenario_p95[
                    "datetime"
                ],
                y=scenario_p95[
                    "prediction"
                ],
                mode="lines",
                name="P95",
                line=dict(
                    color="#9467bd",
                    width=3,
                    dash="dash",
                ),
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "P95: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # ========================================================
    # PEOR CASO
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=scenario_worst[
                "datetime"
            ],
            y=scenario_worst[
                "prediction"
            ],
            mode="lines+markers",
            name="Peor caso histórico",
            line=dict(
                color="#1f77b4",
                width=4,
            ),
            marker=dict(
                size=5,
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Peor caso: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    # ========================================================
    # NIVEL REAL
    # ========================================================

    fig.add_hline(
        y=nivel_actual,
        line_dash="dash",
        line_width=2,
        line_color="black",
        annotation_text=(
            f"Nivel real de partida: "
            f"{nivel_actual:.2f} m"
        ),
    )

    # ========================================================
    # PICO
    # ========================================================

    fig.add_vline(
        x=mw[
            "date"
        ],
        line_dash="dot",
        line_width=1,
    )

    fig.add_annotation(
        x=mw[
            "date"
        ],
        y=mw[
            "max"
        ],
        text=(
            "Pico peor caso"
            "<br>"
            f"{mw['max']:.2f} m"
        ),
        showarrow=True,
        arrowhead=2,
        yshift=25,
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            y=1.08,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        tickformat="%d/%m/%Y",
    )

    # ========================================================
    # ESCALA SIEMPRE 0–7 m
    # ========================================================

    _aplicar_escala_nivel(
        fig,
        height=550,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # DÍAS SOBRE MÁXIMO HISTÓRICO
    # ========================================================

    dias_supera = 0

    if (
        "nivel_max_historico"
        in scenario_worst.columns
    ):

        comparacion = (
            scenario_worst[
                "prediction"
            ]
            > scenario_worst[
                "nivel_max_historico"
            ]
        )

        dias_supera = int(
            comparacion
            .fillna(False)
            .sum()
        )

    if dias_supera > 0:

        st.warning(
            "⚠️ El peor escenario supera el máximo histórico "
            f"diario de referencia durante **{dias_supera} días**."
        )

    # ========================================================
    # EVENTO HISTÓRICO UTILIZADO
    # ========================================================

    block = (
        peor_evento[
            "block"
        ]
        .copy()
        .reset_index(drop=True)
    )

    st.markdown(
        "### 📚 Evento histórico utilizado como patrón"
    )

    e1, e2, e3, e4 = st.columns(
        4
    )

    e1.metric(
        "Inicio histórico",
        peor_evento[
            "start_date"
        ].strftime(
            "%d/%m/%Y"
        ),
    )

    e2.metric(
        "Nivel inicial histórico",
        f"{peor_evento['initial']:.2f} m",
    )

    e3.metric(
        "Máximo histórico evento",
        f"{peor_evento['max']:.2f} m",
    )

    e4.metric(
        "Crecimiento histórico",
        f"+{peor_evento['growth']:.2f} m",
    )

    # ========================================================
    # GRÁFICO EVENTO HISTÓRICO
    # ========================================================

    hist_fig = go.Figure()

    hist_fig.add_trace(
        go.Scatter(
            x=block[
                "datetime"
            ],
            y=block[
                "nivel"
            ],
            mode="lines+markers",
            name="Nivel histórico",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=4,
            ),
        )
    )

    hist_fig.update_xaxes(
        title_text="Fecha histórica",
        tickformat="%d/%m/%Y",
    )

    # ========================================================
    # ESCALA 0–7 m
    # ========================================================

    _aplicar_escala_nivel(
        hist_fig,
        height=330,
    )

    st.plotly_chart(
        hist_fig,
        use_container_width=True,
    )

    # ========================================================
    # VARIABLES DEL EVENTO
    # ========================================================

    lluvia = (
        pd.to_numeric(
            block[
                "precip_mm"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    lluvia_total = float(
        lluvia.sum()
    )

    lluvia_maxima = float(
        lluvia.max()
    )

    caudal_maximo = np.nan

    if (
        block[
            "caudal_m3s"
        ]
        .notna()
        .any()
    ):

        caudal_maximo = float(
            block[
                "caudal_m3s"
            ]
            .max()
        )

    upstream_maximo = np.nan

    if (
        block[
            "upstream_mean"
        ]
        .notna()
        .any()
    ):

        upstream_maximo = float(
            block[
                "upstream_mean"
            ]
            .max()
        )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Lluvia acumulada",
        f"{_format_number(lluvia_total, 1)} mm",
    )

    v2.metric(
        "Lluvia máxima diaria",
        f"{_format_number(lluvia_maxima, 1)} mm",
    )

    v3.metric(
        "Caudal máximo",
        (
            f"{_format_number(caudal_maximo, 0)} m³/s"
            if pd.notna(
                caudal_maximo
            )
            else "--"
        ),
    )

    v4.metric(
        "Aguas arriba máximo medio",
        (
            f"{upstream_maximo:.2f} m"
            if pd.notna(
                upstream_maximo
            )
            else "--"
        ),
    )

    # ========================================================
    # LLUVIA Y CAUDAL
    # ========================================================

    rain_col, flow_col = st.columns(
        2
    )

    with rain_col:

        st.markdown(
            "#### 🌧️ Lluvia del evento"
        )

        rain_fig = go.Figure()

        rain_fig.add_trace(
            go.Bar(
                x=block[
                    "datetime"
                ],
                y=lluvia,
                name="Lluvia",
            )
        )

        rain_fig.update_layout(
            height=280,
            margin=dict(
                l=5,
                r=5,
                t=10,
                b=5,
            ),
            showlegend=False,
        )

        rain_fig.update_xaxes(
            tickformat="%d/%m",
        )

        rain_fig.update_yaxes(
            title_text="Precipitación (mm/día)",
            rangemode="tozero",
        )

        st.plotly_chart(
            rain_fig,
            use_container_width=True,
        )

    with flow_col:

        st.markdown(
            "#### 💧 Caudal del evento"
        )

        if (
            block[
                "caudal_m3s"
            ]
            .notna()
            .any()
        ):

            flow_fig = go.Figure()

            flow_fig.add_trace(
                go.Scatter(
                    x=block[
                        "datetime"
                    ],
                    y=block[
                        "caudal_m3s"
                    ],
                    mode="lines+markers",
                    name="Caudal",
                )
            )

            flow_fig.update_layout(
                height=280,
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=5,
                ),
                showlegend=False,
            )

            flow_fig.update_xaxes(
                tickformat="%d/%m",
            )

            flow_fig.update_yaxes(
                title_text="Caudal (m³/s)",
                rangemode="tozero",
            )

            st.plotly_chart(
                flow_fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No hay caudal histórico suficiente "
                "para este evento."
            )

    # ========================================================
    # AUDITORÍA DIARIA
    # ========================================================

    with st.expander(
        "🔎 Auditoría diaria del peor escenario · 60 días"
    ):

        tabla = (
            scenario_worst
            .copy()
        )

        tabla["Fecha"] = (
            pd.to_datetime(
                tabla["datetime"]
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        tabla["Nivel base"] = (
            tabla["nivel_base"]
            .round(2)
        )

        tabla["Variación diaria"] = (
            tabla["variacion_dia"]
            .round(3)
        )

        tabla["Crecimiento acumulado"] = (
            tabla[
                "crecimiento_desde_actual"
            ]
            .round(2)
        )

        tabla["Nivel escenario"] = (
            tabla["prediction"]
            .round(2)
        )

        tabla["Lluvia"] = (
            tabla["precip_mm"]
            .round(1)
        )

        tabla["Caudal"] = (
            tabla["caudal_m3s"]
            .round(0)
        )

        tabla["Aguas arriba"] = (
            tabla["upstream_mean"]
            .round(2)
        )

        tabla["Fecha histórica origen"] = (
            pd.to_datetime(
                tabla["source_date"],
                errors="coerce",
            )
            .dt
            .strftime(
                "%d/%m/%Y"
            )
        )

        columnas = [
            "Fecha",
            "Nivel base",
            "Lluvia",
            "Caudal",
            "Aguas arriba",
            "Variación diaria",
            "Crecimiento acumulado",
            "Nivel escenario",
            "Fecha histórica origen",
        ]

        if (
            "nivel_max_historico"
            in tabla.columns
        ):

            tabla["Máximo histórico"] = (
                tabla[
                    "nivel_max_historico"
                ]
                .round(2)
            )

            columnas.append(
                "Máximo histórico"
            )

        if (
            "nivel_min_historico"
            in tabla.columns
        ):

            tabla["Mínimo histórico"] = (
                tabla[
                    "nivel_min_historico"
                ]
                .round(2)
            )

            columnas.append(
                "Mínimo histórico"
            )

        st.dataframe(
            tabla[
                columnas
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # INTERPRETACIÓN
    # ========================================================

    with st.expander(
        "ℹ️ Cómo interpretar P90, P95 y peor caso"
    ):

        st.markdown(
            f"""
            Los tres escenarios parten exactamente del último
            nivel disponible de San Nicolás:

            **{nivel_actual:.2f} m**

            **P90**

            Representa aproximadamente una creciente ubicada en
            el percentil 90 de las crecientes históricas de
            60 días encontradas.

            **P95**

            Representa una condición histórica todavía más severa.

            **Peor caso histórico**

            Selecciona la ventana continua de 60 días que produjo
            la mayor suba de nivel observada dentro del historial.

            La trayectoria histórica completa se traslada al nivel
            actual. Por eso conserva las tres etapas:

            **crecida → pico → estabilización o bajante**.

            Después del pico se permite una disminución del nivel
            siguiendo la forma del episodio histórico.

            Sin embargo, como este módulo representa específicamente
            una condición severa de creciente, la trayectoria no se
            traslada por debajo del nivel inicial actual.

            Las líneas roja y verde indican respectivamente los
            máximos y mínimos históricos correspondientes al mismo
            día y mes del horizonte futuro.

            Todos los gráficos de nivel hidrométrico utilizan
            exactamente la misma escala:

            **0 a 7 metros**.
            """
        )

    st.warning(
        "P90, P95 y peor caso histórico son escenarios "
        "experimentales de estrés y no constituyen pronósticos "
        "ni alertas oficiales."
    )
