import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.stress_scenario import (
    build_stress_scenario,
)


# ============================================================
# INTERFAZ ESCENARIO DE ESTRÉS
# ============================================================

def render_stress_scenario(
    df,
    models,
    exog_history,
    upstream_history,
):

    st.divider()

    st.subheader(
        "⚠️ Escenario hipotético de estrés · 60 días"
    )

    st.markdown(
        """
        Esta sección responde a una pregunta de tipo
        **“qué pasa si”**.

        Simula la respuesta del modelo cuando coinciden
        condiciones históricamente elevadas de
        **precipitación, caudal y niveles aguas arriba**.
        """
    )

    scenario_label = st.selectbox(
        "Escenario",
        options=[
            "Alto",
            "Severo",
            "Extremo histórico",
        ],
        index=1,
        help=(
            "Alto utiliza aproximadamente el percentil 90; "
            "Severo el percentil 95; Extremo histórico utiliza "
            "los máximos disponibles en el período analizado."
        ),
    )

    map_scenario = {
        "Alto": "alto",
        "Severo": "severo",
        "Extremo histórico": "extremo",
    }

    scenario_key = map_scenario[
        scenario_label
    ]

    try:

        scenario_df, meta = (
            build_stress_scenario(
                models=models,
                exog_history=exog_history,
                upstream_history=upstream_history,
                days=60,
                scenario=scenario_key,
            )
        )

    except Exception as exc:

        st.warning(
            "No fue posible generar el escenario "
            f"hipotético: {exc}"
        )

        return

    if scenario_df.empty:

        st.info(
            "No existen datos suficientes para "
            "construir el escenario."
        )

        return

    # ========================================================
    # VALORES PRINCIPALES
    # ========================================================

    current_level = None

    if (
        isinstance(
            df,
            pd.DataFrame,
        )
        and not df.empty
        and "nivel"
        in df.columns
    ):

        values = pd.to_numeric(
            df[
                "nivel"
            ],
            errors="coerce",
        ).dropna()

        if len(values):

            current_level = float(
                values.iloc[-1]
            )

    max_level = meta.get(
        "max_level"
    )

    max_date = meta.get(
        "max_level_date"
    )

    level30 = meta.get(
        "level_day_30"
    )

    level60 = meta.get(
        "level_day_60"
    )

    flow_max = meta.get(
        "flow_scenario_max"
    )

    rain_total = meta.get(
        "rain_event_total",
        0.0,
    )

    # ========================================================
    # FILA 1 DE MÉTRICAS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Nivel actual",
        (
            f"{current_level:.2f} m"
            if current_level
            is not None
            else "Sin datos"
        ),
    )

    c2.metric(
        "Máximo del escenario",
        (
            f"{max_level:.2f} m"
            if max_level
            is not None
            else "Sin datos"
        ),
    )

    c3.metric(
        "Referencia día 30",
        (
            f"{level30:.2f} m"
            if level30
            is not None
            else "Sin datos"
        ),
    )

    c4.metric(
        "Referencia día 60",
        (
            f"{level60:.2f} m"
            if level60
            is not None
            else "Sin datos"
        ),
    )

    # ========================================================
    # FILA 2
    # ========================================================

    c5, c6, c7 = st.columns(
        3
    )

    c5.metric(
        "Lluvia del evento",
        f"{rain_total:.1f} mm",
    )

    c6.metric(
        "Caudal máximo escenario",
        (
            f"{flow_max:,.0f} m³/s"
            if flow_max is not None
            and pd.notna(
                flow_max
            )
            else "Sin datos"
        ),
    )

    if max_date is not None:

        try:

            max_date_text = pd.to_datetime(
                max_date
            ).strftime(
                "%d/%m/%Y"
            )

        except Exception:

            max_date_text = str(
                max_date
            )

    else:

        max_date_text = (
            "Sin datos"
        )

    c7.metric(
        "Fecha del máximo",
        max_date_text,
    )

    # ========================================================
    # GRÁFICO DEL ESCENARIO
    # ========================================================

    fig = go.Figure()

    # --------------------------------------------------------
    # OBSERVADO RECIENTE
    # --------------------------------------------------------

    if (
        isinstance(
            df,
            pd.DataFrame,
        )
        and not df.empty
    ):

        obs = df.tail(
            45
        )

        fig.add_trace(
            go.Scatter(
                x=obs[
                    "datetime"
                ],
                y=obs[
                    "nivel"
                ],
                mode="lines",
                name="Nivel observado",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Observado: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # BANDA SUPERIOR
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "upper"
            ],
            mode="lines",
            line=dict(
                width=0,
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # BANDA INFERIOR
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "lower"
            ],
            mode="lines",
            line=dict(
                width=0,
            ),
            fill="tonexty",
            name="Incertidumbre del escenario",
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # NIVEL DEL ESCENARIO
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=scenario_df[
                "datetime"
            ],
            y=scenario_df[
                "prediction"
            ],
            mode="lines+markers",
            line=dict(
                dash="dash",
                width=3,
            ),
            marker=dict(
                size=5,
            ),
            name=(
                f"Escenario {scenario_label}"
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>"
                "Nivel escenario: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=540,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.04,
        ),
    )

    fig.update_xaxes(
        title_text="Fecha",
        type="date",
        tickformat="%d/%m/%Y",
    )

    fig.update_yaxes(
        title_text=(
            "Nivel hidrométrico (m)"
        ),
        range=[
            0,
            7,
        ],
        dtick=0.5,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # FORZANTES DEL ESCENARIO
    # ========================================================

    with st.expander(
        "🌧️💧 Ver condiciones impuestas al escenario"
    ):

        table = scenario_df[
            [
                "datetime",
                "precip_mm",
                "caudal_m3s",
                "prediction",
            ]
        ].copy()

        table[
            "Fecha"
        ] = pd.to_datetime(
            table[
                "datetime"
            ]
        ).dt.strftime(
            "%d/%m/%Y"
        )

        table[
            "Lluvia (mm)"
        ] = pd.to_numeric(
            table[
                "precip_mm"
            ],
            errors="coerce",
        ).round(
            1
        )

        table[
            "Caudal (m³/s)"
        ] = pd.to_numeric(
            table[
                "caudal_m3s"
            ],
            errors="coerce",
        ).round(
            0
        )

        table[
            "Nivel simulado (m)"
        ] = pd.to_numeric(
            table[
                "prediction"
            ],
            errors="coerce",
        ).round(
            2
        )

        st.dataframe(
            table[
                [
                    "Fecha",
                    "Lluvia (mm)",
                    "Caudal (m³/s)",
                    "Nivel simulado (m)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # REFERENCIAS HISTÓRICAS
    # ========================================================

    rain_stats = meta.get(
        "rain_stats",
        {},
    )

    flow_stats = meta.get(
        "flow_stats",
        {},
    )

    with st.expander(
        "📊 Referencias históricas utilizadas"
    ):

        st.write(
            "**Precipitación histórica del período cargado**"
        )

        st.write(
            "Percentil 90 diario:",
            f"{rain_stats.get('p90_day', 0):.1f} mm",
        )

        st.write(
            "Percentil 95 diario:",
            f"{rain_stats.get('p95_day', 0):.1f} mm",
        )

        st.write(
            "Máximo diario:",
            f"{rain_stats.get('max_day', 0):.1f} mm",
        )

        st.write(
            "Máximo acumulado 3 días:",
            f"{rain_stats.get('max_3d', 0):.1f} mm",
        )

        st.write(
            "Máximo acumulado 7 días:",
            f"{rain_stats.get('max_7d', 0):.1f} mm",
        )

        st.write(
            "**Caudal histórico del período cargado**"
        )

        current_q = flow_stats.get(
            "current"
        )

        p90_q = flow_stats.get(
            "p90"
        )

        p95_q = flow_stats.get(
            "p95"
        )

        max_q = flow_stats.get(
            "maximum"
        )

        if pd.notna(
            current_q
        ):

            st.write(
                "Caudal actual:",
                f"{current_q:,.0f} m³/s",
            )

        if pd.notna(
            p90_q
        ):

            st.write(
                "Percentil 90:",
                f"{p90_q:,.0f} m³/s",
            )

        if pd.notna(
            p95_q
        ):

            st.write(
                "Percentil 95:",
                f"{p95_q:,.0f} m³/s",
            )

        if pd.notna(
            max_q
        ):

            st.write(
                "Máximo:",
                f"{max_q:,.0f} m³/s",
            )

    # ========================================================
    # ADVERTENCIA
    # ========================================================

    st.warning(
        "Este gráfico es una simulación hipotética de estrés. "
        "No indica que estas condiciones vayan a ocurrir y no "
        "constituye un pronóstico hidrológico oficial. "
        "Un Random Forest tampoco es un modelo hidráulico y "
        "tiene capacidad limitada para extrapolar niveles fuera "
        "de los rangos observados durante su entrenamiento."
    )
