import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.stress_scenario import (
    build_stress_scenario,
)


# ============================================================
# INTERFAZ DEL ESCENARIO
# ============================================================

def render_stress_scenario(
    df,
    models,
    exog_history,
    upstream_history,
):

    st.divider()

    st.subheader(
        "⚠️ Escenario hipotético de máximos históricos · 60 días"
    )

    st.markdown(
        """
        Simulación de tipo **“qué pasa si”** que evalúa la respuesta
        del modelo cuando condiciones históricamente elevadas de
        **lluvia y caudal coinciden en una misma fecha futura**.

        No representa la probabilidad de que esas condiciones ocurran.
        """
    )

    # ========================================================
    # SELECTOR
    # ========================================================

    scenario_label = st.selectbox(
        "Escenario",
        options=[
            "Alto",
            "Severo",
            "Extremo histórico",
        ],
        index=2,
    )

    scenario_map = {
        "Alto": "alto",
        "Severo": "severo",
        "Extremo histórico": "extremo",
    }

    scenario_key = scenario_map[
        scenario_label
    ]

    # ========================================================
    # CALCULAR
    # ========================================================

    try:

        (
            scenario_df,
            meta,
        ) = build_stress_scenario(
            models=models,
            exog_history=exog_history,
            upstream_history=upstream_history,
            days=60,
            scenario=scenario_key,
        )

    except Exception as exc:

        st.warning(
            "No fue posible construir "
            f"el escenario: {exc}"
        )

        return

    if scenario_df.empty:

        st.info(
            "No hay información suficiente."
        )

        return

    # ========================================================
    # METADATA
    # ========================================================

    current_level = meta.get(
        "current_level"
    )

    max_level = meta.get(
        "max_level"
    )

    growth_m = meta.get(
        "growth_m"
    )

    growth_pct = meta.get(
        "growth_pct"
    )

    level30 = meta.get(
        "level_day_30"
    )

    level60 = meta.get(
        "level_day_60"
    )

    rain_peak = meta.get(
        "rain_peak_scenario"
    )

    rain_total = meta.get(
        "rain_event_total"
    )

    flow_peak = meta.get(
        "flow_scenario_max"
    )

    peak_date = meta.get(
        "peak_future_date"
    )

    max_level_date = meta.get(
        "max_level_date"
    )

    # ========================================================
    # PRIMERA FILA
    # ========================================================

    a1, a2, a3, a4 = st.columns(
        4
    )

    a1.metric(
        "Nivel actual",
        (
            f"{current_level:.2f} m"
            if current_level
            is not None
            else "Sin datos"
        ),
    )

    a2.metric(
        "Máximo estimado",
        (
            f"{max_level:.2f} m"
            if max_level
            is not None
            else "Sin datos"
        ),
    )

    a3.metric(
        "Crecimiento probable",
        (
            f"{growth_m:+.2f} m"
            if growth_m
            is not None
            else "Sin datos"
        ),
    )

    a4.metric(
        "Variación relativa",
        (
            f"{growth_pct:+.1f}%"
            if growth_pct
            is not None
            and pd.notna(
                growth_pct
            )
            else "Sin datos"
        ),
    )

    # ========================================================
    # SEGUNDA FILA
    # ========================================================

    b1, b2, b3, b4 = st.columns(
        4
    )

    b1.metric(
        "Lluvia máxima aplicada",
        (
            f"{rain_peak:.1f} mm"
            if rain_peak
            is not None
            else "Sin datos"
        ),
    )

    b2.metric(
        "Lluvia acumulada del evento",
        (
            f"{rain_total:.1f} mm"
            if rain_total
            is not None
            else "Sin datos"
        ),
    )

    b3.metric(
        "Caudal máximo aplicado",
        (
            f"{flow_peak:,.0f} m³/s"
            if flow_peak
            is not None
            and pd.notna(
                flow_peak
            )
            else "Sin datos"
        ),
    )

    if peak_date is not None:

        peak_date_text = pd.to_datetime(
            peak_date
        ).strftime(
            "%d/%m/%Y"
        )

    else:

        peak_date_text = (
            "Sin datos"
        )

    b4.metric(
        "Fecha máximos conjuntos",
        peak_date_text,
    )

    # ========================================================
    # TERCERA FILA
    # ========================================================

    c1, c2, c3 = st.columns(
        3
    )

    c1.metric(
        "Referencia día 30",
        (
            f"{level30:.2f} m"
            if level30
            is not None
            else "Sin datos"
        ),
    )

    c2.metric(
        "Referencia día 60",
        (
            f"{level60:.2f} m"
            if level60
            is not None
            else "Sin datos"
        ),
    )

    if max_level_date is not None:

        max_date_text = pd.to_datetime(
            max_level_date
        ).strftime(
            "%d/%m/%Y"
        )

    else:

        max_date_text = (
            "Sin datos"
        )

    c3.metric(
        "Fecha máximo nivel",
        max_date_text,
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = go.Figure()

    # --------------------------------------------------------
    # OBSERVADO
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
            )
        )

    # --------------------------------------------------------
    # INCERTIDUMBRE
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
    # ESCENARIO
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
        )
    )

    # ========================================================
    # LÍNEA VERTICAL:
    # MÁXIMA LLUVIA + MÁXIMO CAUDAL
    # ========================================================

    if peak_date is not None:

        fig.add_vline(
            x=pd.to_datetime(
                peak_date
            ).timestamp()
            * 1000,
            line_width=2,
            line_dash="dot",
            annotation_text=(
                "Máx. lluvia + máx. caudal"
            ),
            annotation_position="top",
        )

    # ========================================================
    # MARCAR MÁXIMO NIVEL
    # ========================================================

    if (
        max_level_date
        is not None
        and max_level
        is not None
    ):

        fig.add_trace(
            go.Scatter(
                x=[
                    max_level_date
                ],
                y=[
                    max_level
                ],
                mode="markers+text",
                marker=dict(
                    size=12,
                    symbol="diamond",
                ),
                text=[
                    f"{max_level:.2f} m"
                ],
                textposition=(
                    "top center"
                ),
                name="Máximo nivel estimado",
            )
        )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.05,
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
    # EXPLICACIÓN
    # ========================================================

    st.caption(
        "La línea vertical identifica la fecha hipotética en la que "
        "el máximo de precipitación del escenario y el máximo de "
        "caudal coinciden. El crecimiento mostrado corresponde a "
        "la respuesta del modelo ante esas condiciones."
    )

    # ========================================================
    # REFERENCIAS HISTÓRICAS
    # ========================================================

    with st.expander(
        "📊 Ver máximos históricos utilizados"
    ):

        rain_stats = meta.get(
            "rain_stats",
            {},
        )

        flow_stats = meta.get(
            "flow_stats",
            {},
        )

        st.write(
            "### Precipitación"
        )

        st.write(
            "**Máximo diario histórico:**",
            f"{rain_stats.get('max_day', 0):.1f} mm",
        )

        if rain_stats.get(
            "max_day_date"
        ) is not None:

            st.write(
                "**Fecha histórica:**",
                pd.to_datetime(
                    rain_stats[
                        "max_day_date"
                    ]
                ).strftime(
                    "%d/%m/%Y"
                ),
            )

        st.write(
            "**Máximo acumulado 3 días:**",
            f"{rain_stats.get('max_3d', 0):.1f} mm",
        )

        st.write(
            "**Máximo acumulado 7 días:**",
            f"{rain_stats.get('max_7d', 0):.1f} mm",
        )

        st.write(
            "### Caudal"
        )

        if pd.notna(
            flow_stats.get(
                "maximum"
            )
        ):

            st.write(
                "**Máximo histórico:**",
                f"{flow_stats['maximum']:,.0f} m³/s",
            )

        if flow_stats.get(
            "maximum_date"
        ) is not None:

            st.write(
                "**Fecha histórica:**",
                pd.to_datetime(
                    flow_stats[
                        "maximum_date"
                    ]
                ).strftime(
                    "%d/%m/%Y"
                ),
            )

        st.write(
            "### Fecha trasladada al escenario"
        )

        st.write(
            "Los máximos históricos de lluvia y caudal "
            "se hacen coincidir hipotéticamente el:",
            f"**{peak_date_text}**",
        )

    # ========================================================
    # TABLA
    # ========================================================

    with st.expander(
        "📋 Ver simulación día por día"
    ):

        table = scenario_df.copy()

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
            "Nivel (m)"
        ] = pd.to_numeric(
            table[
                "prediction"
            ],
            errors="coerce",
        ).round(
            2
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

        st.dataframe(
            table[
                [
                    "Fecha",
                    "Nivel (m)",
                    "Lluvia (mm)",
                    "Caudal (m³/s)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # ADVERTENCIA
    # ========================================================

    st.warning(
        "Este escenario es una simulación hipotética de estrés. "
        "No significa que el máximo histórico de lluvia y el máximo "
        "histórico de caudal vayan a ocurrir simultáneamente. "
        "Tampoco constituye un pronóstico oficial ni un modelo "
        "hidráulico de propagación de crecidas."
    )
