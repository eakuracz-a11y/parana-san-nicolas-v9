import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.stress_scenario import (
    build_stress_scenario,
)


# ============================================================
# FORMATEADORES
# ============================================================

def format_date(
    value,
):

    if value is None:

        return "Sin datos"

    try:

        return pd.to_datetime(
            value
        ).strftime(
            "%d/%m/%Y"
        )

    except Exception:

        return str(
            value
        )


# ============================================================
# INTERFAZ
# ============================================================

def render_stress_scenario(
    df,
    models,
    exog_history,
    upstream_history,
):

    st.divider()

    st.subheader(
        "⚠️ Escenario hipotético de creciente · 60 días"
    )

    st.markdown(
        """
        Simulación de tipo **“qué pasa si”**.

        Estima la respuesta potencial del nivel en San Nicolás
        cuando coinciden condiciones históricamente elevadas de
        **precipitación, caudal y niveles aguas arriba**.

        El crecimiento se calcula con un modelo específico
        calibrado sobre las respuestas de creciente observadas
        en el histórico disponible.
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
        key="stress_scenario_selector",
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
            exog_history=(
                exog_history
            ),
            upstream_history=(
                upstream_history
            ),
            days=60,
            scenario=(
                scenario_key
            ),
        )

    except Exception as exc:

        st.warning(
            "No fue posible construir "
            f"el escenario de creciente: {exc}"
        )

        return

    if scenario_df.empty:

        st.info(
            "No existe información suficiente "
            "para construir el escenario."
        )

        return

    # ========================================================
    # DATOS PRINCIPALES
    # ========================================================

    current_level = meta.get(
        "current_level"
    )

    max_level = meta.get(
        "max_level"
    )

    growth = meta.get(
        "growth_m"
    )

    growth_pct = meta.get(
        "growth_pct"
    )

    forcing_date = meta.get(
        "peak_future_date"
    )

    level_peak_date = meta.get(
        "max_level_date"
    )

    response_lag = meta.get(
        "response_lag_days"
    )

    rain_peak = meta.get(
        "rain_peak_scenario"
    )

    rain3 = meta.get(
        "rain_3d_scenario"
    )

    rain7 = meta.get(
        "rain_7d_scenario"
    )

    flow_peak = meta.get(
        "flow_scenario_max"
    )

    level30 = meta.get(
        "level_day_30"
    )

    level60 = meta.get(
        "level_day_60"
    )

    # ========================================================
    # FILA 1
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
        "Crecimiento estimado",
        (
            f"+{growth:.2f} m"
            if growth
            is not None
            else "Sin datos"
        ),
    )

    a4.metric(
        "Variación relativa",
        (
            f"+{growth_pct:.1f}%"
            if growth_pct
            is not None
            and pd.notna(
                growth_pct
            )
            else "Sin datos"
        ),
    )

    # ========================================================
    # FILA 2
    # ========================================================

    b1, b2, b3, b4 = st.columns(
        4
    )

    b1.metric(
        "Lluvia máxima diaria",
        (
            f"{rain_peak:.1f} mm"
            if rain_peak
            is not None
            else "Sin datos"
        ),
    )

    b2.metric(
        "Lluvia acumulada 7 días",
        (
            f"{rain7:.1f} mm"
            if rain7
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

    b4.metric(
        "Máximos conjuntos",
        format_date(
            forcing_date
        ),
    )

    # ========================================================
    # FILA 3
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Retardo estimado",
        (
            f"{response_lag} días"
            if response_lag
            is not None
            else "Sin datos"
        ),
    )

    c2.metric(
        "Fecha máximo nivel",
        format_date(
            level_peak_date
        ),
    )

    c3.metric(
        "Nivel día 30",
        (
            f"{level30:.2f} m"
            if level30
            is not None
            else "Sin datos"
        ),
    )

    c4.metric(
        "Nivel día 60",
        (
            f"{level60:.2f} m"
            if level60
            is not None
            else "Sin datos"
        ),
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
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Observado: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    # --------------------------------------------------------
    # BANDA
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
            name="Incertidumbre experimental",
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
            customdata=(
                scenario_df[
                    [
                        "precip_mm",
                        "caudal_m3s",
                    ]
                ]
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}"
                "<br>Nivel: %{y:.2f} m"
                "<br>Lluvia: %{customdata[0]:.1f} mm"
                "<br>Caudal: %{customdata[1]:,.0f} m³/s"
                "<extra></extra>"
            ),
        )
    )

    # ========================================================
    # FECHA MÁXIMOS CONJUNTOS
    # ========================================================

    if forcing_date is not None:

        forcing_ts = pd.to_datetime(
            forcing_date
        )

        fig.add_vline(
            x=(
                forcing_ts.timestamp()
                * 1000
            ),
            line_width=2,
            line_dash="dot",
            annotation_text=(
                "Máx. lluvia + máx. caudal"
            ),
            annotation_position="top",
        )

    # ========================================================
    # FECHA MÁXIMO DEL NIVEL
    # ========================================================

    if (
        level_peak_date
        is not None
        and max_level
        is not None
    ):

        fig.add_trace(
            go.Scatter(
                x=[
                    level_peak_date
                ],
                y=[
                    max_level
                ],
                mode="markers+text",
                marker=dict(
                    size=13,
                    symbol="diamond",
                ),
                text=[
                    f"{max_level:.2f} m"
                ],
                textposition="top center",
                name="Máximo nivel estimado",
                hovertemplate=(
                    "%{x|%d/%m/%Y}"
                    "<br>"
                    "Máximo nivel: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=560,
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
    # INTERPRETACIÓN
    # ========================================================

    if (
        forcing_date is not None
        and level_peak_date is not None
    ):

        st.info(
            "En este escenario, el máximo de lluvia y "
            "el máximo de caudal se hacen coincidir el "
            f"**{format_date(forcing_date)}**. "
            "Según el retardo aprendido de las respuestas "
            "históricas, el máximo del nivel se ubica "
            f"aproximadamente el **{format_date(level_peak_date)}**."
        )

    # ========================================================
    # MÁXIMOS HISTÓRICOS
    # ========================================================

    with st.expander(
        "📊 Máximos históricos utilizados"
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
            "**Máximo diario detectado:** "
            f"{rain_stats.get('max_day', 0):.1f} mm"
        )

        st.write(
            "**Máximo acumulado 3 días:** "
            f"{rain_stats.get('max_3d', 0):.1f} mm"
        )

        st.write(
            "**Máximo acumulado 7 días:** "
            f"{rain_stats.get('max_7d', 0):.1f} mm"
        )

        if (
            rain_stats.get(
                "max_day_date"
            )
            is not None
        ):

            st.write(
                "**Fecha del máximo diario histórico:** "
                + format_date(
                    rain_stats[
                        "max_day_date"
                    ]
                )
            )

        st.write(
            "### Caudal"
        )

        maximum_flow = (
            flow_stats.get(
                "maximum"
            )
        )

        if (
            maximum_flow
            is not None
            and pd.notna(
                maximum_flow
            )
        ):

            st.write(
                "**Máximo histórico detectado:** "
                f"{maximum_flow:,.0f} m³/s"
            )

        if (
            flow_stats.get(
                "maximum_date"
            )
            is not None
        ):

            st.write(
                "**Fecha histórica del máximo:** "
                + format_date(
                    flow_stats[
                        "maximum_date"
                    ]
                )
            )

    # ========================================================
    # MODELO DE CRECIENTE
    # ========================================================

    with st.expander(
        "🧠 Diagnóstico del modelo de creciente"
    ):

        st.write(
            "**Registros de calibración:**",
            meta.get(
                "flood_training_rows",
                "-",
            ),
        )

        rmse = meta.get(
            "flood_model_rmse"
        )

        mae = meta.get(
            "flood_model_mae"
        )

        if rmse is not None:

            st.write(
                "**RMSE del crecimiento:** "
                f"{rmse:.3f} m"
            )

        if mae is not None:

            st.write(
                "**MAE del crecimiento:** "
                f"{mae:.3f} m"
            )

        historical_p95 = meta.get(
            "historical_p95_growth"
        )

        historical_max = meta.get(
            "historical_max_growth"
        )

        if historical_p95 is not None:

            st.write(
                "**Crecimiento P95 observado en 14 días:** "
                f"{historical_p95:.2f} m"
            )

        if historical_max is not None:

            st.write(
                "**Máximo crecimiento observado en 14 días:** "
                f"{historical_max:.2f} m"
            )

        st.caption(
            "Este segundo modelo estima crecimiento máximo "
            "posterior y retardo de respuesta a partir del "
            "histórico disponible. Es independiente del "
            "Random Forest utilizado para el pronóstico "
            "operativo de 15 días."
        )

    # ========================================================
    # DÍA POR DÍA
    # ========================================================

    with st.expander(
        "📋 Ver escenario día por día"
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
    # AVISO
    # ========================================================

    st.warning(
        "Este escenario es una prueba de estrés. "
        "No expresa la probabilidad de que los máximos históricos "
        "de lluvia, caudal y niveles aguas arriba ocurran "
        "simultáneamente. Tampoco sustituye un modelo hidráulico "
        "ni los pronósticos de organismos oficiales."
    )
