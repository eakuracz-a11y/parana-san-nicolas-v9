import streamlit as st
import traceback

st.set_page_config(
    page_title="Paraná · San Nicolás",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 PARANÁ · SAN NICOLÁS")
st.caption("V12.0 · Diagnóstico de inicio")

st.write(
    "Verificando los módulos necesarios para iniciar la aplicación..."
)

errores = []


# ============================================================
# INA
# ============================================================

try:
    from src.ina import observed

    st.success(
        "✅ src/ina.py cargado correctamente"
    )

except Exception as exc:

    errores.append(
        (
            "src/ina.py",
            exc,
            traceback.format_exc(),
        )
    )

    st.error(
        f"❌ Error en src/ina.py: {exc}"
    )


# ============================================================
# MODEL
# ============================================================

try:

    from src.model import (
        train,
        predict,
        resumen_niveles_estaciones,
    )

    st.success(
        "✅ src/model.py cargado correctamente"
    )

except Exception as exc:

    errores.append(
        (
            "src/model.py",
            exc,
            traceback.format_exc(),
        )
    )

    st.error(
        f"❌ Error en src/model.py: {exc}"
    )


# ============================================================
# EXOGENOUS
# ============================================================

try:

    from src.exogenous import (
        get_exogenous_data,
    )

    st.success(
        "✅ src/exogenous.py cargado correctamente"
    )

except Exception as exc:

    errores.append(
        (
            "src/exogenous.py",
            exc,
            traceback.format_exc(),
        )
    )

    st.error(
        f"❌ Error en src/exogenous.py: {exc}"
    )


# ============================================================
# UPSTREAM
# ============================================================

try:

    from src.upstream import (
        get_upstream_history,
    )

    st.success(
        "✅ src/upstream.py cargado correctamente"
    )

except Exception as exc:

    errores.append(
        (
            "src/upstream.py",
            exc,
            traceback.format_exc(),
        )
    )

    st.error(
        f"❌ Error en src/upstream.py: {exc}"
    )


# ============================================================
# STRESS
# ============================================================

try:

    from src.stress_ui import (
        render_stress_scenario,
    )

    st.success(
        "✅ src/stress_ui.py cargado correctamente"
    )

except Exception as exc:

    st.warning(
        f"⚠️ src/stress_ui.py todavía no es compatible: {exc}"
    )


# ============================================================
# RESULTADO
# ============================================================

st.divider()

if not errores:

    st.success(
        "✅ Todos los módulos principales cargaron correctamente."
    )

    st.info(
        "La base V12 está funcionando. "
        "Podemos volver a colocar la interfaz completa."
    )

else:

    st.error(
        f"Se detectaron {len(errores)} errores."
    )

    for archivo, exc, detalle in errores:

        st.subheader(
            f"Error encontrado en {archivo}"
        )

        st.code(
            detalle,
            language="text",
        )


st.divider()

st.caption(
    "Paraná · San Nicolás V12.0 · Diagnóstico"
)
