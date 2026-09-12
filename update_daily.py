"""
PARANÁ · SAN NICOLÁS
Actualización programada V11.19.

Genera el resultado diario cuando corresponde,
conserva el pronóstico mensual original y
actualiza las lecturas reales disponibles.
"""

from src.monthly import run_update


if __name__ == "__main__":
    run_update()

    print(
        "Resultado diario y archivo mensual "
        "actualizados correctamente."
    )
