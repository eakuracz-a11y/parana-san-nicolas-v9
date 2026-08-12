# Paraná San Nicolás V9 — Página pública

Plataforma Streamlit para consulta, visualización y predicción experimental del nivel del río Paraná en San Nicolás.

## Estructura

```text
parana-san-nicolas-v9/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── ina.py
    ├── model.py
    └── scenario.py
```

## Publicación en GitHub + Streamlit

1. Crear un repositorio público llamado `parana-san-nicolas-v9`.
2. Subir **todos los archivos y la carpeta `src`** manteniendo la estructura.
3. Abrir Streamlit Community Cloud:
   https://share.streamlit.io/
4. Elegir **Create app**.
5. Repository: `tuusuario/parana-san-nicolas-v9`
6. Branch: `main`
7. Main file: `app.py`
8. Elegir un subdominio, por ejemplo `parana-san-nicolas-v9`.
9. Presionar **Deploy**.

La URL final será similar a:

`https://parana-san-nicolas-v9.streamlit.app`

## Importante

La aplicación es experimental y no constituye un sistema oficial de alerta. Los escenarios de lluvia y el semáforo deben calibrarse y validarse con datos históricos antes de cualquier uso operativo.

El módulo `src/ina.py` aísla el acceso al servicio público del INA. Si el INA modifica su API, ese módulo deberá actualizarse.
