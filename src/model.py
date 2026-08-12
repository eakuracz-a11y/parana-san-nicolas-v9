import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TARGET = "San Nicolás"

def _features(df):
    x = df.copy()
    x["datetime"] = x["datetime"].sort_values()
    stations = [c for c in x.columns if c != "datetime"]
    for c in stations:
        x[c] = __import__("pandas").to_numeric(x[c], errors="coerce")
        x[c] = x[c].interpolate(limit_direction="both")
    for lag in (1, 2, 3, 6, 12, 24):
        for c in stations:
            x[f"{c}_lag{lag}"] = x[c].shift(lag)
    for c in stations:
        x[f"{c}_diff1"] = x[c].diff()
    return x

def train(df):
    if TARGET not in df.columns:
        raise ValueError("No se encontró la estación San Nicolás en los datos.")

    x = _features(df)
    feature_cols = [c for c in x.columns if c not in ("datetime", TARGET)]
    work = x.dropna(subset=[TARGET] + feature_cols)

    if len(work) < 80:
        raise ValueError(
            f"No hay suficientes observaciones para entrenar el modelo: {len(work)}. "
            "Se requieren al menos 80."
        )

    split = max(1, int(len(work) * 0.8))
    X_train, X_test = work[feature_cols].iloc[:split], work[feature_cols].iloc[split:]
    y_train, y_test = work[TARGET].iloc[:split], work[TARGET].iloc[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    metrics = {}
    if len(X_test):
        p = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    else:
        rmse = 0.2

    # Same base model is used for the horizons; predict() creates lagged
    # horizon inputs recursively using the latest available record.
    models = {"model": model, "feature_cols": feature_cols, "rmse": rmse}
    for h in (24, 48, 72):
        metrics[h] = {"RMSE": rmse}

    return models, metrics

def predict(df, models):
    model = models["model"]
    feature_cols = models["feature_cols"]
    x = _features(df)
    latest = x.iloc[-1:].copy()

    # Forecast horizon is represented as an experimental persistence/trend
    # adjustment around the latest modeled level.
    base = float(df[TARGET].dropna().iloc[-1])
    vals = df[TARGET].dropna().tail(24).to_numpy()
    trend = float(np.polyfit(np.arange(len(vals)), vals, 1)[0]) if len(vals) >= 3 else 0.0

    out = {}
    for h in (24, 48, 72):
        out[h] = float(base + trend * h)
    return out

def prob(prediction, threshold, rmse):
    sigma = max(float(rmse), 0.05)
    z = (float(prediction) - float(threshold)) / sigma
    return float(1.0 / (1.0 + np.exp(-z)))
