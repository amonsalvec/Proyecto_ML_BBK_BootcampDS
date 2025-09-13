# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, yaml

from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

# ========= Utilidades =========
REQUIRED = ["Issue","Sub-issue","Product","Sub-product","Company","State","Date received"]

def ensure_columns(df: pd.DataFrame):
    return [c for c in REQUIRED if c not in df.columns]

def scores_from_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X); smin, smax = float(np.min(s)), float(np.max(s))
        return (s - smin) / (smax - smin + 1e-9)
    return model.predict(X).astype(float)

def coerce_python(obj):
    """Convierte tipos numpy -> tipos Python puros para JSON."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

# ========= Descubrimiento de rutas (soporta ejecutar fuera de la carpeta) =========
def locate_repo_root(hint_file: Path) -> Path:
    """
    Busca hacia arriba un directorio que contenga 'models/trained_model.pkl'.
    Si no lo encuentra, devuelve la carpeta del archivo actual.
    """
    here = hint_file.resolve()
    for up in [here.parent, *here.parents]:
        candidate = up / "models" / "trained_model.pkl"
        if candidate.exists():
            return up
    return here.parent

# ========= App =========
app = FastAPI(title="API Scoring Timely Response", version="1.0.0")

# Rutas por defecto (robustas ante estructura de proyecto)
HERE = Path(__file__).resolve()
REPO_ROOT = locate_repo_root(HERE)

# Asegurar import de transformaciones custom si el pipeline las usa
for p in (REPO_ROOT, REPO_ROOT/"src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
try:
    import src.runtime_transforms  # noqa: F401
except Exception:
    # Si no existen, no pasa nada; solo era necesario si el pipeline las referencia
    pass

DEFAULT_MODEL = REPO_ROOT / "models" / "trained_model.pkl"
DEFAULT_CFG   = REPO_ROOT / "models" / "model_config.yaml"

# Cargar modelo al iniciar
MODEL = None
THR_DEFAULT = 0.5
MODEL_NAME = "desconocido"
try:
    MODEL = joblib.load(DEFAULT_MODEL)
    if DEFAULT_CFG.exists():
        cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
        THR_DEFAULT = float(cfg.get("threshold", 0.5))
        MODEL_NAME = str(cfg.get("model", MODEL_NAME))
except Exception as e:
    # No abortamos; pero levantamos 500 al primer uso si no hay modelo
    print(f"[WARN] No se pudo cargar el modelo en {DEFAULT_MODEL}: {e}")

# ========= Endpoints =========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model": MODEL_NAME,
        "repo_root": str(REPO_ROOT),
        "expected_columns": REQUIRED
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="CSV con columnas mínimas requeridas"),
    threshold: float = Query(default=THR_DEFAULT, ge=0.0, le=1.0, description="Umbral de clasificación")
):
    """
    Recibe un CSV, devuelve:
    - primeras 100 filas con score y pred,
    - KPIs de scoring,
    - métricas y matriz de confusión si 'Timely response?' está presente.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Verifica 'models/trained_model.pkl'.")

    try:
        df = pd.read_csv(file.file, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {e}")

    missing = ensure_columns(df)
    if missing:
        return JSONResponse(status_code=400, content={"error": f"Faltan columnas requeridas: {missing}"})

    try:
        scores = scores_from_model(MODEL, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al puntuar con el modelo: {e}")

    pred = (scores >= threshold).astype(int)

    # KPIs
    info = {
        "modelo": MODEL_NAME,
        "umbral": float(threshold),
        "filas": int(len(df)),
        "score_medio": float(np.mean(scores)),
        "rate_predicho_yes": float((pred == 1).mean())
    }

    # Resultados (muestra)
    df_out = df.copy()
    df_out["score"] = scores
    df_out["pred"] = pred
    sample_records = df_out.head(100).to_dict(orient="records")
    sample_records = [{k: coerce_python(v) for k, v in row.items()} for row in sample_records]

    response = {
        "info": info,
        "resultados_muestra": sample_records
    }

    # Métricas si existe ground-truth
    if "Timely response?" in df.columns:
        y_true = df["Timely response?"].astype("string").str.strip().str.lower().map(
            {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}
        )
        mask = y_true.notna()
        if mask.any():
            y = y_true[mask].astype(int).to_numpy()
            y_pred = pred[mask]
            try:
                roc = roc_auc_score(y, scores[mask])
            except Exception:
                roc = float("nan")
            try:
                pr = average_precision_score(y, scores[mask])
            except Exception:
                pr = float("nan")

            response["metricas"] = {
                "BalancedAccuracy": float(balanced_accuracy_score(y, y_pred)),
                "F1_No": float(f1_score((y == 0).astype(int), (y_pred == 0).astype(int))),
                "F1_Yes": float(f1_score(y, y_pred)),
                "ROC_AUC": float(roc),
                "PR_AUC": float(pr),
                "ConfusionMatrix": confusion_matrix(y, y_pred, labels=[0, 1]).astype(int).tolist()
            }
        else:
            response["metricas"] = {"info": "La columna 'Timely response?' existe pero sin valores válidos (Yes/No)."}

    return response

@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(..., description="CSV con columnas mínimas requeridas"),
    threshold: float = Query(default=THR_DEFAULT, ge=0.0, le=1.0)
):
    """
    Igual que /predict, pero devuelve un CSV con columnas extra: score, pred.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Verifica 'models/trained_model.pkl'.")

    try:
        df = pd.read_csv(file.file, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {e}")

    missing = ensure_columns(df)
    if missing:
        return JSONResponse(status_code=400, content={"error": f"Faltan columnas requeridas: {missing}"})

    try:
        scores = scores_from_model(MODEL, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al puntuar con el modelo: {e}")

    pred = (scores >= threshold).astype(int)

    df_out = df.copy()
    df_out["score"] = scores
    df_out["pred"] = pred

    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored.csv"}
    )
