
# -*- coding: utf-8 -*-
# App de scoring PRODUCCIÓN (sin métricas de negocio): clara para ejecutivos.
# - Carga automática de modelo (models/trained_model.pkl) y dataset (data/processed/clean_for_model.csv)
# - Si hay etiqueta "Timely response?" en el CSV, calcula métricas y muestra matriz de confusión
# - Si no hay etiqueta, muestra resultados de scoring y KPIs de predicción
# - Umbral ajustable; descarga de scored.csv; gráficos simples y legibles

import io, sys
from pathlib import Path as P

import numpy as np
import pandas as pd
import joblib, yaml
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
)

# ---------- Utilidades ----------
REQUIRED = ["Issue","Sub-issue","Product","Sub-product","Company","State","Date received"]

def ensure_columns(df: pd.DataFrame):
    """Valida columnas mínimas; si faltan, advertir. No rellena (para no engañar)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    return missing

def scores_from_model(model, X: pd.DataFrame) -> np.ndarray:
    """Obtiene score en [0,1]."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X); smin, smax = float(np.min(s)), float(np.max(s))
        return (s - smin)/(smax - smin + 1e-9)
    return model.predict(X).astype(float)

# ---------- Carga artefactos ----------
st.set_page_config(page_title="Timely Response — Producción", layout="wide")
repo_root = P(__file__).resolve().parents[1]
for p in (repo_root, repo_root/"src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
# Necesario si el pipeline contiene FunctionTransformer con funciones de src.runtime_transforms
try:
    import src.runtime_transforms  # noqa: F401
except Exception:
    pass

st.title("Timely Response — Dashboard de Producción (sin métricas de negocio)")
st.caption("Clasificación binaria Yes/No • sklearn==1.5.1 • pipelines con TF-IDF + OneHot + fecha")

# Rutas por defecto
default_model = repo_root/"models"/"trained_model.pkl"
default_cfg   = repo_root/"models"/"model_config.yaml"
default_data  = repo_root/"data"/"processed"/"clean_for_model.csv"

with st.sidebar:
    st.header("⚙️ Artefactos")
    st.write("Modelo por defecto:", f"`{default_model}`")
    st.write("Datos por defecto:", f"`{default_data}`")

    up_model = st.file_uploader("Sube un .pkl (opcional)", type=["pkl"], key="up_model")
    up_cfg   = st.file_uploader("Sube YAML con umbral (opcional)", type=["yaml","yml"], key="up_yaml")

    # Cargar modelo
    model, thr_default, model_name = None, 0.5, "desconocido"
    try:
        if up_model is not None:
            model = joblib.load(io.BytesIO(up_model.read()))
            if up_cfg is not None:
                cfg = yaml.safe_load(up_cfg.read().decode("utf-8"))
                thr_default = float(cfg.get("threshold", 0.5))
                model_name = str(cfg.get("model", model_name))
        else:
            model = joblib.load(default_model)
            if default_cfg.exists():
                cfg = yaml.safe_load(default_cfg.read_text(encoding="utf-8"))
                thr_default = float(cfg.get("threshold", 0.5))
                model_name = str(cfg.get("model", model_name))
    except FileNotFoundError:
        st.warning("No se encontró `models/trained_model.pkl`. Entrena primero o sube un .pkl.")
    except Exception as e:
        st.error(f"Error cargando modelo/YAML: {e}")

    st.header("🎚️ Umbral")
    thr = st.slider("Pred=Yes si score ≥ umbral", 0.0, 1.0, float(thr_default), 0.01)

    st.header("📤 Datos de entrada")
    up_csv = st.file_uploader("Sube un CSV para puntuar (opcional)", type=["csv"], key="up_csv")

# Decidir dataset de trabajo
if up_csv is not None:
    df = pd.read_csv(up_csv, low_memory=False)
elif default_data.exists():
    df = pd.read_csv(default_data, low_memory=False)
else:
    st.warning("No hay CSV por defecto y no has subido ninguno. Sube un archivo para continuar.")
    st.stop()

if model is None:
    st.warning("No hay modelo cargado. Sube un `.pkl` en la barra lateral o entrena primero.")
    st.stop()

missing = ensure_columns(df)
if missing:
    st.error(f"Faltan columnas requeridas para el pipeline: {missing}. Corrige el CSV.")
    st.stop()

# ---------- Scoring ----------
with st.spinner("Generando scores..."):
    scores = scores_from_model(model, df)
pred = (scores >= thr).astype(int)

# ---------- KPIs de scoring ----------
st.subheader("📈 KPIs de scoring")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Timely predicho (rate)", f"{(pred==1).mean()*100:.1f}%")
c3.metric("Score medio", f"{np.mean(scores):.3f}")
c4.metric("Umbral", f"{thr:.2f}")
st.caption(f"Modelo: **{model_name}** · Columnas: {', '.join(REQUIRED)}")

# ---------- Gráficos ----------
st.subheader("📊 Distribuciones")
fig1 = plt.figure(); pd.Series(scores).plot(kind="hist", bins=30); plt.axvline(thr, ls="--")
plt.title("Distribución de scores"); plt.xlabel("score"); plt.ylabel("frecuencia"); st.pyplot(fig1)

fig2 = plt.figure(); pd.Series(np.where(pred==1, "Yes","No")).value_counts().reindex(["Yes","No"]).plot(kind="bar")
plt.title("Conteo por predicción"); plt.ylabel("casos"); st.pyplot(fig2)

# ---------- Métricas si hay ground truth ----------
if "Timely response?" in df.columns:
    y_true = df["Timely response?"].astype("string").str.strip().str.lower().map({"yes":1,"y":1,"true":1,"no":0,"n":0,"false":0})
    mask = y_true.notna()
    if mask.any():
        y = y_true[mask].astype(int).to_numpy()
        y_pred = pred[mask]
        st.subheader("🧪 Métricas (con etiqueta real)")
        bal = balanced_accuracy_score(y, y_pred)
        f1n = f1_score((y==0).astype(int), (y_pred==0).astype(int))
        f1y = f1_score(y, y_pred)
        try:
            roc = roc_auc_score(y, scores[mask])
            pr  = average_precision_score(y, scores[mask])
        except Exception:
            roc, pr = float("nan"), float("nan")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Balanced Acc", f"{bal:.3f}")
        m2.metric("F1_No", f"{f1n:.3f}")
        m3.metric("F1_Yes", f"{f1y:.3f}")
        m4.metric("ROC-AUC | PR-AUC", f"{roc:.3f} | {pr:.3f}")

        cm = confusion_matrix(y, y_pred, labels=[0,1])
        fig3 = plt.figure(); ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(values_format="d"); plt.title("Matriz de confusión")
        st.pyplot(fig3)
    else:
        st.info("La columna 'Timely response?' existe pero no contiene valores válidos (Yes/No).")

# ---------- Tabla + descarga ----------
st.subheader("🧾 Resultados")
scored = df.copy()
scored["score"] = scores
scored["pred"]  = pred
st.dataframe(scored.head(50))
st.download_button("⬇️ Descargar scored.csv", scored.to_csv(index=False).encode("utf-8"),
                   file_name="scored.csv", mime="text/csv")
