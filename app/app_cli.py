# -*- coding: utf-8 -*-
# App ejecutiva simple (sin uploads):
# - Usa SOLO artefactos internos del repo:
#     models/trained_model.pkl
#     models/model_config.yaml
#     data/processed/clean_for_model.csv
# - KPIs de scoring, métricas de modelo (si hay ground truth),
#   y métricas de negocio (expected_cost, P90, top compañías, top casos).
# - Guarda models/scored.csv y permite descargarlo.

import sys
from pathlib import Path as P
import numpy as np
import pandas as pd
import joblib, yaml
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------------- Utilidades ----------------
REQUIRED = ["Issue","Sub-issue","Product","Sub-product","Company","State","Date received"]

def add_src_to_path(repo_root: P):
    """Asegura que src/ sea importable (necesario si el pipeline usa FunctionTransformer)."""
    for p in (repo_root, repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        import src.runtime_transforms  # noqa: F401
    except Exception:
        pass

def scores_from_model(model, X: pd.DataFrame) -> np.ndarray:
    """Devuelve scores en [0,1] para cualquier estimador común de sklearn."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        smin, smax = float(np.min(s)), float(np.max(s))
        return (s - smin) / (smax - smin + 1e-9)
    return model.predict(X).astype(float)

def yn_to_bool(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype="string").str.strip().str.lower()
    return s.map({"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False})

def date_diff_days(s_start: pd.Series, s_end: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s_start, errors="coerce", infer_datetime_format=True)
    d1 = pd.to_datetime(s_end, errors="coerce", infer_datetime_format=True)
    return (d1 - d0).dt.days.astype("float64")

def business_enrichment(df: pd.DataFrame, scores: np.ndarray,
                        base_cost: float, Pu: float, Pd: float, Pf: float, thr: float):
    """
    Calcula métricas de negocio por fila:
      - score, pred, days_to_forward, expected_cost, high_cost_flag (P90)
    Devuelve (enriched_df, p90_cost).
    """
    p_yes = scores
    p_no  = 1.0 - scores

    # disputed y days_to_forward (robustos aunque no existan columnas opcionales)
    disputed = yn_to_bool(df.get("Consumer disputed?")) if "Consumer disputed?" in df.columns \
               else pd.Series(False, index=df.index)
    if "Date sent to company" in df.columns:
        d0 = pd.to_datetime(df.get("Date received"), errors="coerce", infer_datetime_format=True)
        d1 = pd.to_datetime(df.get("Date sent to company"), errors="coerce", infer_datetime_format=True)
        days_to_forward = (d1 - d0).dt.days.astype("float64")
    else:
        days_to_forward = pd.Series(0.0, index=df.index, dtype="float64")

    expected_cost = base_cost + p_no*Pu + (disputed == True).astype(float)*Pd + days_to_forward.fillna(0)*Pf
    q90 = float(np.nanquantile(expected_cost, 0.90)) if len(expected_cost) else 0.0
    high_cost_flag = (expected_cost >= q90).astype(int) if len(expected_cost) else pd.Series(0, index=df.index)

    enriched = df.copy()
    enriched["score"] = p_yes
    enriched["pred"]  = (p_yes >= thr).astype(int)
    enriched["days_to_forward"] = days_to_forward
    enriched["expected_cost"] = expected_cost
    enriched["high_cost_flag"] = high_cost_flag
    return enriched, q90

def aggregate_company(enriched: pd.DataFrame) -> pd.DataFrame:
    """KPIs por compañía (usa columnas ya alineadas en enriched)."""
    g = enriched.groupby("Company", dropna=False)
    kpi = pd.DataFrame({
        "n_cases": g.size(),
        "timely_rate_pred": g["pred"].mean(),
        "avg_score": g["score"].mean(),
        "avg_days_to_forward": g["days_to_forward"].mean(),
        "avg_expected_cost": g["expected_cost"].mean(),
        "total_expected_cost": g["expected_cost"].sum(),
    })
    return kpi.sort_values("total_expected_cost", ascending=False)

# ---------------- App ----------------
st.set_page_config(page_title="Timely Response — Ejecutivo", layout="wide")
repo_root = P(__file__).resolve().parents[1]
add_src_to_path(repo_root)

MODEL_PATH = repo_root / "models" / "trained_model_cli.pkl"
CFG_PATH   = repo_root / "models" / "model_config_cli.yaml"
DATA_PATH  = repo_root / "data" / "processed" / "clean_for_model.csv"
SCORED_OUT = repo_root / "models" / "scored_cli.csv"

st.title("Timely Response — Ejecutivo")
st.caption("Modelo binario Yes/No • TF-IDF + OneHot + Fecha (sklearn==1.5.1) • Sin cargas externas")

# Validación artefactos
if not MODEL_PATH.exists():
    st.error(f"❌ Falta el modelo: {MODEL_PATH}")
    st.stop()
if not DATA_PATH.exists():
    st.error(f"❌ Falta el CSV procesado: {DATA_PATH}")
    st.stop()

# Cargar artefactos
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

thr_default, model_name = 0.5, "desconocido"
if CFG_PATH.exists():
    try:
        cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
        thr_default = float(cfg.get("threshold", 0.5))
        model_name = str(cfg.get("model", model_name))
    except Exception as e:
        st.warning(f"YAML {CFG_PATH} ilegible: {e}")

df = pd.read_csv(DATA_PATH, low_memory=False)
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"❌ El CSV carece de columnas necesarias: {missing}")
    st.stop()

# Sidebar: umbral y costes (ajustables para demo interna)
with st.sidebar:
    st.header("⚙️ Parámetros")
    thr = st.slider("Umbral (pred=Yes si score ≥ umbral)", 0.0, 1.0, float(thr_default), 0.01)
    st.write(f"Modelo: **{model_name}**")
    st.divider()
    st.caption("Métricas de negocio:")
    base_cost = st.number_input("C0 (base por caso)", value=1.0, min_value=0.0, step=0.5)
    Pu        = st.number_input("Pu (penalización por No)", value=8.0, min_value=0.0, step=0.5)
    Pd        = st.number_input("Pd (penalización por disputa)", value=15.0, min_value=0.0, step=0.5)
    Pf        = st.number_input("Pf (coste por día)", value=0.2, min_value=0.0, step=0.1)

# Scoring básico
scores = scores_from_model(model, df)
pred = (scores >= thr).astype(int)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Visión general", "Métricas de negocio", "Métricas de modelo", "Descargas"])

with tab1:
    st.subheader("📈 KPIs de scoring")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Timely predicho", f"{(pred==1).mean()*100:.1f}%")
    c3.metric("Score medio", f"{np.mean(scores):.3f}")
    c4.metric("Umbral", f"{thr:.2f}")
    st.caption(f"Datos: `{DATA_PATH}`")

    st.subheader("📊 Distribuciones")
    fig1 = plt.figure()
    pd.Series(scores).plot(kind="hist", bins=30)
    plt.axvline(thr, linestyle="--")
    plt.title("Distribución de scores"); plt.xlabel("score"); plt.ylabel("frecuencia")
    st.pyplot(fig1)

    fig2 = plt.figure()
    pd.Series(np.where(pred==1, "Yes","No")).value_counts().reindex(["Yes","No"]).plot(kind="bar")
    plt.title("Conteo por predicción"); plt.ylabel("casos")
    st.pyplot(fig2)

with tab2:
    st.subheader("💰 Métricas de negocio")
    enriched, p90 = business_enrichment(df, scores, base_cost, Pu, Pd, Pf, thr)
    # KPIs negocio
    c1, c2, c3 = st.columns(3)
    c1.metric("Coste esperado medio", f"{np.nanmean(enriched['expected_cost']):.2f}")
    c2.metric("P90 coste esperado", f"{p90:.2f}")
    c3.metric("High-cost (≥P90)", f"{int(enriched['high_cost_flag'].sum())}")

    # Top compañías por total_expected_cost
    comp = aggregate_company(enriched)
    st.markdown("**Top compañías por coste total esperado**")
    st.dataframe(comp.head(15).round(3))

    # Top casos por expected_cost
    st.markdown("**Top casos por coste esperado**")
    cols_show = ["Company","score","pred","expected_cost","days_to_forward","Issue","Product","State","Date received"]
    cols_show = [c for c in cols_show if c in enriched.columns]
    st.dataframe(enriched.sort_values("expected_cost", ascending=False)[cols_show].head(25).round(3))

with tab3:
    st.subheader("🧪 Métricas de modelo (si existe ground truth)")
    if "Timely response?" in df.columns:
        y_true = df["Timely response?"].astype("string").str.strip().str.lower().map(
            {"yes":1,"y":1,"true":1,"no":0,"n":0,"false":0}
        )
        mask = y_true.notna()
        if mask.any():
            y = y_true[mask].astype(int).to_numpy()
            y_pred = pred[mask]
            try:
                roc = roc_auc_score(y, scores[mask]); pr = average_precision_score(y, scores[mask])
            except Exception:
                roc, pr = float("nan"), float("nan")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balanced Acc", f"{balanced_accuracy_score(y, y_pred):.3f}")
            c2.metric("F1_No", f"{f1_score((y==0).astype(int), (y_pred==0).astype(int)):.3f}")
            c3.metric("F1_Yes", f"{f1_score(y, y_pred):.3f}")
            c4.metric("ROC-AUC | PR-AUC", f"{roc:.3f} | {pr:.3f}")

            cm = confusion_matrix(y, y_pred, labels=[0,1])
            fig3 = plt.figure()
            ConfusionMatrixDisplay(cm, display_labels=["No","Yes"]).plot(values_format="d")
            plt.title("Matriz de confusión")
            st.pyplot(fig3)
        else:
            st.info("La columna 'Timely response?' no contiene valores válidos (Yes/No).")
    else:
        st.info("No hay columna 'Timely response?' en los datos.")

with tab4:
    st.subheader("⬇️ Descargas")
    scored = df.copy()
    scored["score"] = scores
    scored["pred"]  = pred
    try:
        SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(SCORED_OUT, index=False)
        st.caption(f"`models/scored.csv` actualizado.")
    except Exception as e:
        st.warning(f"No se pudo guardar models/scored.csv: {e}")
    st.download_button(
        "Descargar scored.csv",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="scored.csv",
        mime="text/csv"
    )
