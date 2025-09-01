
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)

def _scores_from_estimator(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:,1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
    else:
        s = estimator.predict(X).astype(float)
    smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-9)

def sweep_threshold_for_F1_no(y_true, scores, n_grid: int = 101) -> pd.DataFrame:
    qs = np.linspace(0.01, 0.99, n_grid)
    thrs = np.quantile(scores, qs)
    rows = []
    for thr in thrs:
        pred = (scores >= thr).astype(int)
        f1_no  = f1_score((y_true==0).astype(int), (pred==0).astype(int))
        f1_yes = f1_score(y_true, pred, pos_label=1)
        balacc = balanced_accuracy_score(y_true, pred)
        rows.append((float(thr), f1_no, f1_yes, balacc))
    return pd.DataFrame(rows, columns=["threshold","F1_No","F1_Yes","BalancedAcc"]).sort_values("F1_No", ascending=False)

def summarize_at_threshold(y_true, scores, thr: float = 0.5):
    pred = (scores >= thr).astype(int)
    return {"threshold": float(thr),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "f1_yes": float(f1_score(y_true, pred, pos_label=1)),
            "f1_no":  float(f1_score((y_true==0).astype(int), (pred==0).astype(int))),
            "balanced_acc": float(balanced_accuracy_score(y_true, pred))},                confusion_matrix(y_true, pred, labels=[0,1])

# Business metrics
def yn_to_bool(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype="string").str.strip().str.lower()
    return s.map({"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False})

def date_diff_days(s_start: pd.Series, s_end: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s_start, errors="coerce", infer_datetime_format=True)
    d1 = pd.to_datetime(s_end, errors="coerce", infer_datetime_format=True)
    return (d1 - d0).dt.days.astype("float64")

def business_metrics(df: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, pred: np.ndarray,
                     dispute_col: str = "Consumer disputed?", date_recv: str = "Date received",
                     date_sent: str = "Date sent to company",
                     base_cost: float = 1.0, Pu: float = 8.0, Pd: float = 15.0, Pf: float = 0.2):
    p_yes = scores; p_no = 1.0 - scores
    disputed = yn_to_bool(df.get(dispute_col))
    days_to_forward = date_diff_days(df.get(date_recv), df.get(date_sent)).fillna(0)
    expected_cost = base_cost + p_no*Pu + (disputed==True).astype(float)*Pd + days_to_forward*Pf
    q90 = float(np.nanquantile(expected_cost, 0.90))
    high_cost_flag = (expected_cost >= q90).astype(int)

    out = df.copy()
    out["p_yes"] = p_yes; out["p_no"] = p_no
    out["pred"] = pred
    out["expected_cost"] = expected_cost
    out["high_cost_flag"] = high_cost_flag

    summary = {"timely_rate_pred": float(np.mean(pred==1)),
               "dispute_rate": float(np.nanmean((disputed==True).astype(float))),
               "avg_days_to_forward": float(np.nanmean(days_to_forward)),
               "avg_expected_cost": float(np.nanmean(expected_cost)),
               "p90_expected_cost": q90}
    return out, summary

def evaluate_best(gs_dict: Dict[str, object], X_test, y_test, df_test: pd.DataFrame):
    """
    Toma {name: GridSearchCV_fitted}, calcula métricas base y óptimas (F1_No) y produce plots.
    Devuelve (tabla_resumen, detalles_por_modelo).
    """
    rows = []
    details = {}
    plt.close("all")  # ahora sí: 'plt' viene del import de nivel superior

    for name, gs in gs_dict.items():
        est = gs.best_estimator_
        scores = _scores_from_estimator(est, X_test)

        # Baseline @0.5
        summ_b, cm_b = summarize_at_threshold(y_test, scores, thr=0.5)
        # Óptimo por F1_No
        df_thr = sweep_threshold_for_F1_no(y_test, scores, n_grid=101)
        thr_opt = float(df_thr.iloc[0]["threshold"]) if len(df_thr) else 0.5
        summ_o, cm_o = summarize_at_threshold(y_test, scores, thr=thr_opt)

        # Business metrics @ thr_opt
        pred_opt = (scores >= thr_opt).astype(int)
        enriched, biz = business_metrics(df_test, y_test, scores, pred_opt)

        rows.append({
            "modelo": name,
            "GS_best_cv_score": float(gs.best_score_),
            "ROC_AUC": summ_b["roc_auc"],
            "PR_AUC": summ_b["pr_auc"],
            "F1_No@0.5": summ_b["f1_no"],
            "BalAcc@0.5": summ_b["balanced_acc"],
            "F1_No@opt": summ_o["f1_no"],
            "BalAcc@opt": summ_o["balanced_acc"],
            "thr_opt": thr_opt,
            "avg_expected_cost@opt": biz["avg_expected_cost"],
        })
        details[name] = {
            "scores": scores, "thr_opt": thr_opt,
            "cm_base": cm_b, "cm_opt": cm_o,
            "biz_summary": biz,
            "enriched_df": enriched,  # p_yes, p_no, expected_cost, high_cost_flag
        }

        # --- Plots ---
        df_curve = sweep_threshold_for_F1_no(y_test, scores, n_grid=101).sort_values("threshold")

        plt.figure(figsize=(7, 4))
        plt.plot(df_curve["threshold"], df_curve["F1_No"], label="F1_No")
        plt.plot(df_curve["threshold"], df_curve["F1_Yes"], label="F1_Yes")
        plt.plot(df_curve["threshold"], df_curve["BalancedAcc"], label="BalancedAcc")
        plt.axvline(thr_opt, linestyle="--")
        plt.title(f"Métricas vs umbral — {name}")
        plt.xlabel("threshold"); plt.ylabel("score"); plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, scores)
        plt.title(f"ROC — {name}")
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, scores)
        plt.title(f"Precision-Recall — {name}")
        plt.tight_layout(); plt.show()

    comp = pd.DataFrame(rows).sort_values(["F1_No@opt", "BalAcc@opt"], ascending=[False, False]).reset_index(drop=True)
    return comp, details
