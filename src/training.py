
import sys, json, yaml
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score

# ensure transforms importable
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
for p in (str(repo_root), str(src_dir)):
    if p not in sys.path: sys.path.insert(0, p)
import src.runtime_transforms as rtf

# --------- Preprocessor ---------
def build_preprocessor(min_df: int = 10, max_features: int = 20000):
    text_pipe = Pipeline([
        ("select_text", FunctionTransformer(rtf.select_text, validate=False)),
        ("concat",      FunctionTransformer(rtf.concat_text, validate=False)),
        ("tfidf",       TfidfVectorizer(min_df=min_df, ngram_range=(1,2), max_features=max_features)),
    ])
    cat_pipe = Pipeline([
        ("select_cat", FunctionTransformer(rtf.select_cat, validate=False)),
        ("imputer",    SimpleImputer(strategy="most_frequent")),
        ("ohe",        OneHotEncoder(handle_unknown="ignore")),
    ])
    date_pipe = Pipeline([
        ("select_date", FunctionTransformer(rtf.select_date, validate=False)),
        ("date_feats",  FunctionTransformer(rtf.date_features, validate=False)),
        ("imputer",     SimpleImputer(strategy="most_frequent")),
        ("scale",       MaxAbsScaler()),
    ])
    feats = FeatureUnion([("text", text_pipe), ("cat", cat_pipe), ("date", date_pipe)])
    return feats

# --------- Models + Grids ---------
def build_models_and_grids(k_best: int = 20000):
    pre = build_preprocessor()
    selector = ("selectkbest", SelectKBest(score_func=chi2, k=k_best))  # chi2 soporta esparso y no-negativo

    reg_log = Pipeline([("features", pre), selector,
                        ("reglog", LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"))])
    reg_log_param = {"selectkbest__k": [5000, 10000, 20000],
                     "reglog__penalty": ["l1","l2"],
                     "reglog__C": np.logspace(-1, 1, 5)}

    linsvm = Pipeline([("features", pre), selector,
                       ("linsvm", LinearSVC(class_weight="balanced", max_iter=4000))])
    linsvm_param = {"selectkbest__k": [5000, 10000, 20000],
                    "linsvm__C": [0.25, 0.5, 1.0]}

    svm = Pipeline([("features", pre), selector,
                    ("svm", SVC(probability=True, class_weight="balanced", kernel="linear"))])
    svm_param = {"selectkbest__k": [5000, 10000],
                 "svm__C": [0.5, 1.0]}

    rand_forest = Pipeline([("features", pre), selector,
                            ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))])
    rf_param = {"selectkbest__k": [5000, 10000, 20000],
                "rf__n_estimators": [200, 400],
                "rf__max_depth": [None, 10, 20],
                "rf__max_features": ["sqrt", 0.2, 0.5]}

    models = {"reg_log": (reg_log, reg_log_param),
              "linsvm": (linsvm, linsvm_param),
              "svm": (svm, svm_param),
              "rand_forest": (rand_forest, rf_param)}

    try:
        from lightgbm import LGBMClassifier  # noqa
        lgbm = Pipeline([("features", pre), selector,
                         ("lgbm", __import__("lightgbm").LGBMClassifier(objective="binary",
                                                                        class_weight="balanced",
                                                                        n_estimators=400, learning_rate=0.1))])
        models["lightgbm"] = (lgbm, {"selectkbest__k": [5000, 10000, 20000],
                                     "lgbm__num_leaves": [31, 63],
                                     "lgbm__max_depth": [-1, 10]})
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier  # noqa
        xgb = Pipeline([("features", pre), selector,
                        ("xgb", __import__("xgboost").XGBClassifier(objective="binary:logistic", tree_method="hist",
                                                                     n_estimators=400, learning_rate=0.1,
                                                                     max_depth=6, subsample=0.9,
                                                                     colsample_bytree=0.9, reg_lambda=1.0))])
        models["xgboost"] = (xgb, {"selectkbest__k": [5000, 10000, 20000],
                                   "xgb__max_depth": [4, 6],
                                   "xgb__reg_lambda": [0.5, 1.0]})
    except Exception:
        pass

    return models

# --------- Fit GridSearch ---------
def fit_grids(X_train: pd.DataFrame, y_train: np.ndarray, models_dict: dict,
              scoring: str = "balanced_accuracy", cv_splits: int = 5, n_jobs: int = -1, verbose: int = 1):
    out = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for name, (pipe, grid) in models_dict.items():
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=True)
        gs.fit(X_train, y_train)
        out[name] = gs
    return out

# --------- Threshold sweep ---------
def _scores_from_estimator(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:,1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
    else:
        s = estimator.predict(X).astype(float)
    smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-9)

def sweep_threshold_f1_no(y_true, scores, n_grid=101):
    qs = np.linspace(0.01, 0.99, n_grid)
    thrs = np.quantile(scores, qs)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thrs:
        pred = (scores >= thr).astype(int)
        f1_no = f1_score((y_true==0).astype(int), (pred==0).astype(int))
        if f1_no > best_f1:
            best_f1, best_thr = f1_no, float(thr)
    return float(best_thr), float(best_f1)

# --------- Train & Save best ---------
def train_and_save_best(df: pd.DataFrame, y: np.ndarray, models_dict: dict,
                        test_size: float = 0.2, random_state: int = 42,
                        out_model: Path = Path("models/trained_model.pkl"),
                        out_config: Path = Path("models/model_config.yaml")):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr, te = next(sss.split(df, y))
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]

    grids = fit_grids(Xtr, ytr, models_dict, scoring="balanced_accuracy", cv_splits=5, n_jobs=-1, verbose=1)

    # elegir por F1_No óptimo en el split de validación
    best_name, best_score, best_thr = None, -1.0, 0.5
    best_est = None
    for name, gs in grids.items():
        est = gs.best_estimator_
        scores = _scores_from_estimator(est, Xte)
        thr, f1n = sweep_threshold_f1_no(yte, scores, n_grid=101)
        if f1n > best_score:
            best_name, best_score, best_thr, best_est = name, f1n, thr, est

    out_model.parent.mkdir(parents=True, exist_ok=True)
    from joblib import dump
    dump(best_est, out_model)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config, "w", encoding="utf-8") as f:
        yaml.dump({"model": best_name, "threshold": float(best_thr)}, f, sort_keys=False, allow_unicode=True)

    return {"best_model": best_name, "f1_no_val": best_score, "threshold": best_thr,
            "out_model": str(out_model), "out_config": str(out_config)}
