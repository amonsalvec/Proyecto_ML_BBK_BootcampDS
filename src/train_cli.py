
import argparse, sys, yaml, json
from pathlib import Path
import numpy as np, pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path: sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path: sys.path.insert(0, str(repo_root / "src"))
from src.preprocessing import yes_no_to_int
from src.training import build_models_and_grids, fit_grids, _scores_from_estimator, sweep_threshold_f1_no

def main():
    ap = argparse.ArgumentParser(description="Train and save best model with threshold (clean build)")
    ap.add_argument("--data", required=True, help="CSV with training data")
    ap.add_argument("--target", default="Timely response?")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out-model", default="models/trained_model.pkl")
    ap.add_argument("--out-config", default="models/model_config.yaml")
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    y = yes_no_to_int(df[args.target]).astype("Int64")
    mask = y.notna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).to_numpy()

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    tr, te = next(sss.split(df, y))
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]

    models = build_models_and_grids(k_best=10000)
    grids = fit_grids(Xtr, ytr, models, scoring="balanced_accuracy", cv_splits=5, n_jobs=-1, verbose=1)

    best_name, best_f1, best_thr, best_est = None, -1.0, 0.5, None
    for name, gs in grids.items():
        est = gs.best_estimator_
        scores = _scores_from_estimator(est, Xte)
        thr, f1n = sweep_threshold_f1_no(yte, scores, n_grid=101)
        if f1n > best_f1:
            best_name, best_f1, best_thr, best_est = name, f1n, thr, est

    from joblib import dump
    out_model = Path(args.out_model); out_model.parent.mkdir(parents=True, exist_ok=True)
    dump(best_est, out_model)
    out_cfg = Path(args.out_config); out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.dump({"model": best_name, "threshold": float(best_thr)}, f, sort_keys=False, allow_unicode=True)

    print(json.dumps({"best_model": best_name, "f1_no_val": best_f1,
                      "threshold": float(best_thr),
                      "out_model": str(out_model), "out_config": str(out_cfg)}, indent=2))

if __name__ == "__main__":
    main()
