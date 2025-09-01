
import argparse, sys, yaml, json
from pathlib import Path
import numpy as np, pandas as pd, joblib

repo_root = Path(__file__).resolve().parents[1]
for p in (str(repo_root), str(repo_root / "src")):
    if p not in sys.path: sys.path.insert(0, p)
import src.runtime_transforms  # needed for unpickling FunctionTransformer

def main():
    ap = argparse.ArgumentParser(description="Score CSV with saved model + threshold (clean build)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    model = joblib.load(args.model)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thr = float(cfg.get("threshold", 0.5))

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(df)[:,1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(df)
        smin, smax = float(np.min(s)), float(np.max(s)); scores = (s - smin)/(smax - smin + 1e-9)
    else:
        scores = model.predict(df).astype(float)

    pred = (scores >= thr).astype(int)
    out = df.copy(); out["score"] = scores; out["pred"] = pred
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(json.dumps({"wrote": args.out, "n": int(len(out)), "threshold_used": thr}, indent=2))

if __name__ == "__main__":
    main()
