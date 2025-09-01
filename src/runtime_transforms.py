
import pandas as pd

def select_text(X: pd.DataFrame):
    return X[["Issue","Sub-issue"]]

def select_cat(X: pd.DataFrame):
    return X[["Product","Sub-product","Company","State"]]

def select_date(X: pd.DataFrame):
    return X[["Date received"]]

def concat_text(X: pd.DataFrame):
    D = X.astype("string").fillna("")
    return D.agg(" ".join, axis=1).to_numpy()

def date_features(X: pd.DataFrame):
    s = X.iloc[:,0] if isinstance(X, pd.DataFrame) else X
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    out = pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month, "day": dt.dt.day, "dow": dt.dt.dayofweek})
    return out.to_numpy(dtype="float64")
