
import pandas as pd, numpy as np

def yes_no_to_int(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    return s.map({"yes":1,"y":1,"true":1,"no":0,"n":0,"false":0}).astype("Int64")

def yn_to_bool(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    return s.map({"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False})

def date_diff_days(s_start: pd.Series, s_end: pd.Series) -> pd.Series:
    d0 = pd.to_datetime(s_start, errors="coerce", infer_datetime_format=True)
    d1 = pd.to_datetime(s_end, errors="coerce", infer_datetime_format=True)
    return (d1 - d0).dt.days.astype("float64")
