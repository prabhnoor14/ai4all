import pandas as pd

LEAKY_COLUMNS = ["Pressure", "Accelerometer"]


def load_cstick_clean(path="data/cStick.csv"):
    """Load cStick.csv and drop columns that leak the Decision label."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=LEAKY_COLUMNS)
    return df
