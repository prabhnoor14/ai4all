import os
import re

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(r"([A-Z])(\d+)_([A-Z0-9]+)_R(\d+)\.txt")
COLUMNS = ["adxl_x", "adxl_y", "adxl_z", "gyro_x", "gyro_y", "gyro_z", "mma_x", "mma_y", "mma_z"]


def list_sisfall_trials(root="data/SisFall_dataset"):
    """Walk the SisFall folder and return trial metadata (one row per trial file)."""
    rows = []
    for subject_dir in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subject_dir)
        if not os.path.isdir(subj_path):
            continue
        for fname in os.listdir(subj_path):
            m = FILENAME_RE.match(fname)
            if not m:
                continue  # skips desktop.ini, images/, Readme.txt etc.
            activity_type, activity_num, subj_id, trial = m.groups()
            rows.append({
                "subject": subj_id,
                "group": subj_id[:2],  # SA = young adult, SE = elderly
                "activity_code": f"{activity_type}{activity_num}",
                "is_fall": activity_type == "F",
                "trial": trial,
                "path": os.path.join(subj_path, fname),
            })
    return pd.DataFrame(rows)


def load_trial_signal(path):
    """Load one SisFall trial file (9 raw ADC columns, semicolon-terminated rows)."""
    df = pd.read_csv(path, header=None, sep=",", skipinitialspace=True)
    df.columns = COLUMNS
    df["mma_z"] = df["mma_z"].astype(str).str.rstrip(";").astype(float)
    return df


def extract_trial_features(signal_df):
    """Collapse one trial's raw signal into a single row of summary features.

    We use trial-level aggregation (not sub-windowing) because each SisFall file
    is already one scripted event (one ADL performance or one simulated fall) --
    slicing it into smaller windows and labeling every window "Fall" would mislabel
    the walk-up/recovery portions of a fall trial as falls.
    """
    adxl_mag = np.sqrt(signal_df["adxl_x"] ** 2 + signal_df["adxl_y"] ** 2 + signal_df["adxl_z"] ** 2)
    gyro_mag = np.sqrt(signal_df["gyro_x"] ** 2 + signal_df["gyro_y"] ** 2 + signal_df["gyro_z"] ** 2)

    features = {
        "accel_mean": adxl_mag.mean(),
        "accel_std": adxl_mag.std(),
        "accel_min": adxl_mag.min(),
        "accel_max": adxl_mag.max(),
        "accel_range": adxl_mag.max() - adxl_mag.min(),
        "gyro_mean": gyro_mag.mean(),
        "gyro_std": gyro_mag.std(),
        "gyro_max": gyro_mag.max(),
        "n_samples": len(signal_df),
    }
    return features


def build_sisfall_dataset(root="data/SisFall_dataset"):
    meta = list_sisfall_trials(root)
    feature_rows = []
    for _, r in meta.iterrows():
        signal = load_trial_signal(r["path"])
        feats = extract_trial_features(signal)
        feats.update({
            "subject": r["subject"],
            "group": r["group"],
            "activity_code": r["activity_code"],
            "is_fall": r["is_fall"],
        })
        feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    df = build_sisfall_dataset()
    df.to_csv("data/sisfall_features.csv", index=False)
    print(f"Built {len(df)} trial-level rows -> data/sisfall_features.csv")
    print(df["is_fall"].value_counts())
