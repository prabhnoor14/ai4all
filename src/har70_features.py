import glob

import numpy as np
import pandas as pd

LABELS = {
    1: "walking", 2: "shuffling", 3: "stairs_up", 4: "stairs_down",
    5: "standing", 6: "sitting", 7: "lying", 8: "cycling_sit", 9: "cycling_stand",
}

WINDOW_SIZE = 250  # 250 samples @ 50Hz = 5 seconds


def load_subject(path):
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    return df


def extract_windows(df, window_size=WINDOW_SIZE):
    """Slice a subject's continuous stream into fixed windows and summarize each one."""
    back_mag = np.sqrt(df["back_x"] ** 2 + df["back_y"] ** 2 + df["back_z"] ** 2)
    thigh_mag = np.sqrt(df["thigh_x"] ** 2 + df["thigh_y"] ** 2 + df["thigh_z"] ** 2)

    n_windows = len(df) // window_size
    rows = []
    for i in range(n_windows):
        s, e = i * window_size, (i + 1) * window_size
        window_labels = df["label"].iloc[s:e]
        majority_label = window_labels.mode().iloc[0]
        purity = (window_labels == majority_label).mean()  # how "pure" this window is (1.0 = single activity)
        rows.append({
            "back_mean": back_mag.iloc[s:e].mean(),
            "back_std": back_mag.iloc[s:e].std(),
            "thigh_mean": thigh_mag.iloc[s:e].mean(),
            "thigh_std": thigh_mag.iloc[s:e].std(),
            "majority_label": majority_label,
            "majority_label_name": LABELS.get(int(majority_label), "unknown"),
            "purity": purity,
        })
    return pd.DataFrame(rows)


def build_har70_window_dataset(root="data/har70plus"):
    all_windows = []
    for path in sorted(glob.glob(f"{root}/*.csv")):
        subject_id = path.split("\\")[-1].split("/")[-1].replace(".csv", "")
        df = load_subject(path)
        windows = extract_windows(df)
        windows["subject"] = subject_id
        all_windows.append(windows)
    return pd.concat(all_windows, ignore_index=True)


if __name__ == "__main__":
    windows = build_har70_window_dataset()
    windows.to_csv("data/har70_windows.csv", index=False)
    print(f"Built {len(windows)} windows -> data/har70_windows.csv")
    print(windows["majority_label_name"].value_counts())
