import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "data/SisFall_dataset"

subject_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
print(f"Subjects: {len(subject_dirs)}")
sa = [s for s in subject_dirs if s.startswith("SA")]
se = [s for s in subject_dirs if s.startswith("SE")]
print(f"  SA (young adults): {len(sa)}")
print(f"  SE (elderly): {len(se)}")

rows = []
for subj in subject_dirs:
    for f in os.listdir(os.path.join(ROOT, subj)):
        m = re.match(r"([A-Z])(\d+)_([A-Z0-9]+)_R(\d+)\.txt", f)
        if not m:
            print("UNMATCHED FILENAME:", f)
            continue
        activity_type, activity_num, subj_id, trial = m.groups()
        rows.append({
            "subject": subj_id,
            "group": subj_id[:2],
            "activity_code": f"{activity_type}{activity_num}",
            "is_fall": activity_type == "F",
            "trial": trial,
            "path": os.path.join(ROOT, subj, f),
        })

meta = pd.DataFrame(rows)
print(f"\nTotal trial files: {len(meta)}")
print("\n=== Files per group (SA vs SE) x fall/ADL ===")
print(meta.groupby(["group", "is_fall"]).size().unstack(fill_value=0))

print("\n=== Does SE (elderly) group contain any Fall trials? ===")
print(meta[meta["group"] == "SE"]["is_fall"].value_counts())

print("\n=== Activity code counts (top 20) ===")
print(meta["activity_code"].value_counts().head(20))

# Row count / duration check on a sample of files
sample = meta.sample(n=min(30, len(meta)), random_state=42)
durations = []
for _, r in sample.iterrows():
    with open(r["path"]) as fh:
        n = sum(1 for _ in fh)
    durations.append(n)
sample = sample.assign(n_rows=durations)
print("\n=== Sample trial row counts (200Hz => rows/200 = seconds) ===")
print(sample[["activity_code", "is_fall", "n_rows"]].sort_values("is_fall"))

def load_trial(path):
    df = pd.read_csv(path, header=None, sep=r"\s*,\s*", engine="python")
    df[df.columns[-1]] = df[df.columns[-1]].astype(str).str.rstrip(";").astype(float)
    df.columns = ["adxl_x", "adxl_y", "adxl_z", "gyro_x", "gyro_y", "gyro_z", "mma_x", "mma_y", "mma_z"]
    return df

fall_example = meta[meta["is_fall"]].iloc[0]
adl_example = meta[~meta["is_fall"]].iloc[0]

fall_df = load_trial(fall_example["path"])
adl_df = load_trial(adl_example["path"])

fall_mag = np.sqrt(fall_df["adxl_x"]**2 + fall_df["adxl_y"]**2 + fall_df["adxl_z"]**2)
adl_mag = np.sqrt(adl_df["adxl_x"]**2 + adl_df["adxl_y"]**2 + adl_df["adxl_z"]**2)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)
axes[0].plot(fall_mag.values)
axes[0].set_title(f"Fall trial signal magnitude ({fall_example['activity_code']}, {fall_example['subject']})")
axes[1].plot(adl_mag.values, color="green")
axes[1].set_title(f"ADL trial signal magnitude ({adl_example['activity_code']}, {adl_example['subject']})")
for ax in axes:
    ax.set_xlabel("sample index (200 Hz)")
    ax.set_ylabel("accel magnitude (raw ADC)")
plt.tight_layout()
plt.savefig("outputs/sisfall_fall_vs_adl.png", dpi=120)
plt.close()
print("\nSaved outputs/sisfall_fall_vs_adl.png")
