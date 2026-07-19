import glob
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    1: "walking", 2: "shuffling", 3: "stairs_up", 4: "stairs_down",
    5: "standing", 6: "sitting", 7: "lying", 8: "cycling_sit", 9: "cycling_stand",
}

files = sorted(glob.glob("data/har70plus/*.csv"))
print(f"Found {len(files)} subject files")

total_rows = 0
label_counts = pd.Series(dtype=int)
missing_total = 0
subject_summaries = []

for f in files:
    df = pd.read_csv(f)
    total_rows += len(df)
    missing_total += df.isna().sum().sum()
    vc = df["label"].value_counts()
    label_counts = label_counts.add(vc, fill_value=0)
    subject_summaries.append({"file": f.split("/")[-1], "rows": len(df), "n_labels": df["label"].nunique()})

print(f"\nTotal rows across all subjects: {total_rows}")
print(f"Total missing values: {missing_total}")

print("\n=== Label distribution (pooled across all subjects) ===")
label_counts = label_counts.sort_index()
for lbl, cnt in label_counts.items():
    name = LABELS.get(int(lbl), "unknown")
    print(f"{int(lbl)} ({name}): {int(cnt)}  ({cnt/total_rows*100:.2f}%)")

print("\n=== Per-subject row counts ===")
print(pd.DataFrame(subject_summaries))

# sampling rate estimate from first file
df0 = pd.read_csv(files[0])
df0["timestamp"] = pd.to_datetime(df0["timestamp"])
diffs = df0["timestamp"].diff().dropna().dt.total_seconds()
print(f"\n=== Sampling interval (subject {files[0]}) ===")
print(f"Median interval: {diffs.median()*1000:.2f} ms  => ~{1/diffs.median():.1f} Hz")

# Plot a short segment showing a sitting -> standing transition if one exists
df0["label"] = df0["label"].astype(int)
transition_idx = df0.index[(df0["label"].shift(1) == 6) & (df0["label"] == 5)]
if len(transition_idx) > 0:
    center = transition_idx[0]
    window = df0.iloc[max(0, center - 100):center + 100]
    plt.figure(figsize=(10, 5))
    plt.plot(window["thigh_x"].values, label="thigh_x")
    plt.plot(window["thigh_y"].values, label="thigh_y")
    plt.plot(window["thigh_z"].values, label="thigh_z")
    plt.axvline(100, color="red", linestyle="--", label="sitting->standing boundary")
    plt.title(f"Thigh accelerometer around a sit-to-stand transition ({files[0]})")
    plt.xlabel("sample index")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/har70_transition.png", dpi=120)
    plt.close()
    print("\nSaved outputs/har70_transition.png")
else:
    print("\nNo direct sitting->standing transition found in first subject file")
