import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.har70_features import build_har70_transitions_dataset

# --- Sit-to-stand transition speed ---
transitions = build_har70_transitions_dataset()
transitions.to_csv("data/har70_transitions.csv", index=False)
print(f"Found {len(transitions)} sit-to-stand transitions across {transitions['subject'].nunique()} subjects")

print("\n=== Transition vigor by subject (mean post-transition movement) ===")
by_subject = transitions.groupby("subject")[["post_std", "post_range", "transition_vigor"]].agg(["mean", "count"])
print(by_subject)

plt.figure(figsize=(10, 5))
sns.boxplot(x="subject", y="post_std", data=transitions, hue="subject", palette="Blues", legend=False)
plt.title("Sit-to-stand transition vigor by subject (thigh accel std, 2.5s post-transition)")
plt.ylabel("post-transition movement (std)")
plt.tight_layout()
plt.savefig("outputs/har70_transition_vigor_by_subject.png", dpi=120)
plt.close()
print("Saved outputs/har70_transition_vigor_by_subject.png")

# --- Time-in-zone mobility profile, from the k-means clusters we already built ---
windows = pd.read_csv("data/har70_windows_clustered.csv")
CLUSTER_NAMES = {0: "sedentary", 1: "moderate", 2: "vigorous"}
windows["zone"] = windows["cluster"].map(CLUSTER_NAMES)

time_in_zone = pd.crosstab(windows["subject"], windows["zone"], normalize="index") * 100
print("\n=== Time-in-zone mobility profile per subject (%) ===")
print(time_in_zone.round(1))

time_in_zone.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="Blues_r")
plt.title("Mobility profile per subject (% time in each movement-intensity zone)")
plt.ylabel("% of windows")
plt.xlabel("subject")
plt.legend(title="zone", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/har70_time_in_zone.png", dpi=120)
plt.close()
print("Saved outputs/har70_time_in_zone.png")

time_in_zone.to_csv("data/har70_time_in_zone.csv")
