import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("data/sisfall_features.csv")

print("=== shape ===")
print(df.shape)

print("\n=== is_fall counts by group (SA=young, SE=elderly) ===")
print(df.groupby(["group", "is_fall"]).size().unstack(fill_value=0))

FEATURES = ["accel_mean", "accel_std", "accel_max", "accel_range", "gyro_mean", "gyro_std", "gyro_max"]

print("\n=== describe by is_fall ===")
print(df.groupby("is_fall")[FEATURES].agg(["mean", "std"]))

print("\n=== ANOVA (feature ~ is_fall) ===")
for col in FEATURES:
    f_stat, p_val = stats.f_oneway(df[df["is_fall"]][col], df[~df["is_fall"]][col])
    print(f"{col}: F={f_stat:.3f}, p={p_val:.4e}")

print("\n=== correlation with is_fall ===")
corr_df = df.copy()
corr_df["is_fall"] = corr_df["is_fall"].astype(int)
print(corr_df[FEATURES + ["is_fall"]].corr()["is_fall"].sort_values())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, ["accel_max", "accel_range", "gyro_max", "accel_std"]):
    sns.boxplot(x="is_fall", y=col, hue="is_fall", data=df, ax=ax, palette="Blues", legend=False)
    ax.set_title(f"{col} by is_fall")
plt.tight_layout()
plt.savefig("outputs/sisfall_features_boxplots.png", dpi=120)
plt.close()
print("\nSaved outputs/sisfall_features_boxplots.png")
