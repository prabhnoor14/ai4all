import sys
sys.path.insert(0, ".")

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import load_cstick_clean

df = load_cstick_clean()

print("=== columns kept ===")
print(list(df.columns))

FEATURES = ["Distance", "HRV", "Sugar level", "SpO2"]

print("\n=== Step 1: descriptive stats by Decision class ===")
stats_summary = df.groupby("Decision")[FEATURES].agg(["mean", "std"])
print(stats_summary)

print("\n=== Step 3: ANOVA (feature ~ Decision) ===")
groups = sorted(df["Decision"].unique())
for col in FEATURES:
    samples = [df[df["Decision"] == g][col] for g in groups]
    f_stat, p_val = stats.f_oneway(*samples)
    sig = "significant" if p_val < 0.05 else "NOT significant"
    print(f"{col}: F={f_stat:.3f}, p={p_val:.4e}  -> {sig}")

# Step 2: visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, FEATURES):
    sns.boxplot(x="Decision", y=col, hue="Decision", data=df, ax=ax, palette="Blues", legend=False)
    ax.set_title(f"{col} by Decision class")
plt.tight_layout()
plt.savefig("outputs/cstick_clean_boxplots.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 5))
corr = df[FEATURES + ["Decision"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("cStick correlation matrix (leaky columns removed)")
plt.tight_layout()
plt.savefig("outputs/cstick_clean_corr.png", dpi=120)
plt.close()

print("\nSaved outputs/cstick_clean_boxplots.png and outputs/cstick_clean_corr.png")

# Save cleaned dataset for Phase 2 model training
df.to_csv("data/cstick_clean.csv", index=False)
print("Saved cleaned dataset to data/cstick_clean.csv")
