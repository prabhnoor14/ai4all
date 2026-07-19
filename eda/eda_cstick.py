import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cStick.csv", encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]

print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== missing values ===")
print(df.isna().sum())

print("\n=== Decision value counts ===")
print(df["Decision"].value_counts().sort_index())
print(df["Decision"].value_counts(normalize=True).sort_index())

print("\n=== describe ===")
print(df.describe())

print("\n=== groupby Decision: mean/std ===")
print(df.groupby("Decision")[["HRV", "Sugar level", "SpO2", "Distance", "Accelerometer", "Pressure"]].agg(["mean", "std"]))

print("\n=== duplicate rows ===")
print(df.duplicated().sum())

print("\n=== Accelerometer / Pressure unique values (check if categorical) ===")
print("Accelerometer:", sorted(df["Accelerometer"].unique()))
print("Pressure:", sorted(df["Pressure"].unique()))

# ANOVA per feature across Decision classes
print("\n=== ANOVA (feature ~ Decision) ===")
groups = df["Decision"].unique()
for col in ["HRV", "Sugar level", "SpO2", "Distance"]:
    samples = [df[df["Decision"] == g][col] for g in groups]
    f_stat, p_val = stats.f_oneway(*samples)
    print(f"{col}: F={f_stat:.3f}, p={p_val:.4e}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, ["HRV", "Sugar level", "SpO2", "Distance"]):
    sns.boxplot(x="Decision", y=col, data=df, ax=ax, palette="Blues")
    ax.set_title(f"{col} by Decision class")
plt.tight_layout()
plt.savefig("outputs/cstick_boxplots.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
corr = df[["HRV", "Sugar level", "SpO2", "Distance", "Accelerometer", "Pressure", "Decision"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("cStick correlation matrix")
plt.tight_layout()
plt.savefig("outputs/cstick_corr.png", dpi=120)
plt.close()

print("\nSaved outputs/cstick_boxplots.png and outputs/cstick_corr.png")
