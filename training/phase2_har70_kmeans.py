import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/har70_windows.csv")

FEATURES = ["back_mean", "back_std", "thigh_mean", "thigh_std"]
X = df[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Quick elbow check to justify k
print("=== inertia by k (elbow check) ===")
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    print(f"k={k}: inertia={km.inertia_:.1f}")

# Settle on k=3 for interpretability (mirrors Normal/Near-Miss/Fall framing used elsewhere in the project)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\n=== cluster sizes ===")
print(df["cluster"].value_counts().sort_index())

print("\n=== cluster centers (movement intensity, in original units) ===")
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
print(centers)

# Validation (NOT used for clustering, only to sanity-check what each cluster represents):
# does each unsupervised cluster line up with a recognizable, distinct activity mix?
print("\n=== cluster vs true activity label (validation only, label not used in clustering) ===")
crosstab = pd.crosstab(df["cluster"], df["majority_label_name"], normalize="index")
print((crosstab * 100).round(1))

plt.figure(figsize=(9, 6))
sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42),
                 x="thigh_mean", y="thigh_std", hue="cluster", palette="deep", alpha=0.5, s=15)
plt.title("K-Means clusters on HAR70+ movement windows (k=3)")
plt.tight_layout()
plt.savefig("outputs/har70_kmeans_clusters.png", dpi=120)
plt.close()
print("\nSaved outputs/har70_kmeans_clusters.png")

df.to_csv("data/har70_windows_clustered.csv", index=False)
