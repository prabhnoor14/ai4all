import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

df = pd.read_csv("data/sisfall_features.csv")

FEATURES = ["accel_mean", "accel_std", "accel_max", "accel_range", "gyro_mean", "gyro_std", "gyro_max"]

X = df[FEATURES].values
y = df["is_fall"].astype(int).values
groups = df["group"].values  # SA = young adult, SE = elderly


def sample_weights_for(y_fold, groups_fold):
    """Inverse-frequency weight by (group, is_fall) combo -- upweights the rare (SE, Fall) case."""
    combo = list(zip(groups_fold, y_fold))
    counts = Counter(combo)
    n_combos = len(counts)
    return [len(combo) / (n_combos * counts[c]) for c in combo]


# --- 5-fold stratified cross-validation: every one of the 75 elderly fall trials gets
# evaluated exactly once, instead of the single 80/20 split's noisy 19-example estimate. ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    se_mask = groups_test == "SE"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # baseline (unweighted)
    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train_s, y_train)
    y_pred_base = baseline.predict(X_test_s)

    # weighted (mitigation)
    weights = sample_weights_for(y_train, groups_train)
    weighted = LogisticRegression(max_iter=1000, random_state=42)
    weighted.fit(X_train_s, y_train, sample_weight=weights)
    y_pred_weighted = weighted.predict(X_test_s)

    fold_results.append({
        "fold": fold_i,
        "n_se_fall": int(((groups_test == "SE") & (y_test == 1)).sum()),
        "baseline_se_fall_recall": recall_score(y_test[se_mask], y_pred_base[se_mask], pos_label=1, zero_division=0),
        "weighted_se_fall_recall": recall_score(y_test[se_mask], y_pred_weighted[se_mask], pos_label=1, zero_division=0),
        "baseline_overall_fall_precision": precision_score(y_test, y_pred_base, pos_label=1, zero_division=0),
        "weighted_overall_fall_precision": precision_score(y_test, y_pred_weighted, pos_label=1, zero_division=0),
    })

results_df = pd.DataFrame(fold_results)
print("=== Per-fold results ===")
print(results_df)

print(f"\nTotal elderly fall trials evaluated across all folds: {results_df['n_se_fall'].sum()} (should be 75)")

print("\n=== 5-fold summary: mean ± std ===")
for col in ["baseline_se_fall_recall", "weighted_se_fall_recall",
            "baseline_overall_fall_precision", "weighted_overall_fall_precision"]:
    print(f"{col}: {results_df[col].mean():.3f} +/- {results_df[col].std():.3f}")

# Plot: per-fold SE-only Fall recall, baseline vs weighted
plot_df = results_df.melt(
    id_vars="fold",
    value_vars=["baseline_se_fall_recall", "weighted_se_fall_recall"],
    var_name="model", value_name="se_fall_recall",
)
plot_df["model"] = plot_df["model"].map({
    "baseline_se_fall_recall": "baseline", "weighted_se_fall_recall": "weighted (mitigation)"
})

plt.figure(figsize=(8, 5))
sns.barplot(x="fold", y="se_fall_recall", hue="model", data=plot_df, palette="Blues")
plt.title("SE (elderly) Fall recall per fold -- baseline vs. reweighted mitigation")
plt.ylabel("Fall recall (elderly subset)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/sisfall_cv_se_recall.png", dpi=120)
plt.close()
print("\nSaved outputs/sisfall_cv_se_recall.png")

# --- Final deployed model: refit on the FULL dataset with weighting, now that CV has
# validated the mitigation generalizes rather than being a single-split fluke. ---
final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X)
final_weights = sample_weights_for(y, groups)

final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_all_scaled, y, sample_weight=final_weights)

with open("models/sisfall_model.pkl", "wb") as f:
    pickle.dump({
        "model": final_model,
        "model_name": "logreg_weighted_full_data",
        "scaler": final_scaler,
        "features": FEATURES,
        "class_names": ["Non-Fall", "Fall"],
    }, f)
print("Saved models/sisfall_model.pkl (refit on full data with elderly-fall reweighting)")
