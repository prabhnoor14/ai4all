import pickle
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

df = pd.read_csv("data/sisfall_features.csv")

FEATURES = ["accel_mean", "accel_std", "accel_max", "accel_range", "gyro_mean", "gyro_std", "gyro_max"]
CLASS_NAMES = ["Non-Fall", "Fall"]

X = df[FEATURES]
y = df["is_fall"].astype(int)
groups = df["group"]  # SA = young adult, SE = elderly -- kept aside for subset evaluation, not used as a feature

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)

results = {}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
results["logreg"] = {"model": logreg, "y_pred": y_pred_logreg}

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
results["tree"] = {"model": tree, "y_pred": y_pred_tree}

for name, r in results.items():
    print(f"=== {name}: overall (SA + SE combined) ===")
    print(classification_report(y_test, r["y_pred"], target_names=CLASS_NAMES))
    r["recall_macro"] = recall_score(y_test, r["y_pred"], average="macro")

# --- Evaluate separately on the elderly (SE) subset ---
se_mask = (groups_test == "SE").values
print(f"\nSE (elderly) subset size in test set: {se_mask.sum()} of {len(y_test)}")
for name, r in results.items():
    print(f"\n=== {name}: SE (elderly) subset ONLY ===")
    y_test_se = y_test[se_mask]
    y_pred_se = r["y_pred"][se_mask]
    print(classification_report(y_test_se, y_pred_se, target_names=CLASS_NAMES, zero_division=0))
    r["recall_macro_se"] = recall_score(y_test_se, y_pred_se, average="macro", zero_division=0)

print("\n=== Model comparison (baseline, unweighted) ===")
for name, r in results.items():
    print(f"{name}: overall macro recall={r['recall_macro']:.4f}  |  SE-only macro recall={r['recall_macro_se']:.4f}")

# --- Mitigation experiment: upweight the rare (elderly, fall) combination ---
# The elderly Fall recall gap exists because SE+Fall trials are a tiny slice of training data
# (most Fall examples are young-adult/SA). Inverse-frequency sample weighting forces the model
# to weigh each SE+Fall training example much more heavily than the abundant SA examples.
combo_train = list(zip(groups_train, y_train))
combo_counts = Counter(combo_train)
n_combos = len(combo_counts)
sample_weight = [len(combo_train) / (n_combos * combo_counts[c]) for c in combo_train]

print("\n=== Sample weights by (group, is_fall) combo ===")
for combo, count in sorted(combo_counts.items()):
    w = len(combo_train) / (n_combos * count)
    print(f"{combo}: n={count}, weight={w:.2f}")

weighted_results = {}

logreg_w = LogisticRegression(max_iter=1000, random_state=42)
logreg_w.fit(X_train_scaled, y_train, sample_weight=sample_weight)
weighted_results["logreg_weighted"] = {"model": logreg_w, "y_pred": logreg_w.predict(X_test_scaled)}

tree_w = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_w.fit(X_train, y_train, sample_weight=sample_weight)
weighted_results["tree_weighted"] = {"model": tree_w, "y_pred": tree_w.predict(X_test)}

for name, r in weighted_results.items():
    print(f"\n=== {name}: overall (SA + SE combined) ===")
    print(classification_report(y_test, r["y_pred"], target_names=CLASS_NAMES))
    r["recall_macro"] = recall_score(y_test, r["y_pred"], average="macro")

    y_pred_se = r["y_pred"][se_mask]
    print(f"=== {name}: SE (elderly) subset ONLY ===")
    print(classification_report(y_test[se_mask], y_pred_se, target_names=CLASS_NAMES, zero_division=0))
    r["recall_macro_se"] = recall_score(y_test[se_mask], y_pred_se, average="macro", zero_division=0)

    fall_recall_se = recall_score(y_test[se_mask], y_pred_se, pos_label=1, zero_division=0)
    r["fall_recall_se"] = fall_recall_se

print("\n=== Before vs after reweighting: SE-only Fall recall (the number that matters) ===")
for base_name in ["logreg", "tree"]:
    base_fall_recall_se = recall_score(
        y_test[se_mask], results[base_name]["y_pred"][se_mask], pos_label=1, zero_division=0
    )
    weighted_fall_recall_se = weighted_results[f"{base_name}_weighted"]["fall_recall_se"]
    print(f"{base_name}: {base_fall_recall_se:.2f} -> {weighted_fall_recall_se:.2f} (weighted)")

results.update(weighted_results)
winner_name = max(results, key=lambda k: results[k]["recall_macro_se"])
winner = results[winner_name]["model"]
print(f"\nWinner (by SE-only macro recall, since that's the disparity we're targeting): {winner_name}")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (name, r) in zip(axes.flat, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix - {name} (overall)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("outputs/sisfall_confusion_matrices.png", dpi=120)
plt.close()
print("\nSaved outputs/sisfall_confusion_matrices.png")

with open("models/sisfall_model.pkl", "wb") as f:
    pickle.dump({
        "model": winner,
        "model_name": winner_name,
        "scaler": scaler if "logreg" in winner_name else None,
        "features": FEATURES,
        "class_names": CLASS_NAMES,
    }, f)
print("Saved models/sisfall_model.pkl")
