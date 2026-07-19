import sys
sys.path.insert(0, ".")
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from src.data_prep import load_cstick_clean

df = load_cstick_clean()

FEATURES = ["Distance", "HRV", "Sugar level", "SpO2"]
CLASS_NAMES = ["Normal", "Near-Miss", "Fall"]

X = df[FEATURES]
y = df["Decision"]

# Stratified split so Near-Miss/Fall are proportionally represented in the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# --- Logistic Regression (scaled) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_logreg, target_names=CLASS_NAMES))
results["logreg"] = {
    "model": logreg,
    "y_pred": y_pred_logreg,
    "recall_macro": recall_score(y_test, y_pred_logreg, average="macro"),
}

# --- Decision Tree ---
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\n=== Decision Tree ===")
print(classification_report(y_test, y_pred_tree, target_names=CLASS_NAMES))
results["tree"] = {
    "model": tree,
    "y_pred": y_pred_tree,
    "recall_macro": recall_score(y_test, y_pred_tree, average="macro"),
}

# --- Compare on macro recall (we care about catching Near-Miss/Fall, not just overall accuracy) ---
print("\n=== Model comparison (macro recall) ===")
for name, r in results.items():
    print(f"{name}: {r['recall_macro']:.4f}")

winner_name = max(results, key=lambda k: results[k]["recall_macro"])
winner = results[winner_name]["model"]
print(f"\nWinner: {winner_name}")

# Confusion matrices for both models
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("outputs/cstick_confusion_matrices.png", dpi=120)
plt.close()
print("\nSaved outputs/cstick_confusion_matrices.png")

# Export winning model (+ scaler, only meaningful if logreg wins)
with open("models/cstick_model.pkl", "wb") as f:
    pickle.dump({
        "model": winner,
        "model_name": winner_name,
        "scaler": scaler if winner_name == "logreg" else None,
        "features": FEATURES,
        "class_names": CLASS_NAMES,
    }, f)
print("Saved models/cstick_model.pkl")
