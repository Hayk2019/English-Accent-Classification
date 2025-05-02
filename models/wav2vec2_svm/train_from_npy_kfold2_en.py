import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# === 1. Load data ===
FEATURE_DIR = "features"
csv_path = os.path.join(FEATURE_DIR, "features.csv")
masked = "british_english"

df = pd.read_csv("merge_new.csv")
df["path"] = df["filename"].apply(lambda x: os.path.join(FEATURE_DIR, x))

# === 2. Split into train/test (exclude masked class from train) ===
df_train = df[df["label"] != masked]
df_test = df[df["label"] == masked]

# === 3. Encode labels ===
le = LabelEncoder()
df_train["label_id"] = le.fit_transform(df_train["label"])

# === 4. Train with Stratified K-Fold ===
X = df_train["path"].tolist()
y = df_train["label_id"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n Fold {fold_idx}")
    X_train_paths = [X[i] for i in train_idx]
    y_train = y[train_idx]
    X_val_paths = [X[i] for i in val_idx]
    y_val = y[val_idx]
    
    X_train = np.array([np.load(p) for p in X_train_paths])
    X_val = np.array([np.load(p) for p in X_val_paths])

    clf = SVC(kernel='linear', C=0.01, gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    all_reports.append(report)

    print(classification_report(y_val, y_pred, target_names=le.classes_))

# === 5. Average F1-scores across folds ===
print("\nAverage F1-score across folds:")
for label in le.classes_:
    f1s = [r[label]['f1-score'] for r in all_reports]
    print(f"{label}: {np.mean(f1s):.4f}")

# === 6. Evaluate on unseen class (masked class) ===
print("\nEvaluation on unseen class ("+masked+": class):")
X_us = np.array([np.load(p) for p in df_test["path"]])
y_us_true = df_test["label"].values
y_us_pred = clf.predict(X_us)
y_us_pred_labels = le.inverse_transform(y_us_pred)

print("Number of test samples:", len(X_us))
print(pd.Series(y_us_pred_labels).value_counts())

# === 7. Create confusion matrix and save figure ===
# Add the masked label manually
all_labels = list(le.classes_) + ["unseen"]

# Prepare true and predicted labels for confusion matrix
y_us_true_fixed = np.array(["unseen"] * len(y_us_true))
all_true = y_us_true_fixed
all_pred = y_us_pred_labels

cm = confusion_matrix(all_true, all_pred, labels=all_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=all_labels, yticklabels=all_labels, 
            cbar=True, square=True)

plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

