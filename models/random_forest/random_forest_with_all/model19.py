import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "reduced_india_accent.csv"
df = pd.read_csv(file_path)

X = df.iloc[:, 2:]
y = df["accent"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],                   
    'max_depth': list(range(5, 26, 5)) + [None],                
    'min_samples_split': [2, 5, 10],                            
    'class_weight': [None, 'balanced']                         
}



grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")



y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
report = classification_report(y_test, y_pred, digits=4)


cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)




report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

class_metrics_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

plt.figure(figsize=(10, 6))
class_metrics_df['precision'].plot(kind='barh', color='skyblue')
plt.title("Precision per Class")
plt.xlabel("Precision")
plt.tight_layout()
plt.savefig("precision_per_class.png")
plt.close()

plt.figure(figsize=(10, 6))
class_metrics_df['recall'].plot(kind='barh', color='orange')
plt.title("Recall per Class")
plt.xlabel("Recall")
plt.tight_layout()
plt.savefig("recall_per_class.png")
plt.close()

plt.figure(figsize=(10, 6))
class_metrics_df['f1-score'].plot(kind='barh', color='green')
plt.title("F1-score per Class")
plt.xlabel("F1-score")
plt.tight_layout()
plt.savefig("f1_score_per_class.png")
plt.close()




print(f"\nTest Accuracy: {accuracy:.2f}%")
print("Classification Report:\n", report)

with open("metrics_report.txt", "w") as f:
    f.write(f"Best Parameters:\n{grid_search.best_params_}\n")
    f.write(f"Best CV Accuracy: {grid_search.best_score_:.4f}\n")
    f.write(f"\nTest Accuracy: {accuracy:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

joblib.dump(best_model, "best_random_forest_model.pkl")

