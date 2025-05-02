import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

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

