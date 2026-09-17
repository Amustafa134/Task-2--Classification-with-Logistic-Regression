#####   modules     #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

#####   loading the raw dataset     #####

df = pd.read_excel('Dry_Bean_Dataset.xlsx')

print(df.head())
print(df.info())
print(df.describe())
print("\nClass balance:\n", df["Class"].value_counts())
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

#####   data preprocessing     #####

#####    Cleaning the data     #####

print("\nRows before removing duplicates:", df.shape[0])

df = df.drop_duplicates()
print("Rows after removing duplicates:", df.shape[0])

le = LabelEncoder()

df["target"] = le.fit_transform(df["Class"])
class_names = le.classes_

print("\nEncoded classes:", dict(zip(class_names, range(len(class_names)))))


#####   defining features and target     #####

X = df.drop(columns=["Class", "target"])
y = df["target"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nFeature columns:", list(X.columns))

#####   splitting the dataset into training and testing sets     #####

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("\nTrain class balance:\n", y_train.value_counts(normalize=True).sort_index())
print("\nTest class balance:\n", y_test.value_counts(normalize=True).sort_index())

#####   feature scaling     #####

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#####   training the logistic regression model     #####

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
print("Predictions on first 10 test samples:", y_pred[:10])
print("Actual labels for first 10 test samples:", y_test.values[:10])

#####   evaluating the model     #####

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("\nLogistic Regression - Full Test Set")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

#####   confusion matrix    #####

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

plt.colorbar()
plt.tight_layout()
plt.show()


#####   ROC curve (one-vs-rest)     #####

y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
y_score = logreg.predict_proba(X_test_scaled)

plt.figure(figsize=(8, 7))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Logistic Regression (One-vs-Rest)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()  

#####  comparing with other classifiers     #####

models = {
    "Logistic Regression": logreg,
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", random_state=42),
}

results = []
for name, model in models.items():
    if name != "Logistic Regression":
        model.fit(X_train_scaled, y_train)
    y_pred_model = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred_model),
        "Precision": precision_score(y_test, y_pred_model, average="macro"),
        "Recall": recall_score(y_test, y_pred_model, average="macro"),
        "F1-score": f1_score(y_test, y_pred_model, average="macro"),
    })


results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))


#####       visualizing the comparison      #####

x_pos = np.arange(len(results_df))
width = 0.2
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

plt.figure()
for i, metric in enumerate(metrics):
    plt.bar(x_pos + i * width, results_df[metric], width, label=metric)
plt.xticks(x_pos + width * 1.5, results_df["Model"])
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Classifier Comparison")
plt.legend()
plt.tight_layout()
plt.show()
