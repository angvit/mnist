import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# from sklearn.datasets import fetch_mldata

# mnist = fetch_mldata('MNIST original')
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# plt.show()

X = digits.data
y = (digits.target == 1).astype(int)

skf = StratifiedKFold(n_splits=5)
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# X_train, X_test, y_train, y_test = train_test_split(digits.data, y, test_size=0.25, random_state=0)

auc_scores = []
accuracy_scores = []

for train_index, test_index in skf.split(X, y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict probabilities and classes
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_probs)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Append metrics
    auc_scores.append(auc)
    accuracy_scores.append(accuracy)


mean_auc = sum(auc_scores) / len(auc_scores)
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print(f"Mean AUC: {mean_auc:.4f} (+/- {max(auc_scores) - min(auc_scores):.4f})")
print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {max(accuracy_scores) - min(accuracy_scores):.4f})")

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# print(classification_report(y_test, y_pred, target_names=["Not 1", "1"]))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

print(f"AUC Score: {auc_score:.4f}")

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
# plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="best")
# plt.show()

best_threshold = tpr - fpr
best_threshold_index = np.argmax(best_threshold)
best_threshold = thresholds[best_threshold_index]
print(f"Best Threshold (Lambda) value: {best_threshold:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="g", cmap="Greens", cbar=False, xticklabels=["Not 1", "1"], yticklabels=["Not 1", "1"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of Digit Images")
plt.show()