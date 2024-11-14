import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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

X, y = digits.data, digits.target

y_binary = (y == 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(digits.data, y_binary, test_size=0.25, random_state=0)

model = LogisticRegression(solver= 'lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred, target_names=["Not 1", "1"]))

