
# IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk


# IMPORTING DATASET
dataset_train = pd.read_csv('mnist_train.csv')
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values
"""y_train = y_train.reshape(len(y_train), 1)"""

dataset_test = pd.read_csv('mnist_test.csv')
X_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values
"""y_test = y_test.reshape(len(y_test), 1)"""

# FEATURE SCALING
"""fac = 0.99 / 255
X_train = np.asfarray(X_train) * fac + 0.01
X_test = np.asfarray(X_test) * fac + 0.01"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""# ENCODING LABELS FOR k-CLASSES (IF REQUIRED)
y_train_kClasses = np.zeros((10, len(y_train)))
for i in range(10):
    y_train_kClasses[i, :] = np.where(y_train == i, 1, 0)"""


# TRAINING MULTI-CLASS CLASSIFIER
"""from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, multi_class='ovr', max_iter=1000)
classifier.fit(X_train, y_train)"""


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
classifier.fit(X_train, y_train)


# PREDICTING RESULTS ON TEST SET
y_pred = classifier.predict(X_test)
y_test_and_pred = np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1)
y_pred_train = classifier.predict(X_train)
print(y_test_and_pred)


# CALCULATING ACCURACY
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)
print(accuracy_score(y_train, y_pred_train))








