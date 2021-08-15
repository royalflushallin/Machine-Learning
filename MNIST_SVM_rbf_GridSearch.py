

# # IMPORTING LIBRARIES
import numpy as np
import pandas as pd

# # IMPORTING DATASET
dataset_train = pd.read_csv('mnist_train.csv')
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values

dataset_test = pd.read_csv('mnist_test.csv')
X_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values


# # FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # TRAINING MULTI-CLASS CLASSIFIER
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
classifier = SVC(random_state=0, kernel='rbf', decision_function_shape='ovr')
classifier.fit(X_train, y_train)


# # APPLYING GRID SEARCH TO FIND OPTIMAL PARAMETER COMBINATIONS
from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.25, 0.50, 0.75, 1.00, 1.25], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.5, 0.6, 0.9]}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, n_jobs=-1, scoring='accuracy', cv=10)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
print(f"Best accuracy: {best_accuracy*100}")
print(f"Best parameters: {best_params}")


# # PREDICTING RESULTS ON TEST SET AND TRAINING SET
"""After picking the best parameters from best_params, model is again trained"""
y_pred_test = classifier.predict(X_test)
y_pred_train = classifier.predict(X_train)

# # CALCULATING ACCURACY
from sklearn.metrics import confusion_matrix, accuracy_score

cm_test = confusion_matrix(y_test, y_pred_test)
print(f"Confusion matrix of ground truth of test set and predictions in test set: \n {cm_test}")
print(f"Accuracy score of cm_test: {accuracy_score(y_test, y_pred_test)}")

cm_train = confusion_matrix(y_train, y_pred_train)
print(f"Confusion matrix of ground truth of training set and predictions on test set: \n {cm_train}")
print(f"Accuracy score of cm_train: {accuracy_score(y_train, y_pred_train)}")








