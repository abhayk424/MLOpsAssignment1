"""Training the models and logging the ezxperiments in MLflow."""

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("data/train.csv")

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Titanic Survival Prediction")

with mlflow.start_run(run_name="Logistic Regression"):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="Perceptron"):
    model = Perceptron()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="SGD"):
    model = SGDClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="SVC"):
    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="Linear SVC"):
    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="Decision Tree"):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

with mlflow.start_run(run_name="Random Forest"):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("Test Accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("Test Precision", precision_score(y_test, y_pred))
    mlflow.log_metric("Test Recall", recall_score(y_test, y_pred))
    mlflow.log_metric("Test F1", f1_score(y_test, y_pred))

    mlflow.sklearn.log_model(model, "rf_base")
