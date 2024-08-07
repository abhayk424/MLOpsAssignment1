import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from sklearn.ensemble import RandomForestClassifier
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def preprocess_data(train_df):
    """Preprocesses the training data."""

     # Dropping Ticket and Cabin and Name columns
    train_df = train_df.drop(["PassengerId", "Ticket", "Cabin", "Name"], axis=1)

    # Sex Feature
    train_df["Sex"] = train_df["Sex"].map({"female": 1, "male": 0}).astype(int)

    # Age Feature
    guess_ages = np.zeros((2, 3))
    freq_port = train_df.Embarked.dropna().mode()[0]
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = train_df[(train_df["Sex"] == i) & (train_df["Pclass"] == j + 1)][
                "Age"
            ].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            train_df.loc[
                (train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j + 1),
                "Age",
            ] = guess_ages[i, j]
    train_df["Age"] = train_df["Age"].astype(int)
    train_df.loc[train_df["Age"] <= 16, "Age"] = 0
    train_df.loc[(train_df["Age"] > 16) & (train_df["Age"] <= 32), "Age"] = 1
    train_df.loc[(train_df["Age"] > 32) & (train_df["Age"] <= 48), "Age"] = 2
    train_df.loc[(train_df["Age"] > 48) & (train_df["Age"] <= 64), "Age"] = 3
    train_df.loc[train_df["Age"] > 64, "Age"] = 4

    # Family Features
    train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
    train_df["IsAlone"] = 0
    train_df.loc[train_df["FamilySize"] == 1, "IsAlone"] = 1
    train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1, inplace=True)

    # Interaction Feature
    train_df["Age*Class"] = train_df.Age * train_df.Pclass

    # Embarked Feature
    train_df["Embarked"] = train_df["Embarked"].fillna(freq_port)
    train_df["Embarked"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    # Fare Feature
    train_df.loc[train_df["Fare"] <= 7.91, "Fare"] = 0
    train_df.loc[(train_df["Fare"] > 7.91) & (train_df["Fare"] <= 14.454), "Fare"] = 1
    train_df.loc[(train_df["Fare"] > 14.454) & (train_df["Fare"] <= 31), "Fare"] = 2
    train_df.loc[train_df["Fare"] > 31, "Fare"] = 3
    train_df["Fare"] = train_df["Fare"].astype(int)

     # Separate features and target variable
    X = train_df.drop("Survived", axis=1) # Assuming 'Survived' is your target column
    y = train_df["Survived"]
    return X, y

# Load the Titanic dataset
data = pd.read_csv('data/titanic.csv')

# Preprocess the data
X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy  # Optuna minimizes, so we maximize accuracy

# Create a study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# Train the model with best parameters
best_params = study.best_params
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)

# Save the best model 
import joblib
joblib.dump(model, "best_model.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best parameters:", study.best_params)
print("Accuracy:", accuracy)
