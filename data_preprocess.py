"""Preprocessing the training data."""

import numpy as np
import pandas as pd

train_df = pd.read_csv("data/train.csv")

# Dropping Ticket and Cabin columns
train_df = train_df.drop(["Ticket", "Cabin"], axis=1)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

guess_ages = np.zeros((2, 3))

freq_port = train_df.Embarked.dropna().mode()[0]


train_df["Title"] = train_df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

train_df["Title"] = train_df["Title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)

train_df["Title"] = train_df["Title"].replace("Mlle", "Miss")
train_df["Title"] = train_df["Title"].replace("Ms", "Miss")
train_df["Title"] = train_df["Title"].replace("Mme", "Mrs")

train_df["Title"] = train_df["Title"].map(title_mapping)
train_df["Title"] = train_df["Title"].fillna(0)

train_df["Sex"] = train_df["Sex"].map({"female": 1, "male": 0}).astype(int)

for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train_df[(train_df["Sex"] == i) & (train_df["Pclass"] == j + 1)][
            "Age"
        ].dropna()

        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
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

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

train_df["IsAlone"] = 0
train_df.loc[train_df["FamilySize"] == 1, "IsAlone"] = 1

train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)

train_df["Age*Class"] = train_df.Age * train_df.Pclass

train_df["Embarked"] = train_df["Embarked"].fillna(freq_port)
train_df["Embarked"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

train_df.loc[train_df["Fare"] <= 7.91, "Fare"] = 0
train_df.loc[(train_df["Fare"] > 7.91) & (train_df["Fare"] <= 14.454), "Fare"] = 1
train_df.loc[(train_df["Fare"] > 14.454) & (train_df["Fare"] <= 31), "Fare"] = 2
train_df.loc[train_df["Fare"] > 31, "Fare"] = 3
train_df["Fare"] = train_df["Fare"].astype(int)

train_df = train_df.drop(["Name", "PassengerId"], axis=1)
train_df.to_csv("data/train.csv", index=False)
