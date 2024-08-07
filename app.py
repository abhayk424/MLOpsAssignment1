from flask import Flask, request, jsonify
import pandas as pd
import joblib

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load the pre-trained model
model = joblib.load("best_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
  """
  Predicts passenger survival based on received data.

  Returns:
      JSON: Dictionary containing the predicted survival status.
  """
  try:
    # Get passenger data from request
    data = request.get_json()
    if not data:
      return jsonify({"error": "Missing passenger data in request body."}), 400
    
    # Convert data to pandas DataFrame
    df = pd.DataFrame(data["features"])

    df["Pclass"] = df["Pclass"].astype(int)
    df["Age"] = df["Age"].astype(int)
    df["SibSp"] = df["SibSp"].astype(int)
    df["Parch"] = df["Parch"].astype(int)
    df["Fare"] = df["Fare"].astype(float)

    # Preprocess data (adapt to your specific feature names)
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)

    # Handle Age values
    df.loc[df["Age"] <= 16, "Age"] = 0
    df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age"] = 1
    df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age"] = 2
    df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age"] = 3
    df.loc[df["Age"] > 64, "Age"] = 4

    # Handle remaining categorical features (adapt to your features)
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)

    # Handle Fare feature (adapt to your logic)
    df.loc[df["Fare"] <= 7.91, "Fare"] = 0
    df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare"] = 1
    df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31), "Fare"] = 2
    df.loc[df["Fare"] > 31, "Fare"] = 3
    df["Fare"] = df["Fare"].astype(int)

    # Handle Family features (adapt to your logic)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
    df.drop(["Parch", "SibSp", "FamilySize"], axis=1, inplace=True)

    # Handle interaction feature (adapt to your logic)
    df["Age*Class"] = df.Age * df.Pclass

    # Make prediction using the loaded model
    prediction = model.predict(df)[0]

    # Return prediction as JSON response
    return jsonify({"survival": int(prediction)})

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  app.run(debug=True)
