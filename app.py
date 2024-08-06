from flask import Flask, request, jsonify
import pandas as pd
import joblib

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
    df = pd.DataFrame(data)

    # Preprocess data (adapt to your specific feature names)
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)

    # Handle missing Age values using your logic
    guess_ages = np.zeros((2, 3))
    freq_port = df.Embarked.dropna().mode()[0]
    for i in range(0, 2):
      for j in range(0, 3):
        guess_df = df[(df["Sex"] == i) & (df["Pclass"] == j + 1)][
          "Age"
        ].dropna()
        age_guess = guess_df.median()
        guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
      for j in range(0, 3):
        df.loc[
          (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), "Age"
        ] = guess_ages[i, j]
    df["Age"] = df["Age"].astype(int)

    # Handle remaining categorical features (adapt to your features)
    df["Embarked"] = df["Embarked"].fillna(freq_port)
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
    return jsonify({"survival": prediction})

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  app.run(debug=True)
