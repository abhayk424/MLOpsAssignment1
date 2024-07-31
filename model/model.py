import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the script's directory as the current working directory
os.chdir(script_dir)

# Load the dataset
train_data = pd.read_csv('../data/titanic/train.csv')
test_data = pd.read_csv('../data/titanic/test.csv')

# Display the first few rows of the training dataset
train_data.head()

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# Handling missing values
# - Fill missing values for 'Age' and 'Fare' with their mean
# - Fill missing values for 'Embarked' with the most frequent value
imputer = SimpleImputer(strategy='mean')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numeric
label_encoders = {}
for col in ['Sex', 'Embarked']:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Display the first few rows of the preprocessed data
X.head()


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "saved_models/rfc_model.joblib")

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

# Generate a confusion matrix and classification report
conf_matrix = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred)

# Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)

