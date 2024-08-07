# Import necessary libraries
from sklearn.datasets import fetch_california_housing  # Import dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LinearRegression  # Linear Regression model
import joblib  # For saving the trained model

# Load the California housing dataset
# This dataset contains information about housing in California from the 1990 Census
california = fetch_california_housing()
X, y = california.data, california.target  # Features and target variable

# Split the dataset into training and testing sets
# We use 80% of the data for training and 20% for testing
# random_state=42 ensures reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Save the trained model to a file for future use
# This allows us to load and use the model without retraining
joblib.dump(model, 'model.joblib')
