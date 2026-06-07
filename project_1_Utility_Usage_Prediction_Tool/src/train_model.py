import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv("data/utility_usage.csv")

# Features (inputs)
X = df[["Temperature", "Residents"]]

# Target (output)
y = df["Usage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, y_pred)

print(f"Model Accuracy (R² Score): {score:.2f}")

# Save model
joblib.dump(model, "models/utility_model.pkl")

print("Model saved successfully!")