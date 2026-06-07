import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Training Started...")

# Load Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "student_performance.csv")

df = pd.read_csv(file_path)

# Features (Input)
X = df[["Attendance", "Study_Hours", "Internal_Marks"]]

# Target (Output)
y = df["Final_Marks"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE =", round(mae, 2))
print("R2 Score =", round(r2, 2))

# Save Model
models_folder = os.path.join(current_dir, "..", "models")
os.makedirs(models_folder, exist_ok=True)

model_path = os.path.join(models_folder, "student_model.pkl")

joblib.dump(model, model_path)

print("\nModel Saved Successfully!")