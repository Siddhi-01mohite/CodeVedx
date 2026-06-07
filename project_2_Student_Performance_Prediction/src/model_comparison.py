import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

# Load Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "student_performance.csv")

df = pd.read_csv(file_path)

# Features
X = df[["Attendance", "Study_Hours", "Internal_Marks"]]

# Target
y = df["Final_Marks"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    scores[name] = score

    print(f"{name}: {score:.4f}")

# Plot Results
plt.figure(figsize=(8,5))

plt.bar(scores.keys(), scores.values())

plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")

image_path = os.path.join(
    current_dir,
    "..",
    "images",
    "model_comparison.png"
)

plt.savefig(image_path)
plt.show()

print("Model comparison graph saved!")