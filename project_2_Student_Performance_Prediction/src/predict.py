import os
import joblib
import pandas as pd

# Load saved model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "student_model.pkl")

model = joblib.load(model_path)

print("Student Performance Prediction System")

attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours: "))
internal_marks = float(input("Enter Internal Marks: "))

data = pd.DataFrame({
    "Attendance": [attendance],
    "Study_Hours": [study_hours],
    "Internal_Marks": [internal_marks]
})

prediction = model.predict(data)

print("\nPredicted Final Marks:", round(prediction[0], 2))