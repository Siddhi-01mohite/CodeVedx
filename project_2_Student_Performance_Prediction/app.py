from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "student_model.pkl")

model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        study_hours = float(request.form["study_hours"])
        internal_marks = float(request.form["internal_marks"])

        data = pd.DataFrame({
            "Attendance": [attendance],
            "Study_Hours": [study_hours],
            "Internal_Marks": [internal_marks]
        })

        prediction = round(model.predict(data)[0], 2)

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)