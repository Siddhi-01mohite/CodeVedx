from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        news_text = request.form["news"]

        # Convert text into vector
        vector = vectorizer.transform([news_text])

        # Predict
        pred = model.predict(vector)[0]

        # Convert prediction to user-friendly output
        if pred == "REAL":
            prediction = "Real News"
        else:
            prediction = "Fake News"

        # Confidence score
        confidence = round(
            max(model.predict_proba(vector)[0]) * 100,
            2
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)