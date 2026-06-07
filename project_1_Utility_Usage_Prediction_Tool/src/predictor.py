import joblib

MODEL_PATH = "models/utility_model.pkl"


def predict_usage():
    try:
        temperature = float(input("Enter Temperature: "))
        residents = int(input("Enter Number of Residents: "))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return

    model = joblib.load(MODEL_PATH)

    prediction = model.predict([[temperature, residents]])

    print(f"\nPredicted Utility Usage: {prediction[0]:.2f}")