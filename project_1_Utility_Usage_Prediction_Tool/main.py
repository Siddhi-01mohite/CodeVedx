from src.data_handler import add_usage_data, update_usage_data
from src.predictor import predict_usage

while True:
    print("\n===== Utility Usage Prediction Tool =====")
    print("1. Add Usage Data")
    print("2. Update Usage Data")
    print("3. Predict Usage")
    print("4. Exit")

    choice = input("Enter Choice: ")

    if choice == "1":
        add_usage_data()

    elif choice == "2":
        update_usage_data()

    elif choice == "3":
        predict_usage()

    elif choice == "4":
        print("Exiting Program...")
        break

    else:
        print("Invalid Choice!")