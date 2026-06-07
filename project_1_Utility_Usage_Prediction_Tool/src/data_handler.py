import pandas as pd

FILE_PATH = "data/utility_usage.csv"


def add_usage_data():
    month = input("Enter Month: ")

    try:
        temperature = float(input("Enter Temperature: "))
        residents = int(input("Enter Number of Residents: "))
        usage = float(input("Enter Utility Usage: "))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return

    new_record = {
        "Month": month,
        "Temperature": temperature,
        "Residents": residents,
        "Usage": usage
    }

    df = pd.read_csv(FILE_PATH)

    df = pd.concat(
        [df, pd.DataFrame([new_record])],
        ignore_index=True
    )

    df.to_csv(FILE_PATH, index=False)

    print("\nRecord Added Successfully!")


def update_usage_data():
    month = input("Enter Month to Update: ")

    df = pd.read_csv(FILE_PATH)

    if month not in df["Month"].values:
        print("Month not found!")
        return

    try:
        new_usage = float(input("Enter New Usage Value: "))
    except ValueError:
        print("Invalid input! Please enter a numeric value.")
        return

    df.loc[df["Month"] == month, "Usage"] = new_usage

    df.to_csv(FILE_PATH, index=False)

    print("Record Updated Successfully!")