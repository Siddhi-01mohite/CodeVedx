print("Program Started")

import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "student_performance.csv")

print("Looking for file at:", file_path)

df = pd.read_csv(file_path)

print("CSV Loaded Successfully")

print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# Save cleaned dataset
cleaned_file = os.path.join(current_dir, "..", "data", "cleaned_student_performance.csv")
df.to_csv(cleaned_file, index=False)

print("Data preprocessing completed successfully.")
print("Cleaned file saved!")