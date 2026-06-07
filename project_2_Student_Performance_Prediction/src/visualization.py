import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Locate dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "student_performance.csv")

# Load dataset
df = pd.read_csv(file_path)

# Graph 1: Attendance vs Final Marks
plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["Final_Marks"])
plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.grid(True)

image_path = os.path.join(current_dir, "..", "images", "attendance_vs_marks.png")
plt.savefig(image_path)
plt.show()

print("Graph saved successfully!")

# Graph 2: Study Hours vs Final Marks

plt.figure(figsize=(8,5))
plt.scatter(df["Study_Hours"], df["Final_Marks"])
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.grid(True)

image_path = os.path.join(current_dir, "..", "images", "studyhours_vs_marks.png")
plt.savefig(image_path)
plt.show()

print("Graphs saved successfully!")

# Graph 3: Marks Distribution Histogram

plt.figure(figsize=(8,5))
plt.hist(df["Final_Marks"], bins=8)

plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")

image_path = os.path.join(
    current_dir,
    "..",
    "images",
    "score_distribution.png"
)

plt.savefig(image_path)
plt.show()

print("Histogram saved successfully!")

# Graph 4: Correlation Heatmap

plt.figure(figsize=(8,6))

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

image_path = os.path.join(
    current_dir,
    "..",
    "images",
    "correlation_heatmap.png"
)

plt.savefig(image_path)
plt.show()

print("Heatmap saved successfully!")

# Graph 5: Box Plot of Final Marks

plt.figure(figsize=(8,5))

sns.boxplot(y=df["Final_Marks"])

plt.title("Box Plot of Final Marks")

image_path = os.path.join(
    current_dir,
    "..",
    "images",
    "boxplot_scores.png"
)

plt.savefig(image_path)
plt.show()

print("Box plot saved successfully!")