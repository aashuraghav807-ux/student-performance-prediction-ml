import pandas as pd

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())

print(data.shape)

print(data.columns)

print(data.info())

print(data.isnull().sum())

print(data["G3"].describe())

print(data["G3"].value_counts().sort_index())

print(data[["studytime", "absences", "G1", "G2", "G3"]].corr())

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


print(data.groupby("studytime")["G3"].mean())

# ==============================
# EDA VISUALIZATIONS
# ==============================

# 1. Final Grade Distribution
plt.figure()
plt.hist(data["G3"], bins=10)
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Grades")
plt.savefig("grade_distribution.png")
plt.show()


# 2. Study Time vs Final Grade
studytime_avg = data.groupby("studytime")["G3"].mean()

plt.figure()
plt.bar(studytime_avg.index, studytime_avg.values)
plt.xlabel("Study Time")
plt.ylabel("Average Final Grade (G3)")
plt.title("Study Time vs Average Final Grade")
plt.savefig("studytime_vs_grade.png")
plt.show()

print(data[["absences", "G3"]].head())

print(data.groupby("absences")["G3"].mean())

plt.figure()

plt.scatter(data["absences"], data["G3"])

plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.title("Absences vs Final Grade")

plt.savefig("absences_vs_grade.png")

plt.show()

# 4. Gender vs Final Grade

sex_avg = data.groupby("sex")["G3"].mean()

plt.figure()
plt.bar(sex_avg.index, sex_avg.values)

plt.xlabel("Gender")
plt.ylabel("Average Final Grade (G3)")
plt.title("Gender vs Average Final Grade")

plt.savefig("sex_vs_grade.png")
plt.show()

# 5. Failures vs Final Grade

failures_avg = data.groupby("failures")["G3"].mean()

plt.figure()
plt.bar(failures_avg.index, failures_avg.values)

plt.xlabel("Number of Failures")
plt.ylabel("Average Final Grade (G3)")
plt.title("Failures vs Average Final Grade")

plt.savefig("failures_vs_grade.png")
plt.show()

# 6. Age vs Final Grade

age_avg = data.groupby("age")["G3"].mean()

plt.figure()
plt.bar(age_avg.index, age_avg.values)

plt.xlabel("Age")
plt.ylabel("Average Final Grade (G3)")
plt.title("Age vs Average Final Grade")

plt.savefig("age_vs_grade.png")
plt.show()

# 7. Study Time vs Absences

plt.figure()

plt.scatter(data["studytime"], data["absences"])

plt.xlabel("Study Time")
plt.ylabel("Number of Absences")
plt.title("Study Time vs Absences")

plt.savefig("studytime_vs_absences.png")
plt.show()

# 8. Failures vs Study Time

plt.figure()

plt.scatter(data["studytime"], data["failures"])

plt.xlabel("Study Time")
plt.ylabel("Number of Failures")
plt.title("Study Time vs Failures")

plt.savefig("studytime_vs_failures.png")
plt.show()

# ==============================
# FEATURE SELECTION
# ==============================

X = data[["studytime", "absences", "G1", "G2"]]

y = data["G3"]

print("Features (X):")
print(X.head())

print("Target (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

print("Predicted G3:")
print(y_pred)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Actual vs Predicted Graph

plt.figure()

plt.scatter(y_test, y_pred)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grade")

plt.savefig("actual_vs_predicted.png")

plt.show()

# Save the trained model
joblib.dump(model, "student_performance_model.pkl")
print("Model saved as 'student_performance_model.pkl'")

# Test prediction for a new student

new_student = [[3, 5, 12, 13]]

predicted_grade = model.predict(new_student)

print("Predicted Final Grade (G3):", predicted_grade[0])

