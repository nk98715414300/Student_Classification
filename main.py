import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Load Dataset
df = pd.read_csv("data/students.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nResult Distribution:")
print(df["result"].value_counts())

# 2. Visualization
plt.scatter(df["study_hours"], df["result"])
plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Study Hours vs Student Result")
plt.show()

# 3. Features and Target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# 6. Cross Validation
cv_score = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:", cv_score)
print("Average CV Accuracy:", cv_score.mean() * 100, "%")

# 7. Train Pipeline
pipeline.fit(X_train, y_train)

print("\nPipeline trained successfully!")

# 8. Save Pipeline
joblib.dump(pipeline, "student_pipeline.pkl")
print("Pipeline saved successfully!")

# 9. Prediction for New Student
while True:

    student_hours = float(input("Enter student hours: "))
    student_attendance = float(input("Enter student attendance: "))
    student_previous_marks = float(input("Enter student previous marks: "))

    new_student = pd.DataFrame(
        [[student_hours, student_attendance, student_previous_marks]],
        columns=["study_hours", "attendance", "previous_marks"]
    )

    prediction = pipeline.predict(new_student)
    probability = pipeline.predict_proba(new_student)

    print("Prediction:", prediction)
    print("Probability:", probability)

    if prediction[0] == 1:
        print("Student Result: PASS")
    else:
        print("Student Result: FAIL")

    choice = input("Enter choice (yes/no): ").lower()

    if choice == "no":
        print("Program closed!")
        break

# 10. Model Evaluation
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nActual Results:", list(y_test))
print("Predicted Results:", list(y_pred))
print("Model Accuracy:", accuracy * 100, "%")

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

display.plot()
plt.title("Student Classification Confusion Matrix")
plt.show()

# 12. Classification Report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Fail", "Pass"]
)

print("\nClassification Report:")
print(report)