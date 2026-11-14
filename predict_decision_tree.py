import joblib
import pandas as pd

# Load trained Decision tree
model = joblib.load("decision_tree_model.pkl")

# Take student details
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance:  "))
previous_marks = float(input("Enter previous marks: "))

# Prepare input
student = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks
}])

# Make prediction
prediction = model.predict(student)

print("\nPredicted Results:", prediction[0])

if prediction[0] == 0:
    print("Predicted Class: 0")
else:
    print("Predicted Class: 1")