import joblib
import pandas as pd

model = joblib.load("best_student_model.pkl")

study_hours = float(input("Enter study hours: "))
attendane = float(input("Enter attendance: "))
previous_marks = float(input("Enter previous marks: "))

student = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendane,
    "previous_marks": previous_marks
}])

prediction = model.predict(student)

if prediction[0] == 1:
    result = "PASS"
else:
    result = "FAIL"

print("\nPredicted Result:", result)

probabilities = model.predict_proba(student)

for class_label, probaility in zip(model.classes_, probabilities[0]):
    result_name = "PASS" if class_label == 1 else "FAIL"
    print(f"{result_name} Probability: {probaility * 100:.2f}%")