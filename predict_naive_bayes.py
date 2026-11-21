import joblib
import pandas as pd

model = joblib.load("naive_bayes_model.pkl")

print("Gaussian Naive Bayes Student Predictor")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance (%): "))
previous_marks = float(input("Enter previous marks: "))

student = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks
}])

prediction = model.predict(student)

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")

probabilities = model.predict_proba(student)[0]

print("\nPrediction Probabilities:")

for label, probability in zip(model.classes_, probabilities):
    result = "PASS" if label == 1 else "FAIL"

    print(result, ":", round(probability * 100, 2), "%")