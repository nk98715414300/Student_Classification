import os
import joblib
import pandas as pd
from datetime import datetime

pipeline = joblib.load("student_pipeline.pkl")

print("pipeline loaded successfully!")
while True:
    try:

        hours = float(input("Enter study hours: "))
        attendance = float(input("Enter attendance: "))
        marks = float(input("Enter previous marks: "))

        if not 0<= hours <= 24:
            raise ValueError("Study hours must be between 0 and 24.")

        if not 0<= attendance <= 100:
            raise ValueError("Attendance must be between 0 and 100.")

        if not 0 <= marks <=100:
            raise ValueError("Previous marks must be between 0 and 100.")
    
        student = pd.DataFrame(
            [[hours, attendance, marks]],
            columns=["study_hours", "attendance", "previous_marks"]
        )

        prediction = pipeline.predict(student)

        probability = pipeline.predict_proba(student)

        print("\nPrediction:", prediction)
        print("Probability:", probability)

        if prediction[0] == 1:
            print("Student Result: PASS")
        else:
            print("Student Result: FAIL")

        result = "PASS" if prediction[0] == 1 else "FAIL"    

        record = pd.DataFrame([{
            "study_hours": hours,
            "attendance": attendance,
            "previous_marks": marks,
            "prediction": int(prediction[0]),
            "result": result,
            "prediction_time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }])

        file_exists = os.path.exists("predictions.csv")

        record.to_csv(
            "predictions.csv",
            mode="a",
            header=not file_exists,
            index=False
        )

        print("Student record saved successfully!")

    except ValueError as error:
        print("Invalid Input:", error)

    choice = input("\nPredict another student? (yes/no): ").lower()

    # if choice != "yes":
    #     print("Program closed!")
    #     break

    try:
        if choice == "no":
            print("Program closed!")
            break
       
        if choice != "yes" and "no":
            raise ValueError("choice must be (yes/no).")
    finally:
        pass