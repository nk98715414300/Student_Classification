import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/students.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Guassian Naive Bayes Results")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))
print("\nclassification Report:", classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "naive_bayes_model.pkl")
print("\nModel saved successfully!")