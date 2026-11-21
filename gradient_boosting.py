import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/students.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:", confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))