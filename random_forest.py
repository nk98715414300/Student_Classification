import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#1. Load dataset
df = pd.read_csv("data/students.csv")

#2. Separate features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

#3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

#4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=42
)

#5. Train model
model.fit(X_train, y_train)

#6. Make predictions
y_pred = model.predict(X_test)

#7. Evaluate model
print("Actual Results:", list(y_test))
print("Predicted Results:", list(y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#8. OOB Score
print("\nOOB Score:", model.oob_score_)

#9. Save model
joblib.dump(model, "random_forest_model.pkl")

print("\nRandom Forest Model Saved Successfully!")