import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#1. Load dataset
df = pd.read_csv("data/students.csv")

#2. Separate features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

#3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#4. Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

#5. Train model
model.fit(X_train, y_train)

#6. Make predictions
y_pred = model.predict(X_test)

#7. Evaluate model
print("Actual Results:", list(y_test))
print("predicted Results:", list(y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#8. Save trained model
joblib.dump(model, "decision_tree_model.pkl")

print("\nModel saved successfully!")

print("\n--- Tree Depth Experiment ---")

for depth in [1, 2, 3, 4, 5, None]:
    test_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    test_model.fit(X_train, y_train)

    train_accuracy = test_model.score(X_train, y_train)
    test_accuracy = test_model.score(X_test, y_test)

    print(
        f"Depth: {depth}, "
        f"Train Accuracy: {train_accuracy:.2f}, "
        f"Test Accuracy: {test_accuracy:.2f}"
    )

# Compare Logistic Regression and Decision Tree

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

decision_tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

# Train both models
logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)

# Calculate accuracy
logistic_accuracy = logistic_model.score(X_test, y_test)
tree_accuracy = decision_tree_model.score(X_test, y_test)

print("\n--- MODEL COMPARISON ---")

print("Logistic Regression Accuarcy:", logistic_accuracy)

print("Decision Tree Accuracy:", tree_accuracy)

if logistic_accuracy > tree_accuracy:
    print("Logistic Regression performed better.")
elif tree_accuracy > logistic_accuracy:
    print("Decision Tree performed better.")
else:
    print("Both models have equal accuracy.")