import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/students.csv")

# Features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Same split for all models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Create models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss"
    )
}

# Train and evaluate
results= []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))