import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/students.csv")

# Separate data
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Create K-fold cross-validation
cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(
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

# Evaluate models
for name, model in models.items():

    scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy"
    )

    print("\n", name)
    print("Fold Scores:", scores)
    print("Mean Accuracy:", scores.mean())