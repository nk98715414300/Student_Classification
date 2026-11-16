import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load dataset
df = pd.read_csv("data/students.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Create Decision Tree
model = DecisionTreeClassifier(random_state=42)

# Hyperparameters to test
param_grid = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2]
}

# Cross validation
cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

# Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy"
)

# Train and search
grid.fit(X, y)

# Results
print("Best Parameters:")
print(grid.best_params_)

print("\nBest Cross-validation Accuracy:")
print(grid.best_score_)

print("\nTotal Combinations Tested:")
print(len(grid.cv_results_["params"]))