import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

df = pd.read_csv("data/students.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

model = DecisionTreeClassifier(
    random_state=42
)

param_grid = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy"
)

grid.fit(X, y)

best_model  = grid.best_estimator_

joblib.dump(best_model, "best_student_model.pkl")

print("Best Parameters:", grid.best_params_)
print("Best Model Saved Successfully!")