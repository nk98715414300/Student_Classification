import joblib
import matplotlib.pyplot as mlt
from sklearn.tree import plot_tree

# Load trained model
model = joblib.load("decision_tree_model.pkl")

# Visualize Decision Tree
mlt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=[
        "study_hours",
        "attendance",
        "previous_marks"
    ],
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True,
    fontsize=10
)

mlt.title("Student Classification - Decision Tree")
mlt.tight_layout()
mlt.show()