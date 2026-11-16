import pandas as pd
import matplotlib.pyplot as mlt
import joblib

# Load trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Feature names
features = [
    "study_hours",
    "attendance",
    "previous_marks"
]

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print(importance_df)

# visualize
mlt.figure(figsize=(8, 5))

mlt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

mlt.xlabel("Importance Score")
mlt.ylabel("Features")
mlt.title("Random Forest Feature Importance")

mlt.tight_layout()
mlt.show()