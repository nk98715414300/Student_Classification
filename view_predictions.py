import pandas as pd

df = pd.read_csv("predictions.csv")

print("\n===== STUDENT PREDICTION RECORDS =====")
print(df.to_string(index=False))

print("\nTotal Students:", len(df))

print("\nResult Summary:")
print(df["result"].value_counts())