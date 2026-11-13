import pandas as pd
import matplotlib.pyplot as mlt

df = pd.read_csv("predictions.csv")

result_counts = df["result"].value_counts()

result_counts.plot(kind="bar")

mlt.title("Student Prediction Results")
mlt.xlabel("Result")
mlt.ylabel("Number of Students")
mlt.xticks(rotation=0)

mlt.tight_layout()
mlt.show()