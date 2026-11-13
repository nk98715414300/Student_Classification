# Student Classification using Machine Learning

## About
A beginner-friendly Machine Learning project that predicts whether a student will Pass or Fail using Logistic Regression.

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Features
- Student dataset analysis
- Data visualization
- StandardScaler preprocessing
- Logistic Regression classification
- Model Pipeline
- Input validation
- Multiple student predictions
- Save predictions in CSV
- Prediction date and time
- View student records
- Visualize PASS/FAIL results

## Project Structure
```text
Student_Classification/
├── data/
│   └── students.csv
├── main.py
├── predict.py
├── view_predictions.py
├── visualize_predictions.py
├── student_pipeline.pkl
├── predictions.csv
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

For predictions:

```bash
python predict.py
```

To view records:

```bash
python view_predictions.py
```

To visualize results:

```bash
python visualize_predictions.py
```

## Important Note
This project uses a small synthetic dataset for learning purposes. Its predictions are not reliable real-world student performance assessments.