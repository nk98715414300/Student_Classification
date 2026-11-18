Student Classification using Machine Learning

About

A beginner-friendly Machine Learning project that predicts whether a student will Pass or Fail using different classification algorithms.

The prediction is based on:

- Study Hours
- Attendance
- Previous Marks

Target variable:

- "0" = Fail
- "1" = Pass

Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

Features

Data Processing

- Student dataset analysis
- Data visualization
- StandardScaler preprocessing
- Input validation

Machine Learning Models

- Logistic Regression
- Decision Tree Classification
- Random Forest Classification
- Gaussian Naive Bayes Classification
- Model Pipeline

Model Evaluation

- Accuracy score
- Confusion matrix
- Classification report
- Feature importance
- Cross-validation (4-fold Stratified Cross-validation)
- Hyperparameter tuning using GridSearchCV
- Model comparison

Predictions

- Single and multiple student predictions
- Prediction probabilities
- Save predictions in CSV
- Prediction date and time
- View student records
- Visualize PASS/FAIL results
- Save and load trained models

Project Structure

Student_Classification/
├── data/
│   └── students.csv
├── main.py
├── predict.py
├── view_predictions.py
├── visualize_predictions.py
├── decision_tree.py
├── predict_decision_tree.py
├── visualize_decision_tree.py
├── random_forest.py
├── feature_importance.py
├── compare_models.py
├── cross_validation.py
├── grid_search.py
├── save_best_model.py
├── predict_best_model.py
├── requirements.txt
└── README.md

Note: Model files and prediction CSV files are generated when the corresponding scripts run.

Installation

Install the required libraries:

pip install -r requirements.txt

Run the Project

Logistic Regression

python main.py
python predict.py

Decision Tree

python decision_tree.py
python predict_decision_tree.py
python visualize_decision_tree.py

Random Forest

python random_forest.py
python feature_importance.py

Model Comparison and Evaluation

python compare_models.py
python cross_validation.py
python grid_search.py
python save_best_model.py

The model comparison and cross-validation scripts evaluate:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gaussian Naive Bayes

Gaussian Naive Bayes is used because the dataset contains numerical features.

Prediction Using Tuned Model

python predict_best_model.py

Important Note

This project uses a small synthetic dataset containing 16 student records for learning purposes. Its predictions and evaluation scores are not reliable real-world student performance assessments.

This project is intended for educational and experimental use only.