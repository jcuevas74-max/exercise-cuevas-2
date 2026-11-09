# SAD Mini-Analysis — Wine Quality Prediction System

## Problem Statement
A wine producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements.

## Users
- Quality-control staff
- Production supervisor
- Data analyst

## Inputs
Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. The original quality score is used only to create the training target.

## Process
1. Retrieve and inspect the red-wine dataset.
2. Remove duplicate records and prepare data.
3. Define target: quality >= 6 is Good Quality; quality < 6 is Needs Improvement.
4. Split into training and testing sets.
5. Impute missing numeric values, scale measurements, and train a Random Forest classifier.
6. Evaluate accuracy, precision, recall, F1-score, and confusion matrix.
7. Load the trained model in Streamlit.
8. Validate uploaded CSV columns and predict quality category.
9. Display and allow download of results.

## Outputs
- GOOD QUALITY / NEEDS IMPROVEMENT prediction
- Dataset preview and basic information
- Model evaluation summary
- Downloadable prediction results CSV

## Key Limitation
The model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. It is a preliminary screening aid and must not replace laboratory testing, sensory evaluation, food-safety procedures, or professional decisions.
