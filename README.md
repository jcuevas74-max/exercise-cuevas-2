# Wine Quality Prediction System

## Project Description
A data mining classroom prototype that classifies wine samples as GOOD QUALITY or NEEDS IMPROVEMENT using physicochemical measurements.

## Dataset
UCI Wine Quality Dataset — Red Wine CSV: https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction Rule
- Quality >= 6: GOOD QUALITY (1)
- Quality < 6: NEEDS IMPROVEMENT (0)

The original `quality` column is used to create the target and is excluded from input features to prevent target leakage.

## Technologies
Python, pandas, scikit-learn (Random Forest), Streamlit, joblib, GitHub.

## Installation and Running
1. Use Python 3.10 or later.
2. Install dependencies: `pip install -r requirements.txt`
3. Train/download dataset: `python train_model.py`
4. Run app: `streamlit run app.py`
5. Open `http://localhost:8501`.
6. Download the sample CSV in the app, upload it, and review/download predictions.

## Files
- `train_model.py`: downloads dataset, preprocesses, trains and evaluates model.
- `app.py`: Streamlit dashboard and CSV predictions.
- `requirements.txt`: dependencies.
- `sad_mini_analysis.md`: systems analysis and design.
- `reflection.md`: reflection answers.
- `submission_notes.md`: submission checklist.

## Limitations
This model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening results, not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
