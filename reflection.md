# Reflection — Wine Quality Prediction System

## 1. What is the prediction target, and why is the original quality column excluded?
The target is wine quality classification. A wine with a score of 6 or higher is GOOD QUALITY, while a score below 6 is NEEDS IMPROVEMENT. The original quality column is excluded because it creates the target; including it as an input could cause target leakage and unreliable predictions.

## 2. What does recall for the GOOD QUALITY class tell you?
Recall measures how many actual good-quality samples the model correctly identifies. Low recall means some good-quality samples are classified as NEEDS IMPROVEMENT, which could affect business decisions when reviewing batches.

## 3. Why might the model perform differently on another producer's wine samples?
The model was trained on a particular red-wine dataset. Other producers may have different wine characteristics, production methods, or measurement conditions, which can affect prediction performance.

## 4. Why should predictions be treated as preliminary screening results?
This is a classroom prototype. It does not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions. It should only be an initial screening aid.

## 5. What additional data or testing could improve reliability?
More samples from different producers, regions, and testing conditions could improve dataset coverage. Cross-validation, further model testing, and comparison with laboratory results could also help assess reliability.
