# Cancer Risk Factor Analysis

**Project Status:** Completed | **Role:** Data Scientist | **Tech Stack:** Python, Scikit-Learn, Pandas

## Project Overview
This project leverages Machine Learning to predict cancer risk levels based on patient lifestyle and clinical data. By comparing multiple algorithms (KNN, AdaBoost, and Random Forest), I built a model that not only predicts risk with high accuracy but also identifies the specific factors (like Diet or Smoking) that contribute most to the diagnosis.

## 1. Data Processing & Security
To ensure a valid model, I performed rigorous data cleaning. This included one-hot encoding categorical variables and, crucially, removing unique identifiers to prevent "Data Leakage" and ensure patient privacy.

```python
# Preventing data leakage by dropping IDs and target-related columns
features_to_drop = [
    'Patient_ID_ST0381', 'Patient_ID_ST0384',
    'Cancer_Type_Colon', 'Cancer_Type_Lung',
    'Risk_Level_Low', 'Risk_Level_Medium'
]
```
X = df.drop(columns=features_to_drop, errors='ignore')

## 2. Model Comparison Results
I trained three distinct classifiers to find the best performer for this specific dataset:

K-Nearest Neighbors (KNN): Served as a baseline distance-based model.

AdaBoost: Tested to see if boosting weak learners improved sensitivity.

Random Forest: The final chosen model due to its ability to handle non-linear relationships and provide feature importance scores.

## 3. Key Findings: Feature Importance
A critical requirement for healthcare AI is "interpretability"—understanding why a model makes a prediction. I used the Random Forest attributes to extract the top 10 risk drivers.
```python
(Note: The chart below highlights that factors like Smoking and Alcohol Use were dominant predictors in this dataset.)
# Extracting and Sorting Feature Importance
importances = model_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Filtering for the Top 10 most significant factors
top_10 = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# Visualization code
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_10, palette='viridis')
```
## 4. Patient Prediction Engine
The final model was deployed as a prediction function that accepts new patient data and outputs a specific risk categorization (Low, Medium, or High).
```python
# Example: Predicting risk for a new 62-year-old patient
patient_data = {
    'Age': 62,
    'Smoking': 10,
    'Alcohol_Use': 10,
    'Obesity': 10,
    # ... other clinical factors
}
result = model_rf.predict(test_patient)
print(f"Predicted Risk Category: {result[0]}")
```
