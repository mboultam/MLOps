from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from zenml import step
@step
def data_preprocessor(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Data preprocessing step.

    Preprocesses the loaded dataset, handling zero values and performing
    feature selection using the SelectKBest method with chi-squared scoring.

    Args:
        df: DataFrame containing the loaded dataset.

    Returns:
        Tuple containing X (features) and y (target).
    """
    # Perform preprocessing to handle 0 values
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
    df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

    # Split features and target
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']

    # Perform feature selection
    best_features = SelectKBest(score_func=chi2, k=4)
    X_selected = best_features.fit_transform(X, y)

    # Convert X_selected to DataFrame
    X_selected_df = pd.DataFrame(X_selected, columns=X.columns[best_features.get_support()])

    return X_selected_df, y
