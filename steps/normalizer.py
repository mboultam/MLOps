from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from zenml import step

@step
def normalizer(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize/normalize features."""
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    # Convert the normalized data to DataFrame
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)
    return X_train_normalized, X_test_normalized
