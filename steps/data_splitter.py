from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import log_artifact_metadata, step
@step
def data_splitter(
    selected_features: pd.DataFrame,  # Entrée: caractéristiques sélectionnées
    y: pd.Series,  # Entrée: variable cible
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=test_size, random_state=random_state)
    # log_artifact_metadata(
    #     artifact_name="data_splitter",
    #     metadata={"random_state": random_state}
    # )
    # log_artifact_metadata(
    #     artifact_name="X_train",
    #     metadata={"X_train": X_train},
    # )
    # log_artifact_metadata(
    #     artifact_name="X_test",
    #     metadata={"X_test": X_test},
    # )
    # log_artifact_metadata(
    #     artifact_name="y_train",
    #     metadata={"y_train": y_train},
    # )
    # log_artifact_metadata(
    #     artifact_name="y_test",
    #     metadata={"y_test": y_test},
    # )

    return X_train, X_test, y_train, y_test
