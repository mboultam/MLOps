#%%
import pandas as pd
from typing import Tuple
from zenml.logger import get_logger
from zenml.pipelines import pipeline
from steps import data_loader, data_preprocessor, data_splitter, normalizer


logger = get_logger(__name__)

@pipeline
def feature_engineering() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Feature engineering pipeline.

    Returns:
        Tuple containing X_train, y_train, X_test, y_test.
    """
    # Define steps
    df = data_loader()
    selected_features, y = data_preprocessor(df)
    X_train, X_test, y_train, y_test = data_splitter(selected_features, y)
    X_train, X_test = normalizer(X_train, X_test)
    
    return X_train, y_train, X_test, y_test



# %%
