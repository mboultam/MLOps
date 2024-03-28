from typing import Optional
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def model_trainer(
    X_train: pd.DataFrame,
    Y_train: pd.Series,  # Y_train doit être une série, pas un DataFrame
    model_type: str = "lg",
    target: Optional[str] = "target",
) -> Annotated[
    ClassifierMixin, ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
]:
    """Configure and train a model on the training dataset.

    Args:
        X_train: The preprocessed train dataset.
        Y_train: The target column in the train dataset.
        model_type: The type of model to train.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    if model_type == "lg":
        model = LogisticRegression()
    elif model_type == "dt":
        model = DecisionTreeClassifier()
    elif model_type == "rf":
        model = RandomForestClassifier()
    elif model_type == "kn":
        model = KNeighborsClassifier()
    elif model_type == "nb":
        model = GaussianNB()
    elif model_type == "svc":
        model = SVC()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    logger.info(f"Training model {model_type}...")

    model.fit(X_train, Y_train)
    return model
