from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import ClassifierMixin
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def model_evaluator(
    model: ClassifierMixin,  # Le modèle entraîné
    X_test: pd.DataFrame,  # Les données de test
    y_test: pd.Series,  # Les étiquettes de test
) -> float:
    """Evaluate the model using test data and return accuracy."""
    # Effectuer des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")

    # Imprimer le rapport de classification
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Imprimer la matrice de confusion
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    return accuracy
