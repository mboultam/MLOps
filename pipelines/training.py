from zenml.client import Client
from zenml.pipelines import pipeline
from steps import model_trainer, model_evaluator, model_promoter
import pandas as pd
from typing import Optional

@pipeline
def training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: Optional[str] = "lg",
):
    # Entraînement du modèle
    model = model_trainer(
        X_train=X_train, 
        y_train=y_train, 
        model_type=model_type
    )

    # Évaluation du modèle
    accuracy = model_evaluator(
        model=model, 
        X_test=X_test, 
        y_test=y_test
    )

    # Promotion du modèle
    model_promoter(accuracy=accuracy)
