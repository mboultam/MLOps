from zenml import step
from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger
logger = get_logger(__name__)

@step
def model_promoter(
    accuracy: float,  # L'exactitude du modèle
    threshold: float = 0.8,  # Seuil d'exactitude pour la promotion
    stage: str = "production"  # La scène de la promotion
) -> bool:
    """Model promoter step.

    Promote the model to the specified stage if accuracy is above the threshold.

    Args:
        accuracy: Accuracy of the model.
        threshold: Threshold accuracy for promotion. Default is 0.8.
        stage: Stage to promote the model to. Default is "production".

    Returns:
        Whether the model was promoted or not.
    """
    is_promoted = False

    if accuracy < threshold:
        logger.info(
            f"Model accuracy {accuracy*100:.2f}% is below the threshold {threshold*100:.2f}%. Not promoting model."
        )
    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True
    # Obtenir le modèle dans le contexte actuel
        current_model = get_step_context().model

        # Obtenir le modèle qui est dans la scène de production
        client = Client()
        try:
            stage_model = client.get_model_version(
                current_model.name, stage
            )
            # Nous comparons leurs métriques
            prod_accuracy = (
                stage_model.get_artifact("sklearn_classifier")
                .run_metadata["test_accuracy"]
                .value
            )
            if float(accuracy) > float(prod_accuracy):
                # Si le modèle actuel a de meilleures métriques, nous le promouvons
                is_promoted = True
                current_model.set_stage(stage, force=True)
        except KeyError:
            # Si aucun tel modèle n'existe, le modèle actuel est promu
            is_promoted = True
            current_model.set_stage(stage, force=True)
    return is_promoted
