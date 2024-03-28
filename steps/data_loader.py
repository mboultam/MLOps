
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zenml import step
from zenml.logger import get_logger
from zenml import log_artifact_metadata
logger = get_logger(__name__)


@step
def data_loader() -> pd.DataFrame:
    """Data loading step.

    Loads the diabetes dataset from a CSV file.

    Returns:
        DataFrame containing the loaded dataset.
    """
    # Load the CSV file
    df = pd.read_csv("diabetes.csv")
     # Log metadata
    # Log metadata
    # log_artifact_metadata(
    #     artifact_name="raw_data",
    #     metadata={
    #         "num_rows": len(df),
    #         "num_columns": len(df.columns),
    #         # Add more metadata as needed
    #     }
    #)
    return df