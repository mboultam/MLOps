# Apache Software License 2.0
# 
# Copyright (c) ZenML GmbH 2024. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
from zenml.client import Client

import os
from typing import Optional
from steps import data_loader, data_preprocessor, data_splitter, normalizer
import click
import yaml
from pipelines import (
    feature_engineering,
    inference,
    training,
)
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Starter project.

Run the ZenML starter project with basic options.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the training pipeline
    python run.py --training-pipeline

  \b 
  # Run the training pipeline with versioned artifacts
    python run.py --training-pipeline --train-dataset-version-name=1 --test-dataset-version-name=1

  \b
  # Run the inference pipeline
    python run.py --inference-pipeline

"""
)
# @click.option(
#     "--train-dataset-name",
#     default="dataset_trn",
#     type=click.STRING,
#     help="The name of the train dataset produced by feature engineering.",
# )
# @click.option(
#     "--train-dataset-version-name",
#     default=None,
#     type=click.STRING,
#     help="Version of the train dataset produced by feature engineering. "
#     "If not specified, a new version will be created.",
# )
# @click.option(
#     "--test-dataset-name",
#     default="dataset_tst",
#     type=click.STRING,
#     help="The name of the test dataset produced by feature engineering.",
# )
# @click.option(
#     "--test-dataset-version-name",
#     default=None,
#     type=click.STRING,
#     help="Version of the test dataset produced by feature engineering. "
#     "If not specified, a new version will be created.",
# )
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)
# @click.option(
#     "--no-cache",
#     is_flag=True,
#     default=False,
#     help="Disable caching for the pipeline run.",
# )
def main(
    train_dataset_name: str = "dataset_trn",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "dataset_tst",
    test_dataset_version_name: Optional[str] = None,
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        train_dataset_name: The name of the train dataset produced by feature engineering.
        train_dataset_version_name: Version of the train dataset produced by feature engineering.
            If not specified, a new version will be created.
        test_dataset_name: The name of the test dataset produced by feature engineering.
        test_dataset_version_name: Version of the test dataset produced by feature engineering.
            If not specified, a new version will be created.
        feature_pipeline: Whether to run the pipeline that creates the dataset.
        training_pipeline: Whether to run the pipeline that trains the model.
        inference_pipeline: Whether to run the pipeline that performs inference.
        no_cache: If `True` cache will be disabled.
    """
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        feature_engineering_pipeline = feature_engineering()
        X_train, y_train, X_test, y_test = feature_engineering_pipeline.run()
        

    # Execute Training Pipeline
    if training_pipeline:
        # run_args_train = {}

        # # If train_dataset_version_name is specified, use versioned artifacts
        # if train_dataset_version_name or test_dataset_version_name:
        #     # However, both train and test dataset versions must be specified
        #     assert (
        #         train_dataset_version_name is not None
        #         and test_dataset_version_name is not None
        #     )
        #     train_dataset_artifact_version = client.get_artifact_version(
        #         train_dataset_name, train_dataset_version_name
        #     )
        #     # If train dataset is specified, test dataset must be specified
        #     test_dataset_artifact_version = client.get_artifact_version(
        #         test_dataset_name, test_dataset_version_name
        #     )
        #     # Use versioned artifacts
        #     run_args_train["train_dataset_id"] = train_dataset_artifact_version.id
        #     run_args_train["test_dataset_id"] = test_dataset_artifact_version.id

        # # Run the SGD pipeline
        # pipeline_args = {}
        # if no_cache:
        #     pipeline_args["enable_cache"] = False
        # pipeline_args["config_path"] = os.path.join(config_folder, "training_sgd.yaml")
        # training.with_options(**pipeline_args)(**run_args_train)
        # logger.info("Training pipeline with SGD finished successfully!\n\n")

        # # Run the RF pipeline
        # pipeline_args = {}
        # if no_cache:
        #     pipeline_args["enable_cache"] = False
        # pipeline_args["config_path"] = os.path.join(config_folder, "training_rf.yaml")
        # training.with_options(**pipeline_args)(**run_args_train)
        ##TO DO


    # Create the feature engineering pipeline
        df = data_loader()
        selected_features, y = data_preprocessor(df)
        X_train, X_test, y_train, y_test = data_splitter(selected_features, y)
        X_train, X_test = normalizer(X_train, X_test)
    
        # Pass X_train to the training pipeline
        training_pipeline = training(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_type="lg")
        training_pipeline.run()

        logger.info("Training pipeline with lg finished successfully!\n\n")

    if inference_pipeline:
        # TO DO
        logger.info("Inference pipeline finished successfully!")


if __name__ == "__main__":
    main()
