"""
Creator: Lucas Torres Marques
Date: 10 Jul. 2022
Test the provided model on the test artifact
and save metrics in wandb.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np
import wandb
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
LOGGER = logging.getLogger()

def process_args(args):
    
    run = wandb.init(project="mlops_AHPP_full_pipeline", job_type="test")

    LOGGER.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df_test = pd.read_csv(test_data_path)

    # Extract the target from the features
    LOGGER.info("Extracting target from dataframe")
    x_test = df_test.copy()
    y_test = x_test.pop("price")
    
    ## Download inference artifact
    LOGGER.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()
    full_pipeline_path = os.path.join(model_export_path, 'full_pipeline')
    regressor_path = os.path.join(model_export_path, 'regressor')

    ## Load the inference pipeline
    full_pipeline = mlflow.sklearn.load_model(full_pipeline_path)
    regressor = mlflow.keras.load_model(regressor_path)
    pipe = Pipeline(steps=[('full_pipeline', full_pipeline),
                           ('regressor', regressor)
                           ]
                    )

    ## Predict test data
    predict = pipe.predict(x_test)

    # Evaluation Metrics
    LOGGER.info("Evaluation metrics")
    # Metric: MAE
    mae = mean_absolute_error(y_test, predict)
    run.summary["MAE"] = mae

    # Metric: RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predict))
    run.summary["RMSE"] = rmse

    # Metric: Max Error
    max_err = max_error(y_test, predict)
    run.summary["Max Error"] = max_err
    
    # Metric: Explained Variance Score
    evs = explained_variance_score(y_test, predict)
    run.summary["Explained Variance Score"] = evs
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)