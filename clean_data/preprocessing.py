"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Receive an input data and fetch the raw data, generating a new artifact in wandb project.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import wandb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

LOGGER = logging.getLogger()


def isolate_columns(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns needed
    Args:
        raw_data(pd.DataFrame): DataFrame to filter the columns
    Returns:
        (pd.DataFrame): DataFrame with all columns needed
    If you need to change columns name, just modify the `columns` list variable.
    """
    LOGGER.info("Isolating specifics columns")
    columns = ['room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds',
               'price', 'number_of_reviews', 'review_scores_rating',
               'review_scores_accuracy', 'review_scores_cleanliness',
               'review_scores_checkin', 'review_scores_communication',
               'review_scores_location', 'review_scores_value']
    return raw_data[columns]


def remove_duplicated(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated rows in raw_data
    Args:
        raw_data(pd.DataFrame): DataFrame to drop duplicated rows
    Returns:
        (pd.DataFrame): DataFrame with no duplicated rows
    """
    LOGGER.info("Dropping duplicated rows")
    return raw_data.drop_duplicates(ignore_index=True)


def treat_missing_values(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Treat missing values from a DataFrame using drop and SimpleImputer
    Args:
        raw_data(pd.DataFrame):  DataFrame to treat missing values
    Returns:
        (pd.DataFrame): DataFrame with no missing values
    If you want change columns to dropna, just change the `columns_drop`
variable.
    """
    LOGGER.info("Treating missing values")
    columns_drop = ['room_type', 'accommodates', 'bathrooms_text', 'bedrooms',
                    'beds', 'price', 'number_of_reviews']
    columns_imputer = ['review_scores_rating', 'review_scores_accuracy',
                       'review_scores_cleanliness', 'review_scores_checkin',
                       'review_scores_communication', 'review_scores_location',
                       'review_scores_value']

    clean_data = raw_data.dropna(subset=columns_drop).reset_index(drop=True)

    inputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    array = clean_data[columns_imputer].values
    array = inputer.fit_transform(array)
    imputer_df = pd.DataFrame(array, columns=columns_imputer)
    clean_data = pd.concat([clean_data[columns_drop], imputer_df], axis=1)
    return clean_data


def treat_bathroom_text(value):
    """Treat bathroom_text column
    Args:
        value(Any): Value from bathroom_text column
    Returns:
        (float): the float treated value for bathroom_text item
    """
    if not isinstance(value, str):
        return value

    try:
        return float(value.split(' ')[0])
    except ValueError as excep:
        LOGGER.debug(excep)
        return 0.5

def treat_special_columns(raw_data: pd.DataFrame) -> pd.DataFrame:
    clean_data = raw_data.copy()

    # Treat bathrooms_text column
    LOGGER.info("Treating bathrooms_text column")
    clean_data['bathrooms'] = clean_data['bathrooms_text'].apply(
        treat_bathroom_text)
    clean_data = clean_data.drop(axis=1, labels=['bathrooms_text'])

    # Treat price column from str ($1,000.00) to float64 (1000.00)
    LOGGER.info("Treating price columns")
    clean_data['price'] = clean_data['price'].apply(
        lambda x: float(x[1:].replace(',', '')) if isinstance(x, str) else x)

    return clean_data

def process_args(args):
    """Process args passed by cmdline and fetch raw data
    Args:
        args - command line arguments
        args.input_artifact: Fully qualified name for the raw data artifact
        args.artifact_name: Name for the W&B artifact that will be created
        args.artifact_type: Type of the artifact to create
        args.artifact_description: Description for the artifact
        args.project_name: Name of WandB project you want to access/create
    """
    run = wandb.init(project=args.project_name,
                     job_type="preproccess_data")

    LOGGER.info("Dowloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    LOGGER.info("Preprocessing dataset")
    raw_data = pd.read_csv(artifact_path)
    raw_data = isolate_columns(raw_data)
    raw_data = remove_duplicated(raw_data)
    raw_data = treat_missing_values(raw_data)
    clean_data = treat_special_columns(raw_data)

    # Generate a "clean data file"
    filename = "preprocessed_data.csv"
    clean_data.to_csv(filename, index=False)

    LOGGER.info("Creating W&B artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(filename)

    LOGGER.info("Logging artifact to wandb project")
    run.log_artifact(artifact)

    LOGGER.info("Removing csv temporary file")
    os.remove(filename)
    run.finish()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Preproccessing raw data from W&B artifact",
        fromfile_prefix_chars="@"
    )

    PARSER.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name for the raw data artifact",
        required=True
    )

    PARSER.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True
    )

    PARSER.add_argument(
        "--artifact_type",
        type=str,
        help="Type of the artifact to create",
        required=False,
        default='clean_data'
    )

    PARSER.add_argument(
        "--project_name",
        type=str,
        help="Name of WandB project you want to access/create",
        required=True
    )

    PARSER.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact to be created",
        default="",
        required=False
    )
    ARGS = PARSER.parse_args()
    process_args(ARGS)
