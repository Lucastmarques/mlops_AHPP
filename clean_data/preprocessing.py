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
    columns = ['room_type', 'accommodates', 'bathrooms_text',
               'bedrooms', 'beds', 'price', 'host_listings_count',
               'availability_30', 'availability_60', 'availability_90',
               'availability_365', 'number_of_reviews', 'minimum_nights',
               'maximum_nights', 'neighbourhood_cleansed', 'host_is_superhost',
               'host_response_time', 'host_response_rate', 'instant_bookable',
               'host_identity_verified', 'host_verifications', 'amenities']
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
    columns_drop = ['room_type', 'bathrooms_text', 'price']
    columns_imputer_numerical = [
        'accommodates', 'bedrooms', 'beds', 'host_listings_count',
        'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'number_of_reviews', 'minimum_nights',
        'maximum_nights',
    ]
    columns_imputer_categorical = [
        'neighbourhood_cleansed', 'host_is_superhost', 'host_response_time',
        'host_response_rate', 'instant_bookable', 'host_identity_verified',
        'host_verifications', 'amenities'
    ]

    clean_data = raw_data.dropna(subset=columns_drop).reset_index(drop=True)

    numerical_inputer = SimpleImputer(strategy='median', missing_values=np.nan)
    array = clean_data[columns_imputer_numerical].values
    array = numerical_inputer.fit_transform(array)
    imputer_numerical_df = pd.DataFrame(
        array, columns=columns_imputer_numerical)

    numerical_inputer = SimpleImputer(
        strategy='most_frequent', missing_values=np.nan)
    array = clean_data[columns_imputer_categorical].values
    array = numerical_inputer.fit_transform(array)
    imputer_categorical_df = pd.DataFrame(
        array, columns=columns_imputer_categorical)

    clean_data = pd.concat(
        [clean_data[columns_drop], imputer_categorical_df, imputer_numerical_df],
        axis=1
    )
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
    LOGGER.info("Treating price column")
    clean_data['price'] = clean_data['price'].apply(
        lambda x: float(x[1:].replace(',', '')) if isinstance(x, str) else x)

    LOGGER.info("Treating host_response_rate column")
    clean_data["host_response_rate"] = clean_data["host_response_rate"].apply(
        lambda x: float(x.replace('%', ''))/100
    )

    LOGGER.info("Treating integer column")
    integer_columns = [
        'accommodates', 'bedrooms', 'beds', 'host_listings_count',
        'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'number_of_reviews', 'minimum_nights',
        'maximum_nights',
    ]
    clean_data[integer_columns] = clean_data[integer_columns].round(
        0).astype(int)

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
    run = wandb.init(job_type="preproccess_data")

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
        "--artifact_description",
        type=str,
        help="Description for the artifact to be created",
        default="",
        required=False
    )
    ARGS = PARSER.parse_args()
    process_args(ARGS)
