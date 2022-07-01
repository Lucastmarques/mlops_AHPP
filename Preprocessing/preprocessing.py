"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Receive an input data and fetch the raw data, generating a new artifact in wandb project.
"""
import argparse
import logging
import os
from sckitlearn.model_selection import train_test_split
import wandb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

LOGGER = logging.getLogger()

def isolate_columns():
    pass

def remove_duplicated():
    pass

def treat_missing_values():
    pass

def categorical_to_numeric():
    pass

def process_args(args):
    """Process args passed by cmdline and fetch raw data
    Args:
        args - command line arguments
        args.input_url: Google Drive URL to download dataset
        (Allowed formats: .csv, .gz, .zip)
        args.artifact_name: Name for the W&B artifact that will be created
        args.artifact_type: Type of the artifact to create
        args.artifact_description: Description for the artifact
    """
    run = wandb.init(project=args.project_name,
                     job_type="fetch_data")

    LOGGER.info("Dowloading file from %s", args.input_url)
    filename = gdown.download(args.input_url, quiet=False)
    file_validation(filename)
    raw_data = get_csv_file(filename)

    LOGGER.info("Creating W&B artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(raw_data)

    LOGGER.info("Logging artifact to wandb project")
    run.log_artifact(artifact)

    LOGGER.info("Removing csv temporary file")
    os.remove(raw_data)

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
