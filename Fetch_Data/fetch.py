"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Receive an input data and fetch the raw data, generating a new artifact in wandb project.
"""
import argparse
import logging
import os
import gzip
import zipfile
from shutil import copyfileobj
import wandb
import gdown

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

LOGGER = logging.getLogger()


def file_validation(filename):
    """Check if file format is valid
    Args:
        filename(str): Name of dataset file
    """
    if not filename.endswith(('.csv', '.gz', '.zip')):
        log = "File format <%s> not accepted. Allowed formats: .csv, .gz, .zip" % \
            filename.split('.')[-1]
        raise ValueError(log)


def get_csv_file(filename):
    """Get csv file from filename
    Args:
        filename(str): Name of dataset file
    Returns:
        output(str): Output csv filename
    """
    output = 'raw_data.csv'
    if filename.endswith('.zip'):
        LOGGER.info("Unzipping zip file")
        with zipfile.ZipFile(filename) as zip_ref:
            zip_ref.extractall(output)

        LOGGER.info("Removing zipped temporary file")
        os.remove(filename)

    elif filename.endswith('.gz'):
        LOGGER.info("Unzipping gz file")
        with gzip.open(filename, 'rb') as zip_ref:
            with open(output, 'wb') as file_out:
                copyfileobj(zip_ref, file_out)

        LOGGER.info("Removing zipped temporary file")
        os.remove(filename)

    else:
        LOGGER.info("File is already unzipped")
        output = filename
    return output


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
    project_name = args.artifact_name.split('/')[0]
    run = wandb.init(project=project_name,
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


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Fetch csv data from google drive",
        fromfile_prefix_chars="@"
    )

    PARSER.add_argument(
        "--input_url",
        type=str,
        help="Google Drive URL to download dataset (Allowed formats: .csv, .gz, .zip)",
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
