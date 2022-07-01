"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Receive a preprocessed data and split into train and test, generating two new
artifact in wandb project.
"""
import argparse
import logging
import os
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

LOGGER = logging.getLogger()

def process_args(args):
    """
    Arguments
        args - command line arguments
        args.input_artifact: Fully qualified name for the artifact
        args.artifact_name:  Name for the W&B artifact that will be created
        args.artifact_type: Type of the artifact to create
        args.test_size: Ratio of dataset used to test
        args.random_state: Integer to use to seed the random number generator
        args.stratify: If provided, it is considered a column name to be used for stratified splitting
    """
    run = wandb.init(project=args.project_name, job_type="data_segregation")

    LOGGER.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    LOGGER.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != 'null' else None
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the name of the split plus the provided root
            artifact_name = f"{split}_{args.artifact_root}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            LOGGER.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path,index=False)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=args.artifact_type,
                description=f"{split} split of dataset {args.input_artifact}",
            )
            artifact.add_file(temp_path)

            LOGGER.info("Logging artifact")
            run.log_artifact(artifact)

            artifact.wait()


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
        "--artifact_root",
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
        "--test_size",
        type=float,
        help="Fraction of dataset or number of items to include in the test split",
        required=True
    )

    PARSER.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the splitting",
        type=int,
        required=False,
        default=42
    )

    PARSER.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  # unfortunately mlflow does not support well optional parameters
    )
    ARGS = PARSER.parse_args()
    process_args(ARGS)
