import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "fetch_data"),
            "main",
            parameters={
                "input_url": config["data"]["input_url"],
                "artifact_name": "raw_data.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            }
        )

    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "clean_data"),
            "main",
            parameters={
                "input_artifact": "raw_data.csv:latest",
                "artifact_name": "clean_data.csv",
                "artifact_type": "clean_data",
                "artifact_description": "Data with preprocessing applied"
            }
        )

    if "segregate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "split_data"),
            "main",
            parameters={
                "input_artifact": "clean_data.csv:latest",
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"],
                "random_state": config["main"]["random_seed"]
            }
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "data_checks"),
            "main",
            parameters={
                "clean_data_artifact": "clean_data.csv:latest",
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact": config["data"]["sample_dataset"],
                "ks_alpha": config["data"]["ks_alpha"]
            }
        )

    if "neural_network" in steps_to_execute:
        # Serialize decision tree configuration
        model_config = os.path.abspath("neural_network_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["neural_network_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "neural_network"),
            "main",
            parameters={
                "train_data": "train_data.csv:latest",
                "model_config": model_config,
                "export_artifact": config["neural_network_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"],
                "target": config["data"]["target"]
            }
        )

    if "evaluate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['neural_network_pipeline']['export_artifact']}:latest",
                "test_data": "test_data.csv:latest"
            }
        )


if __name__ == "__main__":
    process_args()