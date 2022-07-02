# Instructions

This module is responsible to preprocess the raw data from last step (fetch) as a new artifact in Weights and Biases projects.

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P input_artifact="mlops_AHPP_fetch/raw_data.csv:latest" \
             -P artifact_name="clean_data.csv" \
             -P artifact_type="clean_data" \
             -P artifact_description="Clean Aribnb house prices in Rio de Janeiro data" \
             -P project_name="mlops_AHPP_preprocessing"
```