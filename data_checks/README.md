# Instructions

This module is responsible to check if everything is strictly correct in the clean and train/test data.

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P reference_artifact="mlops_AHPP_split/train_data.csv:latest" \
             -P sample_artifact="mlops_AHPP_split/test_data.csv:latest" \
             -P ks_alpha=0.05 \
             -P clean_data_artifact="mlops_AHPP_preprocessing/clean_data.csv:latest"
```