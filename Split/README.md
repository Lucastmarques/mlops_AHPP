# Instructions

This module is responsible to split the clean data saved in the last step (preprocessing) as two new artifact in Weights and Biases projects (`train_data.csv` and `test_data.csv`).

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P input_artifact="mlops_AHPP_preprocessing/clean_data.csv:latest" \
             -P artifact_root="data" \
             -P artifact_type="trainvaltest_data" \
             -P test_size=0.3 \
             -P stratify="null" \
             -P random_state="13" \
             -P project_name="mlops_AHPP_split"
```

Note: Only use `stratify` parameter if you have a classification problem with 2 or more classes. Remember that in binary classification you have two classes: Positive and Negative.