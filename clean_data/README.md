# Instructions

This module is responsible to preprocess the raw data from last step (fetch) as a new artifact in Weights and Biases projects.

By far the most specific and longest step. In this module we had to isolate the columns we wanted from the `raw_data` artifact, remove all duplicated rows, treat the missing values by dropping or inputing their mean value, depending on the columns. Next, we had to tranform categorical variables to numeric columns using LabelEncoder, OneHotEncoder and ColumnTransformer, from scikit-learn pre-processing package. Finally, we had to normalize our dataset, except for those columns we did the one hot encode.

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P input_artifact="mlops_AHPP_fetch/raw_data.csv:latest" \
             -P artifact_name="clean_data.csv" \
             -P artifact_type="clean_data" \
             -P artifact_description="Clean Aribnb house prices in Rio de Janeiro data" \
             -P project_name="mlops_AHPP_preprocessing"
```
