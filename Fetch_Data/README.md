# Instructions

This module is responsible to fetch data as a new artifact in Weights and Biases projects.

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P artifact_name="mlops_AHPP/raw_data.csv" \
             -P artifact_type="raw_data" \
             -P artifact_description="Raw Aribnb house prices in Rio de Janeiro data" \
             -P input_url="https://drive.google.com/uc?id=16zF4MHEP_bBxAEWpQgVocPupTjRRAgfP"
```