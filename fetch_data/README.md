# Instructions

This module is responsible to fetch data as a new artifact in Weights and Biases projects.

Here we are using a compressed *.gz* file saved in Google Drive as `input data`. Keep in mind that for this application, the `input data` must be saved to Google Drive either as a compressed file in *.gz*/*.zip* format, with only a *.csv* file inside, or as a "normal" file in *.csv* format.

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P artifact_name="raw_data.csv" \
             -P artifact_type="raw_data" \
             -P artifact_description="Raw Aribnb house prices in Rio de Janeiro data" \
             -P input_url="https://drive.google.com/uc?id=16zF4MHEP_bBxAEWpQgVocPupTjRRAgfP"
```
