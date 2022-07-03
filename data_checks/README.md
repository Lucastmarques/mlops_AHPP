# Instructions

This module is responsible to check if everything is strictly correct in the clean and train/test data.

Here we do both deterministics and non-deterministics (statistical) tests on clean dataset and train/test data, respectively.

On deterministic tests, we first verify if our clean dataset has more than 3000 rows, then we check that the dataset has all columns we expected, with expected dtypes and within specific ranges. We could also check that categorical data has the expected classes, but since we have already handled the categorical data in the data cleaning step, this is not necessary.

On hypothesis test, we did a [kolmogorov smirnov test](https://pt.wikipedia.org/wiki/Teste_Kolmogorov-Smirnov). Making a long story short, we create two hypothesis where the null hypothesis (H0) is that the two distributions (from test and train data) are identical, while the alternative (H1) is that they are not identical.

Take a look at *conftest.py* to know how we got those data from wandb and command line parser. If you wanna check the tests, take a look at *test_deterministic.py* and/or *test_hypothesis.py* 

## Run Steps

Here is an example to run this step.

```bash
mlflow run . -P reference_artifact="mlops_AHPP_split/train_data.csv:latest" \
             -P sample_artifact="mlops_AHPP_split/test_data.csv:latest" \
             -P ks_alpha=0.05 \
             -P clean_data_artifact="mlops_AHPP_preprocessing/clean_data.csv:latest"
```
