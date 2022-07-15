# Instructions

This module is responsible to check if everything is strictly correct in the clean and train/test data.

Here we do both deterministics and non-deterministics (statistical) tests on clean dataset and train/test data, respectively.

On deterministic tests, we first verify if our clean dataset has more than 3000 rows, then we check that the dataset has all columns we expected, with expected dtypes and within specific ranges. Finally, we also check that categorical data has the expected classes..

On hypothesis test, we did a [kolmogorov smirnov test](https://pt.wikipedia.org/wiki/Teste_Kolmogorov-Smirnov). Making a long story short, we create two hypothesis where the null hypothesis (H0) is that the two distributions (from test and train data) are identical, while the alternative (H1) is that they are not identical.

Take a look at *conftest.py* to know how we got those data from wandb and command line parser. If you wanna check the tests, take a look at *test_deterministic.py* and/or *test_hypothesis.py* 

## Run Steps

Here is an example to run this step. Inside the project root directory, run:

```bash
mlflow run . -P hydra_options="main.execute_steps='check_data'"
```
