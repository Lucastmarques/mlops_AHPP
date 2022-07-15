# Instructions

This module is responsible for evaluating the deep learning model created in `neural_network` step, using some of `sklearn.metrics` functions.

The evaluation metrics are [Mean Absolute Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html); Root Mean Square Error calculated as `np.sqrt(MSE)`, where MSE is the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html); We also use [Max Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html); and finally the [Explained Variance Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html).

## Run Steps

Here is an example to run this step. Inside the project root directory, run:

```bash
mlflow run . -P hydra_options="main.execute_steps='evaluate'"
```
