# Instructions

This module is responsible for creating and training a deep learning model using tensorflow and keras.

The network by default has six layers and the units of each layer can be verified in the file `config.yaml`, in the root of this project.

Many hyperparameters can be controlled via the `config.yaml` file, so feel free to test new networks for this problem.

## Run Steps

Here is an example to run this step. Inside the project root directory, run:

```bash
mlflow run . -P hydra_options="main.execute_steps='neural_network'"
```
