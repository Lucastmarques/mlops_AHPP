# Airbnb House Price Prediction: getting started with MLOps

![GitHub repo size](https://img.shields.io/github/repo-size/Lucastmarques/mlops_AHPP?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/Lucastmarques/mlops_AHPP?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Lucastmarques/mlops_AHPP?style=for-the-badge)
![Github issues](https://img.shields.io/github/issues/Lucastmarques/mlops_AHPP?style=for-the-badge)
![Bitbucket open pull requests](https://img.shields.io/github/issues-pr-raw/Lucastmarques/mlops_AHPP?style=for-the-badge)

<img src="images/header.png" alt="mlops pipeline">

> Implementing a complete MLOps pipeline to deploy an AirBnb House Price Prediction (AHPP) model in production. If you are a portuguese speaker, try to watch my 5 minutes [explanation video](https://bit.ly/3OWtTy4) about this project!

### Adjustments and improvements

This project is still under development and the next steps will focus on the following tasks:

- [x] Fetch Data
- [x] Pre-processing
- [x] Data Checks
- [x] Data Segragation (train/test splitting)
- [ ] Define and create a ML model
- [ ] Train and validation
- [ ] Test
- [ ] Store in Model Registry

## 💻 Requirements

Before you get started, make sure you meet to the following requirements:

* You have installed `<conda 4.8.2 / Python 3.7.x or greater>`
* You have a `<Windows / Linux / Mac>` machine with internet connection (Although some command lines will only work in bash).
* You have good knowledge of Machine Learning, Command Line and Statics.
* You have a [wandb account](https://wandb.ai/site) and have already configured it (See this [guide](https://docs.wandb.ai/quickstart) to quickstart with W&B).

## 🚀 Installing AHPP

To install everything you need to reproduce this project, create the project env using conda:

Command Line:
```
conda env create -f environment.yml
```

Then change your env to mlops:
```
conda activate mlops
```

## ☕ Using AHPP

You may have noticed that each folder has a specific name that references the pipeline in the header image. So, to reproduce the AHPP correctly, you must follow these steps exactly, changing the input arguments as needed: 

### 1. Fetch data:

Go to `fetch_data` folder and run the command written in README. Here is an example:

```
mlflow run . -P project_name="mlops_AHPP_fetch" \
             -P artifact_name="raw_data.csv" \
             -P artifact_type="raw_data" \
             -P artifact_description="Raw Aribnb house prices in Rio de Janeiro data" \
             -P input_url="https://drive.google.com/uc?id=16zF4MHEP_bBxAEWpQgVocPupTjRRAgfP"
```

This command will download the Google Drive file passed by `input_url` argument, save the raw dataset in wandb as an artifact with the informations passed by `artifact_name`, `artifact_type` and `artifact_description`. The artifact will be saved in `mlops_AHPP_fetch` wandb project, according to `project_name` argument.

NOTE: If you haven't created a project named `mlops_AHPP_fetch` yet, the wandb will automatically handle that by creating a new project named `mlops_AHPP_fetch`.

### 2. Exploratory Data Analysis:

This step is optional, but if you really want to reproduce exactly what I did, then go to *eda* folder and run the command written in *eda/README.md*.

```
mlflow run .
```

This step will open a jupyterlab with a python notebook where you can play around and discover new thing about the dataset used in this project. 

### 3. Pre-processing

In this step the dataset will be cleaned according to what we discorvered in EDA step. To reproduce this step, go to *clean_data* folder and execute the following command:

```
mlflow run . -P input_artifact="mlops_AHPP_fetch/raw_data.csv:latest" \
             -P artifact_name="clean_data.csv" \
             -P artifact_type="clean_data" \
             -P artifact_description="Clean Aribnb house prices in Rio de Janeiro data" \
             -P project_name="mlops_AHPP_preprocessing"
```

### 4. Data Segregation

To reproduce this step, go to *split_data* folder and run the following command:

```
mlflow run . -P input_artifact="mlops_AHPP_fetch/raw_data.csv:latest" \
             -P artifact_name="clean_data.csv" \
             -P artifact_type="clean_data" \
             -P artifact_description="Clean Aribnb house prices in Rio de Janeiro data" \
             -P project_name="mlops_AHPP_preprocessing"
```

### 5. Data Checks
Even though we tried to follow the pipeline presented in the header image, we have to put this step after Data Segragation, so we are able to peform a hypothesis test to reject or aprove our train/test data. To reproduce this step, go to *data_checks* folder and run:

```
mlflow run . -P reference_artifact="mlops_AHPP_split/train_data.csv:latest" \
             -P sample_artifact="mlops_AHPP_split/test_data.csv:latest" \
             -P ks_alpha=0.05 \
             -P clean_data_artifact="mlops_AHPP_preprocessing/clean_data.csv:latest"
```


## 📫 Contributing to AHPP
If you want to contribute to the AHPP project, please follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push you branch to origin: `git push origin <project_name> / <local>`
5. Create a pull request.

Alternatively, take a look at the GitHub documentation on [how to create a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## 📝 Licença

This project is under license. See the [LICENSE](LICENSE.md) file for more details.

[⬆ Back to top](https://github.com/Lucastmarques/mlops_AHPP)<us>
