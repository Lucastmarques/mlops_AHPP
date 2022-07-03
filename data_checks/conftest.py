"""
Creator: Lucas Torres Marques
Date: 30 Jun. 2022
Access artifacts from wandb and provide those as variables
to test files.
"""
import pytest
import pandas as pd
import wandb

run = wandb.init(project="mlops_AHPP_data_checks", job_type="data_checks")

def pytest_addoption(parser):
    """Create parse arguments to pytest"""
    parser.addoption("--clean_data_artifact", action="store")
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")

@pytest.fixture(scope="session")
def full_data(request):
    """Access wandb artifacts and return the completed clean dataset"""
    clean_data_artifact = request.config.option.clean_data_artifact
    if clean_data_artifact is None:
        pytest.fail("--clean_data_artifact missing on command line")

    local_path = run.use_artifact(clean_data_artifact).file()
    return pd.read_csv(local_path)

@pytest.fixture(scope="session")
def splitted_data(request):
    """Access wandb artifacts and return the splitted clean datasets"""
    reference_artifact = request.config.option.reference_artifact
    if reference_artifact is None:
        pytest.fail("--clean_data_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact
    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = run.use_artifact(reference_artifact).file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact(sample_artifact).file()
    sample2 = pd.read_csv(local_path)

    return sample1, sample2

@pytest.fixture(scope="session")
def ks_alpha(request):
    """Return the ks_alpha value passed by command line"""
    ks_alpha_value = request.config.option.ks_alpha
    if ks_alpha_value is None:
        pytest.fail("--ks_alpha missing on commando line")

    return float(ks_alpha_value)
        