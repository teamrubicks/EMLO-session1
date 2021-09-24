import pytest
import pandas as pd
import os
import config

print(config.METRICS_PATH)


@pytest.fixture
def metrics():
    return pd.read_csv(config.METRICS_PATH)


def check_if_file_exists(_file):
    path = config.path
    for _, _, files in os.walk(path):
        if _file in files:
            return True
    return False


def test_data_existance():
    # assert "data" not in files
    # assert "data.zip" not in files
    # assert "bottleneck_features_train.npy" not in files
    # assert "bottleneck_features_train.npy.dvc" in files
    # assert "bottleneck_features_valid.npy" not in files
    # assert "bottleneck_features_valid.npy.dvc" in files
    assert not check_if_file_exists("data")
    assert not check_if_file_exists("bottleneck_features_train.npy")
    assert not check_if_file_exists("bottleneck_features_valid.npy")
    assert not check_if_file_exists("train_features.npy")
    assert not check_if_file_exists("valid_features.npy")
    assert check_if_file_exists("bottleneck_features_train.npy.dvc")
    assert check_if_file_exists("bottleneck_features_valid.npy.dvc")
    assert check_if_file_exists("train_labels.npy.dvc")
    assert check_if_file_exists("valid_labels.npy.dvc")


def test_model_existance():
    assert not check_if_file_exists("dognotdog.pt")
    assert check_if_file_exists("dognotdog.pt.dvc")
    # assert "dognotdog.pt" not in files
    # assert "dognotdog.pt.dvc" in files


def test_train_model_accuracy(metrics):
    max_acc = metrics[metrics.state == "train"]["accuracy"].max()
    assert max_acc >= 0.70


def test_train_notadog_accuracy(metrics):
    max_acc = metrics[metrics.state == "train"]["class0_accuracy"].max()
    assert max_acc >= 0.70


def test_train_dog_accuracy(metrics):
    max_acc = metrics[metrics.state == "train"]["class1_accuracy"].max()
    assert max_acc >= 0.70


def test_valid_model_accuracy(metrics):
    max_acc = metrics[metrics.state == "valid"]["accuracy"].max()
    assert max_acc >= 0.70


def test_valid_notadog_accuracy(metrics):
    max_acc = metrics[metrics.state == "valid"]["class0_accuracy"].max()
    assert max_acc >= 0.70


def test_valid_dog_accuracy(metrics):
    max_acc = metrics[metrics.state == "valid"]["class1_accuracy"].max()
    assert max_acc >= 0.70
