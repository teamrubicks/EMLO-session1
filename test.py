import pytest
import pandas as pd
import os


def test_data_existance():
    files = os.listdir()
    assert "data" not in files
    assert "data.zip" not in files
    assert "bottleneck_features_train.npy" not in files
    assert "bottleneck_features_train.npy.dvc" in files
    assert "bottleneck_features_valid.npy" not in files
    assert "bottleneck_features_valid.npy.dvc" in files


def test_model_existance():
    files = os.listdir()
    assert "dognotdog.pt" not in files
    assert "dognotdog.pt.dvc" in files


def test_train_model_accuracy():
    metrics = pd.read_csv("metrics.csv")
    max_acc = metrics.loc[metrics.mode == "train", "accuracy"].max()
    assert max_acc >= 0.70


def test_train_notadog_accuracy():
    metrics = pd.read_csv("metrics.csv")
    max_acc = metrics.loc[metrics.mode == "train", "class0_accuracy"].max()
    assert max_acc >= 0.70


def test_train_dog_accuracy():
    metrics = pd.read_csv("metrics.csv")
    max_acc = metrics.loc[metrics.mode == "train", "class1_accuracy"].max()
    assert max_acc >= 0.70
