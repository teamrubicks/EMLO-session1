import pytest
import pandas as pd
import os




def test_data_existance():
    files = os.listdir()
    assert 'data' not in files

def test_model_existance():
    files = os.listdir()
    assert 'model.h5' not in files

def test_train_model_accuracy():
    metrics = pd.read_csv('metrics.csv')
    max_acc = metrics.loc[metrics.mode=='train','accuracy'].max()
    assert max_acc>= .70

def test_train_notadog_accuracy():
    metrics = pd.read_csv('metrics.csv')
    max_acc = metrics.loc[metrics.mode=='train','class0_accuracy'].max()
    assert max_acc>= .70

def test_train_dog_accuracy():
    metrics = pd.read_csv('metrics.csv')
    max_acc = metrics.loc[metrics.mode=='train','class1_accuracy'].max()
    assert max_acc>= .70