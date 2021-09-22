from numpy.lib.function_base import extract
import pytorch_lightning as pl
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np
import config

from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, leave=True, position=0)


def get_features_from_images(model, dataloader):
    model.eval()
    output = np.array([])
    for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        data = data[0].to("cuda")
        feat = model(data).detach().cpu().numpy()
        if len(output) == 0:
            output = feat
        else:
            output = np.concatenate([output, feat])
    return output


def extract_features():
    train_data = datasets.ImageFolder(
        config.TRAIN_DATA_DIR, transform=config.all_transform
    )
    valid_data = datasets.ImageFolder(
        config.VALID_DATA_DIR, transform=config.all_transform
    )
    train_labels = [lab for _, (_, lab) in enumerate(train_data)]
    valid_labels = [lab for _, (_, lab) in enumerate(valid_data)]
    trainloader = DataLoader(train_data, batch_size=8, shuffle=False)
    validloader = DataLoader(valid_data, batch_size=8, shuffle=False)

    full_model = models.vgg16(pretrained=True)
    model = nn.Sequential(*nn.ModuleList([full_model.features, full_model.avgpool]))
    print(model)
    model.to("cuda")

    train_features = get_features_from_images(model, trainloader)
    valid_features = get_features_from_images(model, validloader)
    np.save(open("bottleneck_features_train.npy", "wb"), train_features)
    np.save(open("bottleneck_features_valid.npy", "wb"), valid_features)
    np.save(open("train_labels.npy", "wb"), train_labels)
    np.save(open("valid_labels.npy", "wb"), valid_labels)
    return train_features, valid_features, train_labels, valid_labels


if __name__ == "__main__":
    # extract_features(True)
    tf, vf, tl, vl = extract_features()
    assert len(tf) == len(tl)
    assert len(vf) == len(vl)
    assert tf[0].shape == (512, 7, 7)
    assert vf[200].shape == (512, 7, 7)
