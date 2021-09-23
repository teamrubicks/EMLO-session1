import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets
import numpy as np
import config
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from feature_loader import DatasetFromArray
from model import DogNotDog

tqdm = partial(tqdm, leave=True, position=0)


def get_features_from_images(model, dataloader):
    model.eval()
    output = np.array([])
    for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        data = data[0].to(config.DEVICE)
        feat = model(data).detach().cpu().numpy()
        if len(output) == 0:
            output = feat
        else:
            output = np.concatenate([output, feat])

    return output


def extract_features(save=False):
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
    model.to(config.DEVICE)
    del full_model

    train_features = get_features_from_images(model, trainloader)
    valid_features = get_features_from_images(model, validloader)
    if save:
        np.save(open("bottleneck_features_train.npy", "wb"), train_features)
        np.save(open("bottleneck_features_valid.npy", "wb"), valid_features)
        np.save(open("train_labels.npy", "wb"), train_labels)
        np.save(open("valid_labels.npy", "wb"), valid_labels)
    return train_features, valid_features, train_labels, valid_labels


class Record:
    def __init__(self, train_acc, train_loss, valid_acc, valid_loss):
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.valid_acc = valid_acc
        self.valid_loss = valid_loss


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

    def train(
        self,
        epochs,
        train_loader,
        valid_loader,
        optimizer,
        loss_fn,
        scheduler=None,
        device=config.DEVICE,
    ):
        for epoch in range(epochs):
            print(f"EPOCH {epoch+1}")
            self._train(train_loader, optimizer, loss_fn, device)
            self._evaluate(valid_loader, loss_fn, device)
            if scheduler:
                scheduler.step()
        return Record(self.train_acc, self.train_loss, self.valid_acc, self.valid_loss)

    def _train(self, train_loader, optimizer, loss_fn, device=config.DEVICE):
        self.model.train()
        correct = 0
        train_loss = 0
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data["data"].to(device), data["target"].to(device)
            optimizer.zero_grad()

            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            train_loss += loss.detach()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.train_loss.append(train_loss * 1.0 / len(train_loader.dataset))
        self.train_acc.append(train_loss * 100.0 / len(train_loader.dataset))
        print(
            f" Training loss = {train_loss * 1.0 / len(train_loader.dataset)}, Training Accuracy : {100.0 * correct / len(train_loader.dataset)}"
        )

    def _evaluate(self, valid_loader, loss_fn, device=config.DEVICE):
        self.model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            for _, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                data, target = data["data"].to(config.DEVICE), data["target"].to(
                    config.DEVICE
                )
                output = self.model(data)
                valid_loss += loss_fn(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdims=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss /= len(valid_loader.dataset) * 1.0
        self.valid_loss.append(valid_loss)
        self.valid_acc.append(100.0 * correct / len(valid_loader.dataset))

        print(
            f" Test loss = {valid_loss}, Test Accuracy : {100.0 * correct / len(valid_loader.dataset)}"
        )


class Trial:
    def __init__(self, name, model, args):
        self.name = name
        self.model = model
        self.args = args
        self.Record = Record
        self.Trainer = Trainer(model)

    def run(self):
        self.Record = self.Trainer.train(**self.args)

    def save_trial(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict, config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    # extract_features(True)
    # tf, vf, tl, vl = extract_features()
    # assert len(tf) == len(tl)
    # assert len(vf) == len(vl)

    model = DogNotDog().to(config.DEVICE)
    criterion = nn.functional.nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_ds = DatasetFromArray(config.TRAIN_FEATURES, config.TRAIN_LABELS)
    valid_ds = DatasetFromArray(config.VALID_FEATURES, config.VALID_LABELS)

    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=True
    )
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, pin_memory=True)

    run = Trial(
        name="First Run",
        model=model,
        args={
            "epochs": config.EPOCHS,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "optimizer": optimizer,
            "device": config.DEVICE,
            "loss_fn": criterion,
        },
    )

    run.run()
    # run.save_trial()
    print("Done!")
