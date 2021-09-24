import torch.nn.functional as F
import torch.nn as nn


class DogNotDog(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.Conv2d(64, 2, kernel_size=3, bias=False),
            nn.AvgPool2d(3),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2)
        return F.log_softmax(x, dim=1)
