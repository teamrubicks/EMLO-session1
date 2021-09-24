import os
import sys


DEVICE = "cuda"
# pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath("")
sys.path.append(path)
# dimensions of our images.
img_width, img_height = 150, 150


TRAIN_DATA_DIR = os.path.join(path, "data", "train")
VALID_DATA_DIR = os.path.join(path, "data", "validation")

TRAIN_FEATURES = os.path.join(
    path, "bottleneck_features", "bottleneck_features_train.npy"
)

VALID_FEATURES = os.path.join(
    path, "bottleneck_features", "bottleneck_features_valid.npy"
)


TRAIN_LABELS = os.path.join(path, "bottleneck_features", "train_labels.npy")
VALID_LABELS = os.path.join(path, "bottleneck_features", "valid_labels.npy")


EPOCHS = 10
BATCH_SIZE = 16


MODEL_SAVE_PATH = os.path.join(path, "models", "dognotdog.pt")
METRICS_PATH = os.path.join(path, "metrics", "metrics.csv")
