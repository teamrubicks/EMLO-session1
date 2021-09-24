import os
import sys
from torchvision import transforms


DEVICE = "cuda"
pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150


TRAIN_DATA_DIR = "E:/TSAI/EMLO/EMLO-session1/data/train"
VALID_DATA_DIR = "E:/TSAI/EMLO/EMLO-session1/data/validation"

TRAIN_FEATURES = "E:/TSAI/EMLO/EMLO-session1/bottleneck_features_train.npy"
VALID_FEATURES = "E:/TSAI/EMLO/EMLO-session1/bottleneck_features_valid.npy"

TRAIN_LABELS = "E:/TSAI/EMLO/EMLO-session1/train_labels.npy"
VALID_LABELS = "E:/TSAI/EMLO/EMLO-session1/valid_labels.npy"
cats_train_path = os.path.join(path, TRAIN_DATA_DIR, "cats")
# nb_train_samples = 2 * len(
#     [
#         name
#         for name in os.listdir(cats_train_path)
#         if os.path.isfile(os.path.join(cats_train_path, name))
#     ]
# )

EPOCHS = 10
BATCH_SIZE = 16


all_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

MODEL_SAVE_PATH = "E:/TSAI/EMLO/EMLO-session1/dognotdog.pt"
METRICS_PATH = "metrics.csv"
