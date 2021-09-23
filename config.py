import os
import sys
from torchvision import transforms


device = "cuda"
pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = "pt_model.h5"
# train_data_dir = os.path.join("data", "train")
# validation_data_dir = os.path.join("data", "validation")
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
nb_validation_samples = 800
epochs = 10
batch_size = 10


all_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)
