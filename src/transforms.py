from torchvision import transforms

all_transform = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)
