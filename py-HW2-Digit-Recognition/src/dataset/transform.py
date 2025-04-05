import torchvision.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(
        T.ToTensor()
    )  # Converts NumPy array to Tensor and normalizes to [0, 1]
    return T.Compose(transforms)
