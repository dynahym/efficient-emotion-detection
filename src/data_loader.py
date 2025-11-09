import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader