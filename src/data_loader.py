import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=64):
    """
    Create data loaders for FER2013 dataset.
    
    Args:
        data_dir (str): Path to the dataset directory containing 'train' and 'test' subdirectories
        batch_size (int): Batch size for training and testing
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Data augmentation and normalization for training
    # FER2013 images are 48x48 grayscale
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip for augmentation
        transforms.RandomRotation(degrees=10),  # Slight rotation for robustness
        transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),  # Random crop with slight zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1] for single channel
    ])
    
    # Only normalization for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=test_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader