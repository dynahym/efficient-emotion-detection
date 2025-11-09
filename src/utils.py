import torch
import matplotlib.pyplot as plt


def show_sample_images(loader, class_names):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()
