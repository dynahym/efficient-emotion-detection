import matplotlib.pyplot as plt
import json
import os


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


def save_results(results, path="../results.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {path}")