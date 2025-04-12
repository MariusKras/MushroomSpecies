"""Functions to evaluate CNN predictions"""

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from torch import nn
from lime import lime_image
from skimage import color
from typing import List
from torch import Tensor
from torch.utils.data import DataLoader

def cnn_evaluation(model: object, class_labels: List[str]) -> None:
    """Evaluate a CNN model by plotting losses and confusion matrix, and printing a classification report."""
    training_losses = model.training_losses
    validation_losses = model.validation_losses
    test_targets = model.test_targets
    test_preds = model.test_preds

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label="Training Loss", color="blue")
    plt.plot(validation_losses, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    cm = confusion_matrix(test_targets, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    ax = plt.subplot(1, 2, 2)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.grid(False)
    ax.legend_ = None
    plt.title("Confusion Matrix for the Test Data")
    plt.tight_layout()
    plt.show()
    print("Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=class_labels))

def denormalize_image(img_tensor: Tensor) -> Tensor:
    """Denormalizes a PyTorch tensor image to the [0, 1] range."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = img_tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
    img_tensor = img_tensor.clamp(0, 1)
    return img_tensor

def lime_plots(
    resnet34_unfrozen_all: nn.Module, test_loader: DataLoader, number_of_images: int
) -> None:
    """Generates LIME explanations for a set of images and displays them in a grid."""

    def batch_predict(images: List[np.ndarray]) -> np.ndarray:
        """Convert a list of NumPy arrays to predictions."""
        resnet34_unfrozen_all.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet34_unfrozen_all.to(device)
        tensor_images = [torch.tensor(img).permute(2, 0, 1) for img in images]
        tensor_images = torch.stack(tensor_images).to(device)
        with torch.no_grad():
            logits = resnet34_unfrozen_all(tensor_images)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    images, labels = next(iter(test_loader))
    images = images[:number_of_images]
    labels = labels[:number_of_images]
    lime_explainer = lime_image.LimeImageExplainer()
    images_with_explanations = []
    true_labels = labels.numpy()

    for i, img_tensor in enumerate(images):
        img = denormalize_image(img_tensor).permute(1, 2, 0).numpy()
        explanation = lime_explainer.explain_instance(
            img,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=5,
            hide_rest=False,
        )

        predicted_label = explanation.top_labels[0]
        probabilities = batch_predict([img])
        predicted_probability = probabilities[0][predicted_label]
        colored_mask = color.label2rgb(
            mask, img, alpha=0.5, bg_label=0, colors=["cyan", "red"]
        )
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        colored_mask = np.uint8(colored_mask * 255)
        blended_image = cv2.addWeighted(img, 0.4, colored_mask, 0.9, -5)
        images_with_explanations.append(
            (blended_image, true_labels[i], predicted_label, predicted_probability)
        )

    num_images = len(images_with_explanations)
    num_rows = (num_images + 4) // 5
    fig, axes = plt.subplots(num_rows, 5, figsize=(14, num_rows * 4))
    axes = axes.flatten()
    for i, (image, true_label, predicted_label, probability) in enumerate(
        images_with_explanations
    ):
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(
            f"True: {true_label}, Pred: {predicted_label}\nProb: {probability:.2f}"
        )
        ax.axis("off")
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    plt.tight_layout(h_pad=0.5)
    plt.show()

