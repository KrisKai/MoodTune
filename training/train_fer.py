import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FER_DATA_DIR, MODEL_PATH, EMOTION_LABELS, NUM_CLASSES,
    IMG_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOP_PATIENCE,
)
from models.fer_model import EmotionResNet


def get_data_loaders():
    """Load FER2013 dataset in image-folder format (train/ and test/ subdirectories)."""
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dir = os.path.join(FER_DATA_DIR, 'train')
    test_dir = os.path.join(FER_DATA_DIR, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights for imbalanced FER2013."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    total = sum(class_counts)
    weights = total / (NUM_CLASSES * class_counts.astype(float))
    return torch.FloatTensor(weights)


def train(model, train_loader, test_loader, class_weights, device):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    test_accs = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)

        # Evaluation
        test_acc = evaluate(model, test_loader, device)
        test_accs.append(test_acc)
        scheduler.step(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | LR: {current_lr:.6f}")

        # Early stopping + checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(best_model_state, MODEL_PATH)
            print(f"  -> New best model saved (acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest test accuracy: {best_acc:.4f}")
    return train_losses, test_accs, best_model_state


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def plot_results(train_losses, test_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(test_accs)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()


def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))


def main():
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader, train_dataset = get_data_loaders()
    class_weights = compute_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")

    model = EmotionResNet(pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, test_accs, best_state = train(model, train_loader, test_loader, class_weights, device)

    # Load best model for final evaluation
    model.load_state_dict(best_state)
    plot_results(train_losses, test_accs)
    plot_confusion_matrix(model, test_loader, device)


if __name__ == '__main__':
    main()
