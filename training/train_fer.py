import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FER_DATA_DIR, MODEL_PATH, EMOTION_LABELS, NUM_CLASSES,
    IMG_SIZE, BATCH_SIZE, LEARNING_RATE, EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING, BASE_DIR,
)
from models.fer_model import build_model


# ── Mixup Data Augmentation ──

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Data Loading ──

def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
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
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    total = sum(class_counts)
    weights = total / (NUM_CLASSES * class_counts.astype(float))
    print(f"Class distribution: {dict(zip(EMOTION_LABELS, class_counts))}")
    return torch.FloatTensor(weights)


# ── Freeze / Unfreeze ──

def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone FROZEN — {trainable:,} trainable params (classifier only)")


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All layers UNFROZEN — {trainable:,} trainable params")


# ── Test-Time Augmentation ──

def evaluate_with_tta(model, test_loader, device):
    """Evaluate with Test-Time Augmentation: average original + horizontal flip."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Original prediction
            logits1 = model(images)

            # Horizontally flipped prediction
            flipped = torch.flip(images, dims=[3])
            logits2 = model(flipped)

            # Average logits
            avg_logits = (logits1 + logits2) / 2
            _, predicted = avg_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total


# ── Training ──

def train_phase(model, train_loader, test_loader, class_weights, device,
                num_epochs, lr, phase_name, best_acc=0.0, best_state=None,
                train_losses=None, train_accs=None, test_accs=None):
    """Run one training phase (frozen or unfrozen)."""

    if train_losses is None:
        train_losses, train_accs, test_accs = [], [], []

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    # Use different LR for frozen vs unfrozen
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name} | Epochs: {num_epochs} | LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            use_mixup = np.random.random() < 0.5
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(images)

            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if use_mixup:
                correct += (lam * predicted.eq(labels_a).float().sum().item() +
                            (1 - lam) * predicted.eq(labels_b).float().sum().item())
            else:
                correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ── Evaluate (with TTA) ──
        test_acc = evaluate_with_tta(model, test_loader, device)
        test_accs.append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.4f} | "
              f"Test(TTA): {test_acc:.4f} | LR: {current_lr:.6f}")

        # ── Checkpoint + Early Stopping ──
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(best_state, MODEL_PATH)
            print(f"  -> New best model saved (acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, train_accs, test_accs, best_acc, best_state


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


# ── Visualization ──

def plot_results(train_losses, train_accs, test_accs, phase1_epochs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(train_losses)
    axes[0].axvline(x=phase1_epochs, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(train_accs, label='Train')
    axes[1].plot(test_accs, label='Test (TTA)')
    axes[1].axvline(x=phase1_epochs, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    gaps = [tr - te for tr, te in zip(train_accs, test_accs)]
    axes[2].plot(gaps, color='orange')
    axes[2].axvline(x=phase1_epochs, color='red', linestyle='--', alpha=0.5, label='Unfreeze')
    axes[2].set_title('Train-Test Gap')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Gap')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # TTA for final evaluation too
            logits1 = model(images)
            logits2 = model(torch.flip(images, dims=[3]))
            avg_logits = (logits1 + logits2) / 2

            _, preds = avg_logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix (with TTA)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

    print("\nClassification Report (with TTA):")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))


# ── Main ──

def main():
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader, train_dataset = get_data_loaders()
    class_weights = compute_class_weights(train_dataset)

    model = build_model(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {total_params:,} total")

    # ── Phase 1: Train classifier only (backbone frozen) ──
    freeze_backbone(model)
    train_losses, train_accs, test_accs, best_acc, best_state = train_phase(
        model, train_loader, test_loader, class_weights, device,
        num_epochs=5, lr=1e-3, phase_name="Frozen backbone (classifier only)",
    )
    phase1_epochs = len(train_losses)

    # ── Phase 2: Fine-tune entire network (all layers unfrozen) ──
    unfreeze_all(model)
    train_losses, train_accs, test_accs, best_acc, best_state = train_phase(
        model, train_loader, test_loader, class_weights, device,
        num_epochs=45, lr=LEARNING_RATE, phase_name="Full fine-tuning (all layers)",
        best_acc=best_acc, best_state=best_state,
        train_losses=train_losses, train_accs=train_accs, test_accs=test_accs,
    )

    print(f"\n{'='*60}")
    print(f"  FINAL BEST ACCURACY (with TTA): {best_acc:.4f}")
    print(f"{'='*60}")

    # Load best model for final evaluation
    model.load_state_dict(best_state)
    plot_results(train_losses, train_accs, test_accs, phase1_epochs)
    plot_confusion_matrix(model, test_loader, device)


if __name__ == '__main__':
    main()
