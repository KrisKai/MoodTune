import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, EMOTION_LABELS, MODEL_PATH, FER_BACKBONE


class EmotionResNet(nn.Module):
    """ResNet-18 backbone for emotion classification."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


class EmotionEfficientNet(nn.Module):
    """EfficientNet-B0 backbone for emotion classification.

    Stronger feature extraction than ResNet-18, better accuracy on FER2013.
    Uses dropout before the classifier for regularization.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def build_model(pretrained=True):
    """Build the emotion model based on config backbone setting."""
    if FER_BACKBONE == 'efficientnet_b0':
        return EmotionEfficientNet(pretrained=pretrained)
    else:
        return EmotionResNet(pretrained=pretrained)


def load_trained_model(device='cpu'):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_emotion(model, face_tensor, device='cpu'):
    """Predict emotion with Test-Time Augmentation (original + horizontal flip)."""
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        logits1 = model(face_tensor)
        logits2 = model(torch.flip(face_tensor, dims=[3]))
        logits = (logits1 + logits2) / 2
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()
    return EMOTION_LABELS[idx], confidence, probs.squeeze().cpu().numpy()
