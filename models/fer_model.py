import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, EMOTION_LABELS, MODEL_PATH


class EmotionResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        # Convert 1-channel grayscale to 3-channel by repeating
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def load_trained_model(device='cpu'):
    model = EmotionResNet(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_emotion(model, face_tensor, device='cpu'):
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        logits = model(face_tensor)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()
    return EMOTION_LABELS[idx], confidence, probs.squeeze().cpu().numpy()
