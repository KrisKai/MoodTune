import cv2
import numpy as np
from torchvision import transforms


class FaceDetector:
    def __init__(self, img_size=48):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def detect_and_preprocess(self, frame):
        """Detect the largest face in a BGR frame and return preprocessed tensor + bbox."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None, None

        # Take the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y + h, x:x + w]
        face_tensor = self.transform(face_crop).unsqueeze(0)  # (1, 1, 48, 48)
        return face_tensor, (x, y, w, h)

    def draw_bbox(self, frame, bbox, label, confidence):
        """Draw bounding box and label on the frame."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} ({confidence:.0%})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame
