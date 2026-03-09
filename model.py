# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class PigReIDModel:
    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize the Pig Re-ID model using MobileNetV3-Small.
        Args:
            checkpoint_path (str): Path to model checkpoint.
            device (str): 'cuda' or 'cpu'. Defaults to GPU if available.
        """
        self.device = "cpu"
        
        # Build MobileNetV3-Small backbone
        self.model = self._build_model()
        self.model.to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint found, using randomly initialized model.")

        self.model.eval()
        self.preprocess = self._get_preprocess()

    def _build_model(self):
        """
        Builds the MobileNetV3-Small model and adds embedding layer.
        """
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Replace classifier with 256-dim embedding layer
        in_features = mobilenet.classifier[0].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
        )
        return mobilenet

    def _load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _get_preprocess(self):
        """
        Returns image preprocessing transform.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """
        Get 256-d embedding for a PIL image or file path.
        Args:
            image (PIL.Image or str)
        Returns:
            embedding (torch.Tensor) normalized
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().squeeze()

    def save_checkpoint(self, path, optimizer=None, epoch=None):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
        }
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")