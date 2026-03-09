# reid_engine.py
import os
import pickle
import numpy as np
import faiss
from model import PigReIDModel
import torch 
import torch.nn.functional as F
import json
import timm
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn



class PigDatabase:
    def __init__(self, farm_name, embedding_dim=256, threshold=0.75):
        """
        Initialize a PigDatabase for a farm.
        Args:
            farm_name (str): Name of the farm
            embedding_dim (int): Dimension of embeddings
            threshold (float): Cosine similarity threshold to detect unknown pigs
        """
        self.farm_name = farm_name
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.embeddings = np.zeros((0, embedding_dim), dtype='float32')
        self.labels = []
        self.index = None

        # Load existing database if exists
        self.db_path = os.path.join("farms", farm_name, "embeddings.pkl")
        if os.path.exists(self.db_path):
            self.load_database()

    def add_pig(self, embedding, label):
        """
        Add a pig embedding to the database.
        Args:
            embedding (np.array or torch.Tensor)
            label (str or int)
        """
        embedding = embedding.detach().cpu().numpy().reshape(1, -1).astype('float32')
        self.embeddings = np.vstack([self.embeddings, embedding])
        self.labels.append(label)
        self._build_index()

    def _build_index(self):
        """
        Build FAISS index for fast similarity search.
        """
        if self.embeddings.shape[0] == 0:
            self.index = None
            return
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # cosine similarity via inner product
        faiss.normalize_L2(self.embeddings)  # normalize embeddings
        self.index.add(self.embeddings)

    def find_closest(self, query_embedding):
        """
        Find closest pig in database.
        Returns label and similarity score. If below threshold, returns None.
        """
        if self.index is None:
            return None, 0.0

        query = query_embedding.detach().cpu().numpy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        D, I = self.index.search(query, k=1)
        score = float(D[0][0])
        idx = int(I[0][0])
        if score < self.threshold:
            return None, score
        return self.labels[idx], score

    def save_database(self):
        """
        Save embeddings and labels to disk.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "embeddings": self.embeddings,
            "labels": self.labels,
            "threshold": self.threshold
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Database saved to {self.db_path}")

    def load_database(self):
        """
        Load embeddings and labels from disk.
        """
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.threshold = data.get("threshold", 0.75)
        self._build_index()
        print(f"Database loaded from {self.db_path}")

# Helper function for inference

def identify_pig(image,model_checkpoint, classes_json, device="cpu"):
    """
    Identify pig based on its ear image using a trained classification model.

    Parameters:
    - image_path (str or PIL.Image): path to the pig ear image
    - model_checkpoint (str): path to the trained .pth model checkpoint
    - classes_json (str): path to JSON file mapping class indices to Pig IDs
    - device (str): 'cpu' or 'cuda'

    Returns:
    - predicted_class (str): Pig ID predicted
    - confidence (float): Probability of predicted class
    """

    # ----------------------------
    # Load classes
    # ----------------------------
    with open(classes_json, "r") as f:
        data = json.load(f)

    classes =sorted(data["classes"])

    # ----------------------------
    # Load model
    # ----------------------------
    num_classes = len(classes)
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(1024, num_classes)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # ----------------------------
    # Preprocess image
    # ----------------------------
    # if isinstance(image_path, str):
    #     image = Image.open(image_path).convert("RGB")
    # else:
    #     image = image_path

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    # ----------------------------
    # Forward pass
    # ----------------------------
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        predicted_class = classes[pred_idx.item()]

    return predicted_class, conf.item()