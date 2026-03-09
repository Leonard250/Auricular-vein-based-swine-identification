# database.py
import os
import pickle
import numpy as np

class FarmDatabaseManager:
    def __init__(self, base_path="farms"):
        """
        Manage multiple farm databases.
        Args:
            base_path (str): Root directory to store farm data.
        """
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def create_farm(self, farm_name, embedding_dim=256, threshold=0.75):
        """
        Initialize a new farm database.
        """
        farm_dir = os.path.join(self.base_path, farm_name)
        os.makedirs(farm_dir, exist_ok=True)
        db_path = os.path.join(farm_dir, "embeddings.pkl")

        data = {
            "embeddings": np.zeros((0, embedding_dim), dtype="float32"),
            "labels": [],
            "threshold": threshold
        }

        with open(db_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Created new farm database at {db_path}")
        return db_path

    def load_farm(self, farm_name):
        """
        Load a farm database.
        Returns:
            data dict with embeddings, labels, threshold
        """
        db_path = os.path.join(self.base_path, farm_name, "embeddings.pkl")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"No database found for farm: {farm_name}")
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        return data

    def save_farm(self, farm_name, embeddings, labels, threshold=0.75):
        """
        Save updated embeddings and labels for a farm.
        """
        farm_dir = os.path.join(self.base_path, farm_name)
        os.makedirs(farm_dir, exist_ok=True)
        db_path = os.path.join(farm_dir, "embeddings.pkl")

        data = {
            "embeddings": embeddings,
            "labels": labels,
            "threshold": threshold
        }
        with open(db_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved farm database at {db_path}")

    def list_farms(self):
        """
        List all available farms.
        """
        return [name for name in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, name))]