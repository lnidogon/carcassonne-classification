from .Classificator import Classificator
import joblib
import torch

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
class KMeansClassificator(Classificator):
    def __init__(self, kmeans_location : str, cluster2class_location : str, device):
        self.kmeans = joblib.load(kmeans_location)
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.model.eval()
        self.model.to(device)
        self.cluster2class = torch.load(cluster2class_location)
        self.device = device
    def classify(self, image_path, device):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))  
        with torch.no_grad():
            inputs = self.image_processor(images=img, return_tensors="pt").to(device)
            embedding = self.model(**inputs).last_hidden_state[:, 0, :]
            embedding = embedding.detach().cpu().numpy()
        cluster = self.kmeans.predict(embedding)[0]
        cls = self.cluster2class[cluster]        
        return cls