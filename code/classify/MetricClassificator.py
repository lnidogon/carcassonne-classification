from .Classificator import Classificator
import torch
import torch.nn.functional as F
import numpy as np
import cv2 
import optparse
class MetricClassificator(Classificator):
    representations: any
    model: any
    def classify(self, image_path, device):
        image = torch.from_numpy(cv2.resize(cv2.imread(image_path), (256, 256))/255.).float().to(device)
        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            embedding = self.model(image)
        distances = []
        for class_idx, class_rep in enumerate(self.representations):
            dist = F.pairwise_distance(embedding, class_rep.unsqueeze(0))
            distances.append(dist.item())
        
        predicted_class = np.argmin(distances)
        return predicted_class