from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import Dinov2Model

class DinoModel(nn.Module):
    def __init__(self, num_maps_out, size):
        super(DinoModel, self).__init__()
        self.network = Dinov2Model.from_pretrained(f"facebook/dinov2-{size}") 
        embed_dim = self.network.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 512 * (2 if size=="base" else 1)),
            nn.BatchNorm1d(512* (2 if size=="base" else 1)),
            nn.ReLU(),
            nn.Linear(512* (2 if size=="base" else 1), num_maps_out)
        )
        for param in self.network.parameters():
            param.requires_grad = False
        
        num_layers = len(self.network.encoder.layer)
        for i in range(1, 3):
            for param in self.network.encoder.layer[num_layers - i].parameters():
                param.requires_grad = True

    def get_features(self, x):
        x  = self.network(x)  
        features = x.last_hidden_state[:, 0, :]
        features = F.normalize(features, p=2, dim=1)
        features_normalized = F.normalize(self.projection(features), p=2, dim=1)
        return features_normalized
    
    def loss(self, anchor, positive, negative, identity=False):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        distP = F.pairwise_distance(a_x, p_x)
        distN = F.pairwise_distance(a_x, n_x)
        loss = F.relu(torch.max(distP - distN + 1, torch.tensor(0.0))).mean()
        return loss
    
    def forward(self, img):
        features = self.get_features(img)
        return features