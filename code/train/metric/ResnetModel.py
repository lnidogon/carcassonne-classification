from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet50Model(nn.Module):
    def __init__(self, num_maps_out, pretrained : bool):
        super(ResNet50Model, self).__init__()
        self.network = models.resnet50(pretrained=pretrained)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_maps_out)
        ) 

    def get_features(self, x):
        x = self.network(x)  
        return x
    
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