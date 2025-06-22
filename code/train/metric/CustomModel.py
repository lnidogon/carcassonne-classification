import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, k, padding=k//2, bias=bias))
class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.conv3 = _BNReluConv(emb_size, emb_size)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.global_pool = nn.AvgPool2d(9)  
    def get_features(self, img):
        x = self.conv1(img)
        x = self.pool(x)        
        x = self.conv2(x)
        x = self.pool(x)        
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
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