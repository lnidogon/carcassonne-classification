from .Classificator import Classificator
from ..train.metric.ResnetModel import ResNet50Model
import torch
import torch.nn.functional as F
import numpy as np
from .MetricClassificator import MetricClassificator
class ResnetMetricClassificator(MetricClassificator):
    def __init__(self, modelLoaction : str, device, repsLocation : str, pretrained : bool):
        self.model = ResNet50Model(64, pretrained=pretrained).to(device)
        self.model.load_state_dict(torch.load(modelLoaction, map_location=device))
        self.model.eval()
        self.representations = torch.load(repsLocation)