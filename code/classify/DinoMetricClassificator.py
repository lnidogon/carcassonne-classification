from .Classificator import Classificator
from ..train.metric.DinoModel import DinoModel
import torch
import torch.nn.functional as F
import numpy as np
from .MetricClassificator import MetricClassificator
class DinoMetricClassificator(MetricClassificator):
    def __init__(self, modelLoaction : str, device, repsLocation : str, size : str):
        print(size)
        print(modelLoaction)
        self.model = DinoModel(64, size).to(device)
        self.model.load_state_dict(torch.load(modelLoaction, map_location=device))
        self.model.eval()
        self.representations = torch.load(repsLocation)
