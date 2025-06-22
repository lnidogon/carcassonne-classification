from .Classificator import Classificator
from ..train.metric.CustomModel import SimpleMetricEmbedding
import torch
import torch.nn.functional as F
import numpy as np
from .MetricClassificator import MetricClassificator
class CustomMetricClassificator(MetricClassificator):
    def __init__(self, modelLoaction : str, device, repsLocation : str):
        self.model = SimpleMetricEmbedding(3, 256).to(device)
        self.model.load_state_dict(torch.load(modelLoaction, map_location=device))
        self.model.eval()
        self.representations = torch.load(repsLocation)
