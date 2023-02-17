import torch
import torchvision
from torch import nn
from d2l import torch as d2l


class Trainer:
    def __init__(self, base_model, train_transformations=None):
        self.base_model = base_model
        self.train_transformations = train_transformations
