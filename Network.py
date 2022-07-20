import torch
from torch import nn
from collections import deque
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Set fixed random number seed
torch.manual_seed(42)

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_size, output_size):
    super().__init__()
    self.hidden_size =128*3 
    self.layers = nn.Sequential(
      nn.Linear(input_size, self.hidden_size),
      nn.ReLU(),
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.ReLU(),
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.ReLU(),
      nn.Linear(self.hidden_size, output_size)
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return  self.layers(x)

