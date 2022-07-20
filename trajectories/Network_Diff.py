import torch
from torch import nn
from collections import deque
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Set fixed random number seed
torch.manual_seed(42)

# mass 6.4
# I 0.68

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_size, output_size, X, Y, X_true, X_true_next):
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

    self.pushdata = Pushdata(X, Y, X_true, X_true_next)
    self.loader = DataLoader(self.pushdata, batch_size=32, shuffle=True)


  def forward(self, x):
    '''
      Forward pass
    '''
    return  self.layers(x)

  def remember(self, learning_x, learning_y):
    self.pushdata.remember(learning_x, learning_y)


class Pushdata(Dataset):

  def __init__(self, X, Y, X_true, X_true_next):
    self.X = X
    self.Y = Y
    self.X_true = X_true
    self.X_true_next = X_true_next

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # overwrite index
    index =  np.random.randint(0, self.X.shape[0])
    _x = self.X[index]
    _y = self.Y[index]
    _xtrue = self.X_true[index]
    _xtrue_next = self.X_true_next[index]

    return _x, _y, _xtrue, _xtrue_next


