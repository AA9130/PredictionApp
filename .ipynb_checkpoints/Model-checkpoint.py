# Model definition
import torch.nn as nn
import torch.nn.functional as F

class ClothingClassifier(nn.Module):
  def __init__(self):
    super(ClothingClassifier, self).__init__()
    # Convolutional layers
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    
    # Fully connected layers
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)
    
    # Regularization
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
   
  def forward(self, x):
    # Convolutional layers
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    
    # Fully connected layers 
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    
    return output