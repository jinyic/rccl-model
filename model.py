import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


net_ = Net()

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs are available. Using DataParallel to distribute the model.")
    net_ = nn.DataParallel(net_)
    print(net_)

# Move the model to GPU if available
if torch.cuda.is_available():
    if torch.version.hip:
        print(f"Using HIP")
    # do something specific for HIP
    elif torch.version.cuda:
        # do something specific for CUDA
        print(f"Using Cuda")
    net_.cuda()

def get_net():
    return net_

device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_device():
    return device_
