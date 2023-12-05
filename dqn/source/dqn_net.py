import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    
    def __init__(self, in_channels=4, n_actions=9):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First convolutional layer: 32 filters of 8x8 with stride 4
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Second convolutional layer: 64 filters of 4x4 with stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Third convolutional layer: 64 filters of 3x3 with stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Flatten the tensor for the fully-connected layers
            nn.Flatten()
        )

        # Fully connected layer with 512 rectified units
        # Note: Final spatial dimension is calculated to be 7x7 after undergoing three convolution layers 
        self.fc1 = nn.Linear(7 * 7 * 64, 512)

        # Fully connected layer with n_actions outputs
        self.fc2 =  nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


"""
Used for testing purposes for CartPole
This class is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
class DQN_RAM(nn.Module):
    def __init__(self, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
