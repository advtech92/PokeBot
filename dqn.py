import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        
        self.fc_input_dim = self._get_conv_output((3, 160, 240))  # (channels, height, width)
        self.fc = nn.Linear(self.fc_input_dim, 5)  # 5 actions: up, down, left, right, interact

    def _get_conv_output(self, shape):
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (240, 160))
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW
    image = image / 255.0  # Normalize
    return torch.tensor(image, dtype=torch.float).unsqueeze(0)


def choose_action(net, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(5)  # Random action
    else:
        with torch.no_grad():
            return net(state).argmax().item()


def train_step(net, optimizer, criterion, state, action, reward, next_state, done):
    target = reward + (0.99 * net(next_state).max().item() * (1 - done))
    target = torch.tensor(target).float().unsqueeze(0)  # Convert target to tensor and match shape
    output = net(state)[0][action].unsqueeze(0)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
