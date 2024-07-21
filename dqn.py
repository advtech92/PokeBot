import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(self._feature_size(), 512)
        self.fc2 = nn.Linear(512, 5)  # 5 actions: up, down, left, right, interact

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _feature_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 84, 84)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (84, 84))
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW
    image = image / 255.0  # Normalize
    return torch.tensor(image, dtype=torch.float).unsqueeze(0)


def choose_action(net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(5)  # Random action
    else:
        with torch.no_grad():
            return net(state).argmax().item()


def train_step(net, optimizer, criterion, state, action, reward, next_state, done):
    q_values = net(state)
    next_q_values = net(next_state)
    q_value = q_values[0, action]
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + (0.99 * next_q_value * (1 - done))

    loss = criterion(q_value, expected_q_value.unsqueeze(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
