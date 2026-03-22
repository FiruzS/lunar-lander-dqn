import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network для Lunar Lander
    Архитектура сети для обработки 8 состояний и 4 действий
    """
    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        super(DQN, self).__init__()
        
        # Входной слой: 8 состояний -> 64 нейрона
        self.fc1 = nn.Linear(state_size, hidden_size)
        # Скрытый слой: 64 -> 64 нейрона
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Выходной слой: 64 -> 4 действия
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        """Прямое распространение с ReLU активацией"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Без активации на выходе (Q-значения)
