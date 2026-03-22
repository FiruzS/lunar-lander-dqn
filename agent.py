import torch
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import math
from model import DQN


class ReplayBuffer:
    """Буфер воспроизведения опыта для DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Добавление перехода в буфер"""
        experience = namedtuple('Transition', 
                               field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer.append(experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Случайная выборка из буфера"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN агент для обучения и игры"""
    def __init__(self, state_size=8, action_size=4, hidden_size=64, 
                 learning_rate=0.001, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
                 memory_size=10000, batch_size=64, device=None):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Инициализация сетей
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Оптимизатор и буфер
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(memory_size)
        
    def select_action(self, state, training=True):
        """Выбор действия с использованием epsilon-greedy стратегии"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def optimize_model(self):
        """Один шаг оптимизации модели"""
        if len(self.memory) < self.batch_size:
            return
        
        # Выборка батча
        transitions = self.memory.sample(self.batch_size)
        batch = namedtuple('Batch', field_names=['state', 'action', 'reward', 'next_state', 'done'])(*zip(*transitions))
        
        # Конвертация в тензоры
        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        
        # Вычисление Q-значений для текущих состояний
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Вычисление целевых Q-значений
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Вычисление функции потерь
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # Обратное распространение и оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Обновление весов целевой сети"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Затухание epsilon для исследования"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Сохранение модели"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"Модель сохранена в {path}")
    
    def load_model(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        print(f"Модель загружена из {path}")
