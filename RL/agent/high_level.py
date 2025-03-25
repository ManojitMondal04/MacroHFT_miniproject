import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# ====================== Configuration ======================
class Config:
    device = torch.device("cpu")  # Force CPU training
    seed = 42
    lr = 0.0003                   # Slightly higher learning rate for CPU
    batch_size = 32               # Reduced batch size
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.998         # Slower epsilon decay
    target_update = 20            # Less frequent target updates
    memory_size = 5000            # Smaller replay buffer
    num_episodes = 150            # Reduced episodes
    
    # Data settings (optimized for CPU)
    max_train_rows = 20000        # 20k training samples
    max_val_rows = 4000           # 20% of training size
    
    # Technical indicators (reduced set)
    tech_indicators = ['Close', 'Volume', 'High-Low']  # Computationally efficient features

# Set random seeds
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)

# ====================== Optimized Neural Network ======================
class DQN(nn.Module):
    def _init_(self, input_dim, output_dim):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        # Weight initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        return self.net(x)

# ====================== Efficient Trading Environment ======================
class TradingEnv:
    def _init_(self, df, initial_balance=10000, transaction_cost=0.0002):
        # Preprocess data efficiently
        df = df.copy()
        df['High-Low'] = df['High'] - df['Low']  # Add computed feature
        
        self.df = df[Config.tech_indicators].values.astype(np.float32)
        self._validate_data()
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()
        
    def _validate_data(self):
        assert not np.isnan(self.df).any(), "NaN values detected"
        assert (self.df[:, 0] > 0).all(), "Invalid prices"
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0
        self.portfolio_value = self.initial_balance
        return self._get_state()
    
    def _get_state(self):
        return torch.FloatTensor(self.df[self.current_step]).to(Config.device)
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.df[self.current_step, 0]
        next_price = self.df[self.current_step + 1, 0]
        
        # Action execution with safeguards
        if action == 1 and self.holding == 0:
            cost = current_price * (1 + self.transaction_cost)
            self.holding = self.balance / cost
            self.balance = 0
        elif action == 0 and self.holding > 0:
            self.balance = self.holding * current_price * (1 - self.transaction_cost)
            self.holding = 0
        
        # Reward calculation with clipping
        new_value = self.balance + (self.holding * next_price)
        reward = np.clip(np.log(new_value / self.portfolio_value), -5, 5)
        self.portfolio_value = new_value
        self.current_step += 1
        
        return self._get_state(), reward, (self.current_step >= len(self.df) - 1), {}

# ====================== CPU-Optimized RL Agent ======================
class DQNAgent:
    def _init_(self, input_dim, output_dim):
        self.policy_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimized optimizer settings
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.lr, weight_decay=1e-5)
        self.memory = deque(maxlen=Config.memory_size)
        self.epsilon = Config.epsilon_start
        self.loss_fn = nn.HuberLoss()  # More robust than SmoothL1
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def update_model(self):
        if len(self.memory) < Config.batch_size:
            return 0
        
        # Efficient batch sampling
        batch = random.sample(self.memory, Config.batch_size)
        states = torch.stack([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch])
        
        # Q-learning update
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * Config.gamma * next_q
        
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(Config.epsilon_end, self.epsilon * Config.epsilon_decay)
        return loss.item()

# ====================== Training Loop ======================
def train():
    # Load and preprocess data efficiently
    train_df = pd.read_csv(Config.train_path, nrows=Config.max_train_rows)
    val_df = pd.read_csv(Config.val_path, nrows=Config.max_val_rows)
    
    env = TradingEnv(train_df)
    val_env = TradingEnv(val_df)
    agent = DQNAgent(len(Config.tech_indicators), 2)
    
    best_val_return = -np.inf
    returns = []
    
    for episode in range(Config.num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            
            if len(agent.memory) >= Config.batch_size:
                loss = agent.update_model()
            
            state = next_state
        
        # Periodic validation
        if episode % 5 == 0:
            val_return = 0
            val_state = val_env.reset()
            val_done = False
            
            with torch.no_grad():
                while not val_done:
                    val_action = agent.policy_net(val_state).argmax().item()
                    val_state, val_reward, val_done, _ = val_env.step(val_action)
                    val_return += val_reward
            
            returns.append(val_return)
            
            if val_return > best_val_return:
                best_val_return = val_return
                torch.save(agent.policy_net.state_dict(), "best_model.pth")
            
            print(f"Ep {episode+1:03d}/{Config.num_episodes} | "
                  f"Val: {val_return:+.2f} | ε: {agent.epsilon:.3f}")
    
    # Save final results
    plt.figure(figsize=(10, 4))
    plt.plot(returns)
    plt.title("Validation Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    plt.savefig("training_results.png")
    plt.close()

if _name_ == "_main_":
    print(f"Starting CPU-optimized training at {datetime.now().strftime('%H:%M:%S')}")
    train()
    print(f"Training completed at {datetime.now().strftime('%H:%M:%S')}")
