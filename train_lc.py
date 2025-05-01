#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用TD3算法训练本地控制器(LC)，基于预处理后的区域1数据
实现马尔可夫决策过程(MDP)框架的能源管理策略
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple
import time
from datetime import datetime

# 创建存储模型的目录
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

if not os.path.exists('results'):
    os.makedirs('results')

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ESS约束 - 提前定义这些参数
MAX_CHARGE = 1.0     # 最大充电功率 (归一化后)
MAX_DISCHARGE = 1.0  # 最大放电功率 (归一化后)
MIN_ENERGY = 0.0     # 最小能量水平 (归一化后)
MAX_ENERGY = 1.0     # 最大能量水平 (归一化后)
INIT_ENERGY = 0.5    # 初始能量水平 (归一化后)

# 加载预处理后的数据
print("加载预处理后的数据...")
try:
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    
    # 加载特征名称
    with open('processed_data/feature_names.txt', 'r') as f:
        feature_names = f.read().splitlines()
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征维度: {len(feature_names)}")
    print(f"特征名称: {feature_names}")
    
    # 添加ESS能量水平维度，处理成完整的状态空间
    print("\n扩展状态空间，添加ESS能量水平...")
    # 初始化ESS能量水平为0.5（归一化值）
    X_train_with_energy = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
    X_test_with_energy = np.zeros((X_test.shape[0], X_test.shape[1] + 1))
    
    # 复制原始特征
    X_train_with_energy[:, :-1] = X_train
    X_test_with_energy[:, :-1] = X_test
    
    # 初始化能量水平为0.5（归一化值，表示50%容量）
    X_train_with_energy[:, -1] = INIT_ENERGY
    X_test_with_energy[:, -1] = INIT_ENERGY
    
    # 使用扩展后的状态空间
    X_train = X_train_with_energy
    X_test = X_test_with_energy
    
    # 添加能量水平到特征名称列表
    feature_names.append('ESS Energy Level')
    print(f"扩展后训练集大小: {X_train.shape}")
    print(f"扩展后特征维度: {len(feature_names)}")
    
except FileNotFoundError as e:
    print(f"错误: 找不到预处理后的数据文件: {e}")
    print("请先运行 preprocess_data.py 生成数据")
    exit(1)

# 解析特征，确定状态空间的映射
# 状态空间: st = (dyt, dwt, ht, Et, pLt, p̃Lt, pgt, p̃gt, λt, λ̃t)
feature_map = {}
for i, feature in enumerate(feature_names):
    feature_map[feature] = i

# 找到相关特征的索引
day_of_year_idx = feature_map.get('Day of Year')
day_of_week_idx = feature_map.get('Day of Week')
hour_of_day_idx = feature_map.get('Hour of Day')
energy_level_idx = feature_map.get('ESS Energy Level')  # ESS能量水平索引
actual_load_idx = feature_map.get('Actual Load Value')
pred_load_idx = feature_map.get('Predicted Load Value')
actual_pv_idx = feature_map.get('Actual PV Value')
pred_pv_idx = feature_map.get('Predicted PV Value')
actual_price_idx = feature_map.get('Actual Price Value')
pred_price_idx = feature_map.get('Predicted Price Value')

print("\n特征映射:")
for feature, idx in feature_map.items():
    print(f"  {feature}: {idx}")

# 按时间特征对数据进行排序，确保连续时间步
print("\n按时间特征对数据排序，确保连续时间步...")
if day_of_year_idx is not None and day_of_week_idx is not None and hour_of_day_idx is not None:
    # 创建排序键（年中日、周中日、日中时）
    time_features = np.column_stack((
        X_train[:, day_of_year_idx], 
        X_train[:, day_of_week_idx], 
        X_train[:, hour_of_day_idx]
    ))
    
    # 按年中日、周中日、日中时的顺序排序
    sorted_indices = np.lexsort((X_train[:, hour_of_day_idx], 
                                 X_train[:, day_of_week_idx], 
                                 X_train[:, day_of_year_idx]))
    
    X_train = X_train[sorted_indices]
    
    print("数据已按时间特征排序")
    print("排序后的前5个时间样本:")
    for i in range(5):
        print(f"  样本 {i}: 年中日={X_train[i, day_of_year_idx]:.4f}, "
              f"周中日={X_train[i, day_of_week_idx]:.4f}, "
              f"日中时={X_train[i, hour_of_day_idx]:.4f}")
    
    # 对测试集也进行相同的排序
    test_sorted_indices = np.lexsort((X_test[:, hour_of_day_idx], 
                                      X_test[:, day_of_week_idx], 
                                      X_test[:, day_of_year_idx]))
    X_test = X_test[test_sorted_indices]

# TD3算法参数设置，根据MDP框架
# TD3超参数
BUFFER_SIZE = int(1e5)      # 经验回放缓冲区大小
BATCH_SIZE = 64             # 批次大小
GAMMA = 0.99                # 折现因子 γ
TAU = 0.001                 # 目标网络软更新参数
LR_ACTOR = 0.0005           # 演员网络学习率
LR_CRITIC = 0.0005          # 评论家网络学习率
POLICY_NOISE = 0.2          # 策略噪声标准差
NOISE_CLIP = 0.5            # 噪声裁剪范围
POLICY_UPDATE_FREQ = 2      # 策略更新频率

# MDP配置
STATE_DIM = X_train.shape[1]  # 状态维度 (特征数量)
ACTION_DIM = 1                # 动作维度 (ESS控制输出)
MAX_ACTION = 1.0              # 最大放电功率
MIN_ACTION = -1.0             # 最小放电功率（负值表示充电）

# 成本函数参数
PRICE_WEIGHT = 1.0   # 价格权重
REG_WEIGHT = 0.01    # 正则化权重 m

print("\nMDP参数:")
print(f"状态维度: {STATE_DIM}")
print(f"动作维度: {ACTION_DIM}")
print(f"折现因子 γ: {GAMMA}")
print(f"学习率: {LR_ACTOR} (Actor), {LR_CRITIC} (Critic)")
print(f"ESS约束: 充电={MAX_CHARGE}, 放电={MAX_DISCHARGE}, 能量范围=[{MIN_ENERGY}, {MAX_ENERGY}]")
print(f"成本函数权重: 价格={PRICE_WEIGHT}, 正则化={REG_WEIGHT}")

# 定义经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # 三个隐藏层，分别有64、128和64个神经元
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.max_action * torch.tanh(self.l4(a))

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1架构
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)
        
        # Q2架构
        self.l5 = nn.Linear(state_dim + action_dim, 64)
        self.l6 = nn.Linear(64, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        
        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        
        return q1

# 噪声生成器 - Ornstein-Uhlenbeck随机过程
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# TD3算法
class TD3:
    def __init__(self, state_dim, action_dim, max_action, min_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.max_action = max_action
        self.min_action = min_action
        self.device = device
        
        self.noise = OUNoise(action_dim, sigma=POLICY_NOISE)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
        
        self.total_it = 0
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
            
        # 裁剪动作
        return np.clip(action, self.min_action, self.max_action)
    
    def train(self):
        self.total_it += 1
        
        # 如果内存中的样本不足，则跳过
        if len(self.memory) < BATCH_SIZE:
            return 0, 0
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # 更新Critic
        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(self.min_action, self.max_action)
            
            # 目标Q值
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q
            
        # 当前Q值
        current_Q1, current_Q2 = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0
        
        # 延迟策略更新
        if self.total_it % POLICY_UPDATE_FREQ == 0:
            # Actor损失
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
        return critic_loss.item(), actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# 本地控制器类
class LocalController:
    def __init__(self, state_dim, action_dim, max_action, min_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        
        # ESS约束
        self.max_charge = MAX_CHARGE
        self.max_discharge = MAX_DISCHARGE
        self.min_energy = MIN_ENERGY
        self.max_energy = MAX_ENERGY
        self.current_energy = INIT_ENERGY  # 初始能量水平
        
        # 成本函数组件跟踪
        self.cost_components = []
        self.episode_costs = []
        
        # 创建TD3代理
        self.agent = TD3(state_dim, action_dim, max_action, min_action, device)
        
    def set_ess_constraints(self, max_charge, max_discharge, min_energy, max_energy, init_energy=0.5):
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.current_energy = init_energy
        
    def determine_action(self, state, add_noise=False):
        # 确定动作（ESS充放电功率）
        action = self.agent.select_action(state, add_noise)
        
        # 应用ESS约束
        # 1. 检查当前能量水平，限制充放电动作
        current_energy = state[energy_level_idx] if energy_level_idx is not None else self.current_energy
        
        if current_energy <= self.min_energy:  # 电池电量最低，不能再放电
            action = np.maximum(0, action)  # 只能充电或不操作
        elif current_energy >= self.max_energy:  # 电池电量最高，不能再充电
            action = np.minimum(0, action)  # 只能放电或不操作
            
        # 2. 应用充放电限制
        action = np.clip(action, -self.max_charge, self.max_discharge)
        
        # 确保返回标量值而非数组
        if isinstance(action, np.ndarray):
            action = float(action.item()) if action.size == 1 else float(action[0])
            
        return action
        
    def update_energy(self, action, state=None, efficiency=0.9):
        """更新ESS能量水平
        
        参数:
            action: ESS充放电功率，正值表示放电，负值表示充电
            state: 当前状态，如果提供则从中读取能量水平
            efficiency: 充放电效率
            
        物理意义:
            - 基于250kW的最大功率和1000kWh的能量容量
            - action为归一化值，实际功率为action*250 kW
            - SOC变化量 = (功率 * 时间) / 总容量
            - 假设时间步长为1小时，则SOC变化 = 功率(kW) / 1000(kWh)
        """
        # 确保action是标量
        if isinstance(action, np.ndarray):
            action = float(action.item()) if action.size == 1 else float(action[0])
            
        # 如果提供了状态，从状态中读取当前能量水平
        if state is not None and energy_level_idx is not None:
            self.current_energy = float(state[energy_level_idx])
        
        # 功率最大值(kW)和能量容量(kWh)
        MAX_POWER_KW = 250
        ENERGY_CAPACITY_KWH = 1000
         
        if action >= 0:  # 放电：减少能量水平
            # 归一化功率 * 最大功率值(kW) / 效率 / 能量容量(kWh)
            energy_change = (action * MAX_POWER_KW) / efficiency / ENERGY_CAPACITY_KWH
            self.current_energy -= energy_change
        else:  # 充电：增加能量水平
            # 归一化功率 * 最大功率值(kW) * 效率 / 能量容量(kWh)
            energy_change = (abs(action) * MAX_POWER_KW) * efficiency / ENERGY_CAPACITY_KWH
            self.current_energy += energy_change
            
        # 确保能量水平在约束范围内
        self.current_energy = float(np.clip(self.current_energy, self.min_energy, self.max_energy))
        
        return self.current_energy
    
    def calculate_cost(self, state, action):
        """计算MDP成本函数
        
        修改后的成本函数: Ct(st, at) = (λt - λ̃t) * pESSt + m * ||pESSt||
        使用当前价格与预测价格的差值，鼓励在价格低时充电，价格高时放电
        """
        # 获取当前价格和预测价格
        current_price = 0.0
        predicted_price = 0.0
        
        if actual_price_idx is not None:
            current_price = float(state[actual_price_idx])
        
        if pred_price_idx is not None:
            predicted_price = float(state[pred_price_idx])
        #else:
            # 如果预测价格不可用，就使用当前价格作为近似
            #predicted_price = current_price
            
        # 确保action是标量
        if isinstance(action, np.ndarray):
            action = float(action.item()) if action.size == 1 else float(action[0])
            
        # 计算成本函数 - 使用当前价格与预测价格的差值
        # 注意: action>0表示放电(卖电)，action<0表示充电(买电)
        # 当 λt > λ̃t 时，放电会获得更多收益；当 λt < λ̃t 时，充电成本更低
        #print(f"当前价格: {current_price}, 预测价格: {predicted_price}")
        price_diff = current_price - predicted_price
        energy_cost = PRICE_WEIGHT * price_diff * action
        reg_term = REG_WEIGHT * np.abs(action)       # m * ||pESSt||
        
        total_cost = float(energy_cost + reg_term)
        
        # 记录详细的成本信息（可选，用于调试）
        if hasattr(self, 'cost_components') and isinstance(self.cost_components, list):
            self.cost_components.append({
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_diff': price_diff,
                'action': action,
                'energy_cost': energy_cost,
                'reg_term': reg_term,
                'total_cost': total_cost
            })
        
        return total_cost
    
    def train(self):
        return self.agent.train()
    
    def save(self, filename):
        self.agent.save(filename)
    
    def load(self, filename):
        self.agent.load(filename)

# 创建和配置本地控制器
print("\n创建本地控制器...")
local_controller = LocalController(STATE_DIM, ACTION_DIM, MAX_ACTION, MIN_ACTION)
local_controller.set_ess_constraints(MAX_CHARGE, MAX_DISCHARGE, MIN_ENERGY, MAX_ENERGY, INIT_ENERGY)

# 用于记录训练过程的列表
episode_rewards = []
critic_losses = []
actor_losses = []
energy_levels = []
time_sequences = []  # 新增：记录每个回合的时间序列数据

# 训练参数
NUM_EPISODES = 5000  # 训练回合数
MAX_T = 200         # 每个回合的最大时间步数
print(f"训练配置: {NUM_EPISODES}个回合, 每个回合{MAX_T}个连续时间步")

# 训练循环
print("\n开始训练本地控制器（使用价格差价成本函数）...")
for episode in range(NUM_EPISODES):
    # 重置环境
    total_reward = 0
    episode_costs = []
    print(f"回合: {episode+1}")
    # 重置成本组件记录
    local_controller.cost_components = []
    
    # 重置噪声过程
    local_controller.agent.noise.reset()
    
    # 随机选择起始点，然后使用连续的时间步
    # 确保我们不会超出数据集范围
    max_start_idx = len(X_train) - MAX_T
    if max_start_idx <= 0:
        print("警告: 数据集大小小于单个回合所需的时间步数，将循环使用数据")
        start_idx = 0
    else:
        start_idx = np.random.randint(0, max_start_idx)
    
    end_idx = min(start_idx + MAX_T, len(X_train))
    
    if (episode + 1) % 100 == 0 or episode == 0:
        print(f"回合 {episode+1} 使用连续时间步: {start_idx} 到 {end_idx-1}")
    
    # 初始状态
    state = X_train[start_idx].copy()
    
    # 设置初始能量水平
    if energy_level_idx is not None:
        state[energy_level_idx] = INIT_ENERGY
    local_controller.current_energy = INIT_ENERGY
    
    episode_energy = [local_controller.current_energy]
    batch_critic_losses = []
    batch_actor_losses = []
    
    # 记录时间数据以验证连续性
    time_data = []
    if day_of_year_idx is not None and day_of_week_idx is not None and hour_of_day_idx is not None:
        time_data.append((
            state[day_of_year_idx],
            state[day_of_week_idx], 
            state[hour_of_day_idx]
        ))
    
    try:
        for t in range(MAX_T):
            # 决定动作
            action = local_controller.determine_action(state, add_noise=True)
            #print(f"动作: {action}")
            # 计算成本
            cost = local_controller.calculate_cost(state, action)
            episode_costs.append(cost)
            
            # 更新ESS能量水平
            energy_level = local_controller.update_energy(action, state)
            episode_energy.append(energy_level)
            
            # 获取下一个连续时间步的状态
            next_idx = start_idx + t + 1
            if next_idx >= len(X_train):
                next_idx = next_idx % len(X_train)  # 循环回到数据集开始
            
            next_state = X_train[next_idx].copy()
            
            # 记录时间数据以验证连续性
            if day_of_year_idx is not None and day_of_week_idx is not None and hour_of_day_idx is not None:
                time_data.append((
                    next_state[day_of_year_idx],
                    next_state[day_of_week_idx], 
                    next_state[hour_of_day_idx]
                ))
            
            # 更新下一个状态中的能量水平
            if energy_level_idx is not None:
                next_state[energy_level_idx] = energy_level
                
            # 负成本作为奖励（我们想要最小化成本）
            reward = -cost
            
            # 判断是否结束
            done = (t == MAX_T - 1)
            
            # 存储经验
            local_controller.agent.memory.add(state, action, reward, next_state, done)
            
            # 训练代理
            critic_loss, actor_loss = local_controller.train()
            # 确保损失是标量
            if isinstance(critic_loss, np.ndarray):
                critic_loss = float(critic_loss.item() if critic_loss.size == 1 else critic_loss[0])
            if isinstance(actor_loss, np.ndarray):
                actor_loss = float(actor_loss.item() if actor_loss.size == 1 else actor_loss[0])
                
            batch_critic_losses.append(critic_loss)
            batch_actor_losses.append(actor_loss)
            
            # 更新状态
            state = next_state.copy()
            total_reward += reward
            
            if done:
                break
    except Exception as e:
        print(f"训练过程中出错(回合 {episode+1}): {e}")
        print(f"当前状态: {state}")
        print(f"当前动作: {action}")
        print(f"当前能量水平: {local_controller.current_energy}")
        continue
    
    # 保存本回合的成本组件记录
    if local_controller.cost_components:
        local_controller.episode_costs.append(local_controller.cost_components)
    
    # 第一个回合结束后，打印时间数据样本以验证连续性
    if episode == 0:
        print("\n第一个回合的时间数据样本（验证连续时间步）:")
        for i in range(min(20, len(time_data))):
            day_of_year, day_of_week, hour_of_day = time_data[i]
            if i > 0:
                prev_day_of_year, prev_day_of_week, prev_hour_of_day = time_data[i-1]
                day_change = day_of_year - prev_day_of_year
                week_change = day_of_week - prev_day_of_week
                hour_change = hour_of_day - prev_hour_of_day
                print(f"  时间步 {i}: 年中日={day_of_year:.4f} (变化:{day_change:.4f}), "
                      f"周中日={day_of_week:.4f} (变化:{week_change:.4f}), "
                      f"日中时={hour_of_day:.4f} (变化:{hour_change:.4f})")
            else:
                print(f"  时间步 {i}: 年中日={day_of_year:.4f}, "
                      f"周中日={day_of_week:.4f}, "
                      f"日中时={hour_of_day:.4f}")

    # 记录时间序列数据
    time_sequences.append(time_data)

    # 记录本回合的平均损失、总奖励和能量轨迹
    episode_rewards.append(total_reward)
    
    # 过滤掉非数值的损失值
    valid_critic_losses = [loss for loss in batch_critic_losses if isinstance(loss, (int, float)) and loss != 0]
    valid_actor_losses = [loss for loss in batch_actor_losses if isinstance(loss, (int, float)) and loss != 0]
    
    avg_critic_loss = np.mean(valid_critic_losses) if valid_critic_losses else 0
    avg_actor_loss = np.mean(valid_actor_losses) if valid_actor_losses else 0
    
    critic_losses.append(avg_critic_loss)
    actor_losses.append(avg_actor_loss)
    energy_levels.append(episode_energy)
    
    # 每100个回合打印训练进度
    if (episode + 1) % 100 == 0 or episode == 0:
        # 确保所有值都是标量，而不是数组
        total_reward_val = float(total_reward) if isinstance(total_reward, (np.ndarray)) else total_reward
        avg_cost = float(-total_reward_val/MAX_T) if isinstance(total_reward, (np.ndarray)) else -total_reward/MAX_T
        avg_critic_loss_val = float(avg_critic_loss) if isinstance(avg_critic_loss, (np.ndarray)) else avg_critic_loss
        avg_actor_loss_val = float(avg_actor_loss) if isinstance(avg_actor_loss, (np.ndarray)) else avg_actor_loss
        current_energy_val = float(local_controller.current_energy) if isinstance(local_controller.current_energy, (np.ndarray)) else local_controller.current_energy
        
        print(f"回合: {episode+1}/{NUM_EPISODES}, "
              f"总奖励: {total_reward_val:.4f}, "
              f"平均成本: {avg_cost:.4f}, "
              f"Critic损失: {avg_critic_loss_val:.4f}, "
              f"Actor损失: {avg_actor_loss_val:.4f}, "
              f"最终能量水平: {current_energy_val:.2f}")
        
        # 保存检查点模型
        local_controller.save(f"saved_models/lc_model_episode_{episode+1}.pt")

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存最终模型
model_path = f"saved_models/lc_model_{timestamp}.pt"
local_controller.save(model_path)
print(f"\n最终模型已保存: {model_path}")

# 同时保存为标准名称
standard_path = "saved_models/lc_model_final.pt"
local_controller.save(standard_path)
print(f"标准模型已保存: {standard_path}")

# 保存训练历史
results_path = f"results/training_history_{timestamp}"
np.save(f"{results_path}_rewards.npy", np.array(episode_rewards))
np.save(f"{results_path}_critic_losses.npy", np.array(critic_losses))
np.save(f"{results_path}_actor_losses.npy", np.array(actor_losses))
# 保存连续时间序列数据
np.save(f"{results_path}_time_sequences.npy", np.array(time_sequences, dtype=object))
print(f"训练历史已保存: {results_path}*.npy")

# 绘制训练曲线
plt.figure(figsize=(15, 18))  # 增加图表高度以容纳新的子图

# 绘制奖励曲线
plt.subplot(3, 2, 1)
plt.plot(range(1, len(episode_rewards)+1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Training Reward Curve')
plt.grid(True)

# 绘制平滑后的奖励曲线
plt.subplot(3, 2, 2)
window_size = min(100, len(episode_rewards) // 10)
if window_size > 0:
    smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size, len(episode_rewards)+1), smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Cumulative Reward')
    plt.title(f'Smoothed Training Reward Curve (Window Size: {window_size})')
    plt.grid(True)

# 绘制Critic损失曲线
plt.subplot(3, 2, 3)
plt.plot(range(1, len(critic_losses)+1), critic_losses)
plt.xlabel('Episode')
plt.ylabel('Critic Loss')
plt.title('Critic Loss Curve')
plt.grid(True)

# 绘制Actor损失曲线
plt.subplot(3, 2, 4)
plt.plot(range(1, len(actor_losses)+1), actor_losses)
plt.xlabel('Episode')
plt.ylabel('Actor Loss')
plt.title('Actor Loss Curve')
plt.grid(True)

# 绘制最后一个回合的能量水平轨迹
plt.subplot(3, 2, 5)
if len(energy_levels) > 0:
    plt.plot(range(len(energy_levels[-1])), energy_levels[-1])
    plt.axhline(y=MIN_ENERGY, color='r', linestyle='--', label=f'Min Energy ({MIN_ENERGY})')
    plt.axhline(y=MAX_ENERGY, color='g', linestyle='--', label=f'Max Energy ({MAX_ENERGY})')
    plt.xlabel('Time Step')
    plt.ylabel('Energy Level')
    plt.title('Energy Level Trajectory of Last Episode')
    plt.legend()
    plt.grid(True)

# 绘制能量水平随回合变化的热图
plt.subplot(3, 2, 6)
# 选择10个均匀分布的回合进行可视化
if len(energy_levels) > 0:
    num_episodes_to_show = min(10, len(energy_levels))
    episode_indices = np.linspace(0, len(energy_levels)-1, num_episodes_to_show, dtype=int)
    selected_trajectories = [energy_levels[i] for i in episode_indices]
    if selected_trajectories:
        max_length = max(len(traj) for traj in selected_trajectories)
        energy_matrix = np.zeros((len(episode_indices), max_length))

        for i, traj in enumerate(selected_trajectories):
            # 确保轨迹被正确填充到矩阵中
            energy_matrix[i, :len(traj)] = np.array(traj, dtype=float)

        plt.imshow(energy_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Energy Level')
        plt.xlabel('Time Step')
        plt.yticks(range(len(episode_indices)), [f'Episode {episode_indices[i]+1}' for i in range(len(episode_indices))])
        plt.title('Energy Level Trajectories Across Episodes')



plt.tight_layout()
plt.savefig(f'results/td3_training_curves_{timestamp}.png')
plt.savefig('results/td3_training_curves.png')  # 同时保存为标准名称

# 绘制成本分析图表
if len(local_controller.episode_costs) > 0:
    # 选择最后一个回合的成本组件数据
    last_episode_costs = local_controller.episode_costs[-1]
    
    if len(last_episode_costs) > 0:
        # 提取成本组件数据
        time_steps = range(len(last_episode_costs))
        current_prices = [c['current_price'] for c in last_episode_costs]
        predicted_prices = [c['predicted_price'] for c in last_episode_costs]
        price_diffs = [c['price_diff'] for c in last_episode_costs]
        actions = [c['action'] for c in last_episode_costs]
        energy_costs = [c['energy_cost'] for c in last_episode_costs]
        reg_terms = [c['reg_term'] for c in last_episode_costs]
        total_costs = [c['total_cost'] for c in last_episode_costs]
        
        # 创建分析图表
        plt.figure(figsize=(15, 15))
        
        # 价格图
        plt.subplot(3, 2, 1)
        plt.plot(time_steps, current_prices, label='Current Price')
        plt.plot(time_steps, predicted_prices, label='Predicted Price')
        plt.plot(time_steps, price_diffs, label='Price Difference', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title('Prices and Price Differences')
        plt.legend()
        plt.grid(True)
        
        # 动作图
        plt.subplot(3, 2, 2)
        plt.plot(time_steps, actions)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Action (Charge/Discharge Power)')
        plt.title('ESS Actions (Positive=Discharge, Negative=Charge)')
        plt.grid(True)
        
        # 成本组件图
        plt.subplot(3, 2, 3)
        plt.plot(time_steps, energy_costs, label='Energy Cost')
        plt.plot(time_steps, reg_terms, label='Regularization Term')
        plt.plot(time_steps, total_costs, label='Total Cost')
        plt.xlabel('Time Step')
        plt.ylabel('Cost')
        plt.title('Cost Components')
        plt.legend()
        plt.grid(True)
        
        # 价格与动作散点图
        plt.subplot(3, 2, 4)
        plt.scatter(price_diffs, actions, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Price Difference (Current - Predicted)')
        plt.ylabel('Action (Charge/Discharge Power)')
        plt.title('Relationship Between Price Difference and ESS Actions')
        plt.grid(True)
        
        # 能量水平与价格差值的关系
        episode_energy_level = energy_levels[-1][1:]  # 跳过初始能量水平
        min_len = min(len(episode_energy_level), len(price_diffs))
        
        plt.subplot(3, 2, 5)
        plt.scatter(price_diffs[:min_len], episode_energy_level[:min_len], alpha=0.6, c=range(min_len), cmap='viridis')
        plt.colorbar(label='Time Step')
        plt.xlabel('Price Difference')
        plt.ylabel('Energy Level')
        plt.title('Relationship Between Price Difference and Energy Level')
        plt.grid(True)
        
        # 累积成本和能量水平
        plt.subplot(3, 2, 6)
        ax1 = plt.gca()
        ax1.plot(time_steps, np.cumsum(total_costs), 'b-', label='Cumulative Cost')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Cumulative Cost', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(range(len(episode_energy_level[:min_len])), episode_energy_level[:min_len], 'r-', label='Energy Level')
        ax2.set_ylabel('Energy Level', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Cumulative Cost and Energy Level')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/cost_analysis_{timestamp}.png')
        plt.savefig('results/cost_analysis.png')  # 同时保存为标准名称
        print(f"Cost analysis plots saved to 'results/cost_analysis_{timestamp}.png' and 'results/cost_analysis.png'")

print("\nTraining completed!")
print(f"Training curves saved to 'results/td3_training_curves_{timestamp}.png' and 'results/td3_training_curves.png'") 