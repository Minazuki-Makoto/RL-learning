import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Buffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def get_sample(self,Betch_size):
        samples=random.sample(self.buffer,Betch_size)
        state,action,reward,next_state,done=zip(*samples)
        return state,action,reward,next_state,done

    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class DQN():
    def __init__(self,env,state_dim,hidden_dim,action_dim,eplision,gamma):
        self.gamma=gamma
        self.eplision=eplision
        self.env=env
        self.action_dim=action_dim

        self.buffer=Buffer(capacity=8000)
        self.q_net=QNet(state_dim,hidden_dim,action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.target_q_net=QNet(state_dim,hidden_dim,action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.history_reward=[]
        self.state_history=[]

    def action_choose(self,state,alt):
        eplis = max(0.01, self.eplision * (0.995 ** alt))

        if np.random.rand()<eplis:
            return np.random.randint(self.action_dim)

        else:
            state=torch.tensor(state,dtype=torch.float32)
            return self.q_net(state).argmax().item()

    def train_step(self):
        if len(self.buffer)<1000:
            return
        else:
            state,action,reward,next_state,done=self.buffer.get_sample(Betch_size)

            state=torch.tensor(np.array(state),dtype=torch.float32)
            action=torch.tensor(action,dtype=torch.long)
            reward=torch.tensor(reward,dtype=torch.float32)
            next_state=torch.tensor(np.array(next_state),dtype=torch.float32)
            done=torch.tensor(done,dtype=torch.float32)

            q_value=self.q_net(state)
            q_sa=q_value.gather(1,action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_action=self.q_net(next_state).argmax(dim=1)
                q_next=self.target_q_net(next_state).gather(1,next_action.unsqueeze(1)).squeeze(1)
                target=reward+self.gamma*q_next*(1-done)
            loss=F.smooth_l1_loss(q_sa,target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(tau=0.001)
    def soft_update(self,tau=0.05):
        for t,s in zip(self.target_q_net.parameters(),self.q_net.parameters()):
            t.data.copy_(tau*s.data+(1-tau)*t.data)

    def update(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    def train_run(self):
        for alt in range(3000):
            rewards=0
            state_primary,_=self.env.reset()
            self.pass_by=[]
            while 1:
                self.pass_by.append(state_primary)
                action=self.action_choose(state_primary,alt)
                next_state, reward, terminated, truncated, info=self.env.step(action)
                done=terminated or truncated
                rewards+=reward
                self.buffer.push(state_primary,
                                 action,
                                 reward,
                                 next_state,
                                 done)
                self.train_step()
                state_primary=next_state
                if done:
                    break
            if alt % 50==0 and alt!=0:
                print(f'第{alt}次迭代已完成，当前总回报为{rewards}')
            self.history_reward.append(rewards)
            self.state_history.append(self.pass_by)

if '__main__' == __name__:
    Betch_size=100
    hidden_dim=128
    gamma=0.98
    eplision=0.1
    target_alter=100

    environment=gym.make('CartPole-v1')
    state_dim=environment.observation_space.shape[0]
    action_dim=environment.action_space.n

    np.random.seed(0)
    agent=DQN(environment,state_dim,hidden_dim,action_dim,eplision,gamma)
    agent.train_run()
    history_reward=agent.history_reward
    argmax_idx=np.argmax(history_reward)
    best_strategy=agent.state_history[argmax_idx]
    print('最优策略为：')
    for i in range(len(best_strategy)):
        print(f'{best_strategy[i]}-->')

    max_value=max(history_reward)
    min_value=min(history_reward)
    print(f'最大回报为{max_value}')
    alt=[i for i in range(len(history_reward))]

    plt.figure(figsize=(18,14))
    plt.title('DQN在车杆环境下的训练')
    plt.xlabel('迭代次数')
    plt.ylabel('价值回报')

    plt.xlim(0,len(history_reward)+1)
    plt.ylim(min_value-5,max_value+15)
    plt.grid(True)
    plt.plot(alt,history_reward,'r',lw=1.5)
    plt.show()
