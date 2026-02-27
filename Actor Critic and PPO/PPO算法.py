import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Actor_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        probs=F.softmax(x,dim=-1)
        return probs

class Critic_net(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPOBuffer():
    def __init__(self,capacity=1000):
        self.buffer=deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done,old_probs):
        self.buffer.append([state,action,reward,next_state,done,old_probs])

    def sample(self,batch_size):
        state,action,reward,next_state,done,old_probs=zip(*self.buffer)
        return np.array(state),action,reward,np.array(next_state),done,old_probs

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class PPO_Agent():
    def __init__(self,state_dim,hidden_dim,action_dim,eps,gamma=0.99):
        self.gamma=gamma
        self.eps=eps
        self.buffer=PPOBuffer(capacity=1000)

        self.actor_net=Actor_net(state_dim,hidden_dim,action_dim)
        self.critic_net=Critic_net(state_dim,hidden_dim)

        self.actor_net_optimizer=optim.Adam(self.actor_net.parameters(),lr=4e-4)
        self.critic_net_optimizer=optim.Adam(self.critic_net.parameters(),lr=1e-3)

    def choose_action(self,state):
        state=torch.tensor(state,dtype=torch.float32)
        prob=self.actor_net(state)
        dist=torch.distributions.Categorical(prob)
        action=dist.sample()
        return action.item(),dist.log_prob(action).detach()

    def update(self):
        state,action,reward,next_state,done,old_probs=self.buffer.sample(len(self.buffer))
        state=torch.tensor(state,dtype=torch.float32)
        reward=torch.tensor(reward,dtype=torch.float32)
        next_state=torch.tensor(next_state,dtype=torch.float32)
        done=torch.tensor(done,dtype=torch.float32)
        action=torch.tensor(action).unsqueeze(1)
        old_probs=torch.stack(old_probs).detach()
        for _ in range(10):
            value = self.critic_net(state).squeeze()
            next_value = self.critic_net(next_state).squeeze()
            td_target = reward + self.gamma * next_value * (1 - done)
            critic_loss = F.mse_loss(value, td_target.detach())

            self.critic_net_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_net_optimizer.step()

            advantage = reward + self.gamma * next_value * (1 - done) - value
            advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantages = advantages.detach()
            prob=self.actor_net(state).gather(1,action).squeeze()
            log_prob_a=torch.log(prob)
            ratio=torch.exp(log_prob_a-old_probs)
            error1=ratio*advantages
            erro2=torch.clamp(ratio,1-self.eps,1+self.eps)*advantages
            actor_loss=-torch.min(error1,erro2).mean()

            self.actor_net_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_net_optimizer.step()


if __name__ == '__main__':
    env=gym.make('CartPole-v1')
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    reward_history=[]
    agent=PPO_Agent(state_dim,hidden_dim=64,action_dim=action_dim,eps=0.1,gamma=0.99)
    for alt in range(2000):
        state_primary,_=env.reset()
        rewards=0
        agent.buffer.clear()
        while 1:
            action,log_prob=agent.choose_action(state_primary)
            next_state,reward,terminate,turncate,info=env.step(action)
            rewards+=reward
            done=terminate or turncate
            agent.buffer.push(state_primary,action,reward,next_state,done,log_prob)
            state_primary=next_state
            if done:
                break
        for _ in range(10):
            agent.update()
        reward_history.append(rewards)
        if alt %50 ==0 and alt !=0:
            print(f'第{alt}次迭代后的回报为{rewards}')

    max_value=max(reward_history)
    min_value=min(reward_history)
    alt=[i for i in range(len(reward_history))]

    plt.figure(figsize=(18,16))
    plt.title('PPO算法')
    plt.grid(True)
    plt.ylabel('回报价值')
    plt.xlabel('迭代次数')
    plt.xlim(0,len(reward_history)+2)
    plt.ylim(min_value-5,max_value+5)
    plt.plot(alt,reward_history,color='b',lw=1.5)
    plt.show()










