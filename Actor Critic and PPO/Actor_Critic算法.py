import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

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


class Actor_Critic():
    def __init__(self,state_dim,hidden_dim,action_dim,gamma):
        self.actor_net=Actor_net(state_dim,hidden_dim,action_dim)
        self.critic_net=Critic_net(state_dim,hidden_dim)
        self.gamma=gamma
        self.actor_optimizer=optim.Adam(self.actor_net.parameters(),lr=1e-4)
        self.critic_optimizer=optim.Adam(self.critic_net.parameters(),lr=5e-4)

    def select_action(self,state):
        state=torch.tensor(state,dtype=torch.float32)
        probs=self.actor_net(state)
        dist=torch.distributions.Categorical(probs)
        action=dist.sample()#取样，如tensor[3]
        return action.item(),dist.log_prob(action)

    def update(self,state,reward,next_state,done,log_prob):
        state=torch.tensor(state,dtype=torch.float32)
        next_state=torch.tensor(next_state,dtype=torch.float32)
        reward=torch.tensor(reward,dtype=torch.float32)

        value=self.critic_net(state)
        next_value=self.critic_net(next_state)
        done=int(done)
        td_target=reward+self.gamma*next_value*(1-done)
        advantages=td_target-value
        advantages.detach()
        critic_loss=advantages.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss=-log_prob*advantages.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

if __name__ == '__main__':
    env=gym.make('CartPole-v1')
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    gamma=0.98
    agent=Actor_Critic(state_dim,hidden_dim=128,action_dim=action_dim,gamma=gamma)
    reward_history = []
    for alt in range(2000):
        rewards=0
        state_primary,_=env.reset()
        while 1:
            act, log_prob = agent.select_action(state_primary)
            next_state,reward,teminate,tur,info=env.step(act)
            rewards+=reward
            done=teminate or tur
            agent.update(state_primary, reward, next_state, done, log_prob)
            state_primary = next_state
            if done:
                break
        reward_history.append(rewards)
        if alt%40==0 and alt!=0:
            print(f'第{alt}次迭代已完成，当前总损失为{rewards}')

    max_value=max(reward_history)
    min_value=min(reward_history)
    alt=[i for i in range(len(reward_history))]

    plt.figure(figsize=(18,16))
    plt.title('Actor_Critic算法')
    plt.grid(True)
    plt.ylabel('回报价值')
    plt.xlabel('迭代次数')
    plt.xlim(0,len(reward_history)+2)
    plt.ylim(min_value-5,max_value+5)
    plt.plot(alt,reward_history,color='r',lw=1.5)
    plt.show()
