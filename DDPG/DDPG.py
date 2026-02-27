import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli,Normal
import copy
from collections import  deque
import random

class Actor_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor_net,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc_mu=nn.Linear(hidden_dim,action_dim)
        self.fc_std=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu=self.fc_mu(x)
        std=F.softplus(self.fc_std(x)).expand_as(mu)
        return mu,std

class Critic_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Critic_net,self).__init__()
        self.fc1=nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.f3=nn.Linear(hidden_dim,1)

    def forward(self,s,a):
        x=torch.cat((s,a),dim=1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.f3(x)
        return x

class Buffer_Pool:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        samples=random.sample(self.buffer,batch_size)
        states, actions, reward, next_states, dones = zip(*samples)
        return states,actions,reward,next_states,dones

    def __len__(self):
        return len(self.buffer)

class ddpg_agent:
    def __init__(self,state_dim,hidden_dim,action_dim,gamma,noise_std,tau,capacity,batch_size):

        self.actor_net=Actor_net(state_dim,hidden_dim,action_dim)
        self.critic_net=Critic_net(state_dim,hidden_dim,action_dim)

        self.actor_optimizer=optim.Adam(self.actor_net.parameters(),lr=1e-4)
        self.critic_optimizer=optim.Adam(self.critic_net.parameters(),lr=2e-4)

        self.gamma=gamma
        self.noise_std=noise_std
        self.tau=tau

        self.target_actor_net=copy.deepcopy(self.actor_net)
        self.target_critic_net=copy.deepcopy(self.critic_net)

        self.buffer=Buffer_Pool(capacity)
        self.batch_size=batch_size

    def choose_action(self,state,flag=False):
        if not torch.is_tensor(state):
            state=torch.tensor(state,dtype=torch.float32)

        if flag==False:
            state=state.unsqueeze(0)
        mu,std=self.actor_net(state)
        actions=[]
        for i in range(mu.shape[1]):
            if i in [0,1]:
                p=torch.sigmoid(mu[:,i]+torch.normal(0, self.noise_std,mu[:,i].shape ))
                dist=Bernoulli(p)
                action=dist.sample()
            else:
                dist=Normal(mu[:,i],std[:,i])
                raw_action=dist.rsample()
                raw_action1=raw_action+torch.normal(0,self.noise_std,raw_action.shape)
                action=torch.tanh(raw_action1)

            actions.append(action)

        actions=torch.stack(actions,dim=1)

        if flag==False:
            actions=actions.detach().squeeze(0).cpu().numpy()

        return actions

    def choose_target_action(self,state):
        if not torch.is_tensor(state):
            state=torch.tensor(state,dtype=torch.float32)

        mu,std=self.target_actor_net(state)
        actions=[]

        for i in range(mu.shape[1]):
            if i in [0,1]:
                p=torch.sigmoid(mu[:,i])
                dist=Bernoulli(p)
                action=dist.sample()
            else:
                dist=Normal(mu[:,i],std[:,i])
                raw_action=dist.rsample()
                action=torch.tanh(raw_action)
            actions.append(action)
        actions=torch.stack(actions,dim=1)
        return actions
    def soft_update(self):
        for local_pama,target_pama in zip(self.actor_net.parameters(),self.target_actor_net.parameters()):
            target_pama.data.copy_(self.tau*local_pama+(1-self.tau)*target_pama)

        for clocal_pama,ctarget_pama in zip(self.critic_net.parameters(),self.target_critic_net.parameters()):
            ctarget_pama.data.copy_(clocal_pama*self.tau+(1-self.tau)*ctarget_pama)

    def update(self):
        if len(self.buffer)>3000:
            states, actions, reward, next_states, dones=self.buffer.sample(self.batch_size)
        else:
            return

        states=torch.tensor(np.array(states),dtype=torch.float32)
        actions=torch.tensor(np.array(actions),dtype=torch.float32)
        reward=torch.tensor(np.array(reward),dtype=torch.float32).view(-1,1)
        next_states=torch.tensor(np.array(next_states),dtype=torch.float32)
        dones=torch.tensor(np.array(dones),dtype=torch.float32).view(-1,1)

        for _ in range(10):
            value = self.critic_net(states, actions)
            with torch.no_grad():
                next_actions=self.choose_target_action(next_states)
                next_value=self.target_critic_net(next_states,next_actions)
                td_target=reward+self.gamma*(1-dones)*next_value
            critic_loss=F.mse_loss(value,td_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        new_actions=self.choose_action(states,True)
        q=self.critic_net(states,new_actions)
        policy_loss=-q.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()



