import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from random import sample
from collections import deque
class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim,out_dim, action_bound):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim,out_dim)
        self.fc_std = nn.Parameter(torch.tensor([-1]*out_dim,dtype=torch.float32))
        self.action_bound=action_bound

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        mu=self.fc_mu(x)
        std=self.fc_std
        std=torch.sigmoid(std)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        action_out = action * self.action_bound
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return action_out,log_prob

class QNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc_out=nn.Linear(hidden_dim,1)

    def forward(self,state,action):
        x=torch.cat((state,action),dim=1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc_out(x)

class Buffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def get_sample(self,batch_size):
        samples=sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*samples)
        return state,action,reward,next_state,done

    def __len__(self):
        return len(self.buffer)

class SAC_agent():
    def __init__(self, state_dim, action_dim, hidden_dim,gamma,tau,alpha,action_bound,batch_size):
        self.actor_net=ActorNet(state_dim,hidden_dim,action_dim,action_bound)

        self.Q_net1=QNet(state_dim,hidden_dim,action_dim)
        self.Q_net2=QNet(state_dim,hidden_dim,action_dim)

        self.target_Q_net1=copy.deepcopy(self.Q_net1)
        self.target_Q_net2=copy.deepcopy(self.Q_net2)
        self.gamma=gamma
        self.tau = tau
        self.alpha=alpha
        self.action_bound=action_bound
        self.batch_size=batch_size

        self.actor_optimizer=optim.Adam(self.actor_net.parameters(),lr=4e-4)
        self.Q_net1_optimizer=optim.Adam(self.Q_net1.parameters(),lr=4e-3)
        self.Q_net2_optimizer=optim.Adam(self.Q_net2.parameters(),lr=4e-3)

        self.buffer=Buffer(8000)
    def choose_action(self,state_primary):
        state_primary=torch.tensor(state_primary,dtype=torch.float32)
        with torch.no_grad():
            action,_ = self.actor_net(state_primary)
        return action.cpu().numpy()

    def compute_Qtarget(self,reward,Q1_next_value,Q2_next_value,next_log_probs,done):
        Q_next_value = torch.min(Q1_next_value, Q2_next_value)
        Q_target=reward+(1-done)*self.gamma*(Q_next_value-self.alpha*next_log_probs)
        return Q_target

    def soft_update(self):
        for param, target_param in zip(self.Q_net1.parameters(), self.target_Q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.Q_net2.parameters(), self.target_Q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self):
        if len(self.buffer)>3000:
            state_history,action_history,reward_history,next_state_history,done_history=self.buffer.get_sample(self.batch_size)
        else :
            return
        state_history=torch.tensor(np.array(state_history),dtype=torch.float32)
        action_history=torch.tensor(np.array(action_history),dtype=torch.float32)
        next_state_history=torch.tensor( np.array(next_state_history),dtype=torch.float32)
        reward_history=torch.tensor(np.array(reward_history),dtype=torch.float32).view(-1,1)
        done_history=torch.tensor(np.array(done_history),dtype=torch.float32).view(-1,1)

        Q1_value=self.Q_net1(state_history,action_history)
        Q2_value=self.Q_net2(state_history,action_history)

        with torch.no_grad():
            next_action, next_log_probs = self.actor_net(next_state_history)
            Q1_next_value = self.target_Q_net1(next_state_history, next_action)
            Q2_next_value = self.target_Q_net2(next_state_history, next_action)
        Q_target=self.compute_Qtarget(reward_history,Q1_next_value,Q2_next_value,next_log_probs,done_history)
        Q_target=Q_target.detach()
        Q1_loss=F.mse_loss(Q_target,Q1_value)
        Q2_loss=F.mse_loss(Q_target,Q2_value)

        self.Q_net1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q_net1_optimizer.step()

        self.Q_net2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q_net2_optimizer.step()

        '''V=E(Q-a*log(p))'''
        action,log_prob=self.actor_net(state_history)
        q1=self.Q_net1(state_history,action)
        q2=self.Q_net2(state_history,action)
        q_new=torch.min(q1,q2)
        actor_loss=(self.alpha*log_prob-q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()




