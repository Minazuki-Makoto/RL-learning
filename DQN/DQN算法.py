import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer=deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(state_dimision,hidden_dimision)
        self.fc2=nn.Linear(hidden_dimision,output_dimision)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class CliffEnv():
    def __init__(self,row,col):
        self.row=row
        self.col=col
        self.s=self.reset()

    def move(self,place,act):
        x_next=place[0]+act[0]
        y_next=place[1]+act[1]
        return [x_next,y_next]

    def reset(self):
        primary_state=[self.row-1,0]
        return primary_state

    def step(self, action_idx):
        act=Action[action_idx]
        x,y=self.s
        x_next=min(self.row-1,max(0,x+act[0]))
        y_next=min(self.col-1,max(0,y+act[1]))
        if x_next == self.row-1 and 0<y_next<self.col-1:
            reward=-100
            done=False
            self.s=self.reset()
        elif x_next == self.row-1 and y_next == self.col-1:
            reward=0
            done=True
            self.s=self.reset()
        else:
            reward=-1
            done=False
            self.s=[x_next,y_next]
        return self.s,reward,done

class DQN():
    def __init__(self,Env,eplision,alpha,gamma):
        self.Env=Env
        self.eplision=eplision
        self.alpha=alpha
        self.gamma=gamma
        self.buffer=ReplayBuffer(capacity)
        self.hiestory_reward=[]
        self.q_net=QNet()
        self.target_q_net=QNet()
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.cnt=0
        self.optimizer=optim.Adam(self.q_net.parameters(),lr=LR)

    def action_choose(self,state):
        eplis=max(0.01,self.eplision*(0.96**self.cnt))

        if np.random.rand()<eplis:
            return np.random.randint(len(Action))

        else:
            actions=self.q_net(state)
            return actions.argmax().item()

    def train_step(self):
        if len(self.buffer)<batch_size:
            return
        state,action,reward,next_state,done=self.buffer.sample(batch_size)
        state=torch.tensor(state,dtype=torch.float32)
        next_state=torch.tensor(next_state,dtype=torch.float32)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float32)
        done=torch.tensor(done,dtype=torch.float32)

        q_value=self.q_net(state)
        q_sa=q_value.gather(1,action.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q=self.target_q_net(next_state).max(1)[0]
            target=reward+self.gamma*next_q*(1-done)
        loss=F.mse_loss(q_sa,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_q_net_refresh(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    def train(self):
        for episode in range(2000):
            self.cnt+=1
            state=self.Env.reset()
            state=torch.tensor(state,dtype=torch.float32)
            total_reward=0
            while 1:
                action_idx=self.action_choose(state)
                next_state,reward,done=self.Env.step(action_idx)
                self.buffer.push(
                    state.tolist(),
                    action_idx,
                    reward,
                    next_state,
                    done)
                next_state=torch.tensor(next_state,dtype=torch.float32)
                self.train_step()
                total_reward+=reward
                if done:
                    break
            self.hiestory_reward.append(total_reward)
            if episode%10==0:
                print(f'在第{episode+1}次迭代后的总价值回报为{total_reward}')
            if episode%target_freq==0:
                self.target_q_net_refresh()

if '__main__' == __name__:
    state_dimision=2
    hidden_dimision=4
    output_dimision=4
    Action = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    capacity=5000
    batch_size=64
    LR=0.001
    alpha=0.03
    gamma=0.9
    eplision=0.5
    target_freq=100
    env=CliffEnv(4,12)
    policy=DQN(env,eplision,alpha,gamma)
    policy.train()
    r_min=min(policy.hiestory_reward)
    r_max=max(policy.hiestory_reward)
    alt=[i for i in range(len(policy.hiestory_reward))]

    plt.figure(figsize=(18,14))
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('价值回报')
    plt.xlim([0,2000])
    plt.ylim(r_min,r_max+10)
    plt.plot(alt,policy.hiestory_reward,label='hiestory',color='blue',lw=1.5)
    plt.show()