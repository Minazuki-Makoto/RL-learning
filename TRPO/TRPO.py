import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from copy import deepcopy

class Actor_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim,action_dim)
        self.fc_std=nn.Parameter(torch.tensor([-2]*action_dim,dtype=torch.float32))

    def forward(self,x):
        x=F.relu(self.fc1(x))
        mu=self.fc_mu(x)
        std=self.fc_std.clamp(-5,2)
        std = torch.exp(std).expand_as(mu)
        return mu,std

class Critic_net(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class TRPO_agent:
    def __init__(self,state_dim,hidden_dim,action_dim,act_bound,gamma,delta,alpha):
        self.actor_net=Actor_net(state_dim,hidden_dim,action_dim)
        self.critic_net=Critic_net(state_dim,hidden_dim)

        self.actor_optimizer=optim.Adam(self.actor_net.parameters(),lr=5e-5)
        self.critic_optimizer=optim.Adam(self.critic_net.parameters(),lr=1e-4)

        self.act_bound=act_bound
        self.gamma=gamma
        self.delta=delta
        self.alpha=alpha

    def choose_action(self,state):
        state=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        mu,std=self.actor_net(state)
        dist=Normal(mu,std)
        raw_action=dist.rsample()
        action=torch.tanh(raw_action)
        raw_prob=dist.log_prob(raw_action)
        true_action=self.act_bound*action
        log_prob = raw_prob - torch.log(1-action.pow(2)+1e-6).squeeze(0)
        true_action=true_action.squeeze(0).detach().numpy()
        return true_action,log_prob.sum(dim=-1,keepdim=True).detach()

    def GAE(self,value,next_value,reward,done,lamda=0.95):
        advantage=torch.zeros_like(reward)
        lenth=reward.shape[0]
        adv=0
        for i in range(lenth-1,-1,-1):
            delta=reward[i]+(1-done[i])*self.gamma*next_value[i]-value[i]
            adv=adv*self.gamma*lamda+delta
            advantage[i]=adv
        return advantage

    def hassin_matrix_vector_product(self,states,old_dists,vector):
        if not torch.is_tensor(states):
            states = torch.tensor(states, dtype=torch.float32)
        mu,std=self.actor_net(states)
        new_dists=Normal(mu,std)
        kl=torch.mean(torch.distributions.kl.kl_divergence(old_dists,new_dists))
        kl_grad=torch.autograd.grad(kl,self.actor_net.parameters(),create_graph=True)
        kl_grad_vector=torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product=torch.dot(kl_grad_vector,vector)
        grad2=torch.autograd.grad(kl_grad_vector_product,self.actor_net.parameters())
        grad2_vector=torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def solve(self,state,old_dists,grad):
        x=torch.zeros_like(grad)
        error=grad.clone()
        distance=grad.clone()
        dotr=torch.dot(error,error)
        for i in range(20):
            Hp=self.hassin_matrix_vector_product(state,old_dists,distance)
            alpha=dotr/(torch.dot(distance,Hp)+1e-8)
            x+=alpha*distance
            error-=alpha*Hp
            new_dotr=torch.dot(error,error)
            if new_dotr <1e-8:
                break
            beta=new_dotr/dotr
            distance=error+beta*distance
            dotr=new_dotr
        return x

    def seek(self,improvement,state,old_dists):

        old_parameters=torch.nn.utils.convert_parameters.parameters_to_vector(self.actor_net.parameters())
        for i in range(1,15):
            step_frac = self.alpha ** i
            new_step = step_frac * improvement
            new_parameters=old_parameters+new_step
            new_net=deepcopy(self.actor_net)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_parameters,new_net.parameters()
            )
            with torch.no_grad():
                mu, std = new_net(state)

                if torch.isnan(mu).any() or torch.isnan(std).any():
                    continue  # 直接尝试更小步长
                new_dist = Normal(mu, std)
                kl = torch.mean(
                    torch.distributions.kl.kl_divergence(old_dists, new_dist)
                )
            if kl < self.delta :
                return  new_parameters
        return old_parameters

    def actor_parameter_update(self,improvement,state,old_dists):
        new_params=self.seek(improvement,state,old_dists)
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_params, self.actor_net.parameters()
        )

    def update(self,state,action,reward,next_state,done,log_old_probs):

        state=torch.tensor(np.array(state),dtype=torch.float32)
        action=torch.tensor(np.array(action),dtype=torch.float32)
        reward=torch.tensor(np.array(reward),dtype=torch.float32).unsqueeze(1)
        next_state=torch.tensor(np.array(next_state),dtype=torch.float32)
        done=torch.tensor(np.array(done),dtype=torch.float32).unsqueeze(1)
        log_old_probs = torch.stack(log_old_probs).detach()

        with torch.no_grad():
            mu_old,std_old=self.actor_net(state)
        old_dist=Normal(mu_old,std_old)

        for _ in range(10):
            value = self.critic_net(state)
            next_value = self.critic_net(next_state)
            with torch.no_grad():
                td_target = self.gamma * (1 - done) * next_value + reward
            critic_loss=F.mse_loss(td_target,value)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        value = self.critic_net(state)
        with torch.no_grad():
            next_value = self.critic_net(next_state)
            td_target = self.gamma * (1 - done) * next_value + reward
        advatage = self.GAE(value, next_value, reward, done)
        advatage = (advatage - advatage.mean()) / (advatage.std() + 1e-8)
        advatage = advatage.detach()

        mu,std=self.actor_net(state)
        dist_new=Normal(mu,std)
        new_log_prob=[]
        for i in range(mu.shape[1]):
            raw_action=torch.atanh(action[:,i].clamp(-0.995,0.995))
            log_prob=dist_new.log_prob(raw_action)-torch.log(1-action[:,i].pow(2)+1e-6)
            new_log_prob.append(log_prob)
        new_log_prob=torch.stack(new_log_prob,dim=1)
        new_log_prob=new_log_prob.sum(dim=-1,keepdim=True)
        ratio=torch.exp(new_log_prob-log_old_probs)

        loss = -(ratio * advatage).mean()
        grads = torch.autograd.grad(loss, self.actor_net.parameters())
        grad = torch.cat([g.view(-1) for g in grads])
        x=self.solve(state,old_dist,grad)
        Hx=self.hassin_matrix_vector_product(state,old_dist,x)
        dot_x=torch.dot(x,Hx)
        improvement=torch.sqrt(self.delta*2/(dot_x+1e-8))*x

        self.actor_parameter_update(improvement,state,old_dist)





