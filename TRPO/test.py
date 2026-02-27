import torch
from torch.distributions import Normal
import numpy as np
mu=1
std=0.1
dist=Normal(mu,std)
action=dist.sample()
print(action)
log_prob=dist.log_prob(action)
x=torch.tensor([1,2,3])
print(x.unsqueeze(0))