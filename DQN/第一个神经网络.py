import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1=nn.Linear(in_put,hidden)
        self.fc2=nn.Linear(hidden,out_put)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == '__main__':
    in_put,hidden,out_put=3,2,4
    model=QNet()
    state=torch.tensor([1.0,2.0,4.0])
    value=model.forward(state)
    print(value.max())