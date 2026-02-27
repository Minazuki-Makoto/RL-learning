import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(2,4)

    def forward(self,x):
        return self.fc(x)

model=QNet()
optimizer=optim.SGD(model.parameters(),lr=0.01)
criterion=nn.MSELoss()

state=torch.tensor([1.0,2.0])
action=1
reward=-1
next_state=torch.tensor([1.0,3.0])
gamma=0.9

q_values=model(state)
q_sa=q_values[action]
with torch.no_grad():
    next_q=model(next_state)
    target=reward+gamma*torch.max(next_q)

loss=criterion(q_sa,target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss)