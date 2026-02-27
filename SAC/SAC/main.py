import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from SAC import SAC_agent
import random
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_curve(data, window=20):
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed

if __name__=='__main__':
    env=gym.make('Pendulum-v1')
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    action_bound=env.action_space.high[0]

    hidden_dim=128
    gamma=0.95
    tau=0.005
    alpha=0.1
    batch_size=128

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    sac_agent = SAC_agent(state_dim,action_dim,hidden_dim,gamma,tau,alpha,action_bound,batch_size)
    reward_all=[]
    for alt in range(8000):
        state_primary, info=env.reset()
        rewards=0
        step=0
        while 1:
            step+=1
            action=sac_agent.choose_action(state_primary)
            next_state, reward, terminated, truncated, info=env.step(action)
            done = terminated or truncated
            rewards+=reward
            sac_agent.buffer.push(state_primary,action,reward,next_state,done)
            if done==True:
                break
            state_primary=next_state
        for _ in range(2):
            sac_agent.update()
            sac_agent.soft_update()
        reward_all.append(rewards)
        if alt % 200 ==0:
            print(f'第{alt}次回报的回报为{rewards}')
    reward_all=smooth_curve(reward_all,50)
    alt=[i for i in range(len(reward_all))]
    plt.figure(figsize=(18,16))
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('回报')
    plt.title('SAC在Pendulum-v1中的应用')
    plt.plot(alt,reward_all,color='blue',lw=1.5)
    plt.show()
