import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from TRPO import  TRPO_agent
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def curve_smooth(data,window=20):
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
    return smoothed

if __name__ == '__main__':
    env=gym.make('Pendulum-v1')
    hidden_dim=256
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    act_bound=env.action_space.high[0]
    alpha=0.9
    delta=0.01
    gamma=0.95
    history_reward=[]
    trpo_agent=TRPO_agent(state_dim,hidden_dim,action_dim,act_bound,gamma,delta,alpha)
    for alt in range(10000):
        state_history=[]
        action_history=[]
        reward_history=[]
        next_state_history=[]
        done_history=[]
        log_prob_history=[]
        state_primary,_=env.reset()
        rewards=0
        while 1 :
            action,log_prob=trpo_agent.choose_action(state_primary)
            next_state,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            state_history.append(state_primary)
            action_history.append(action)
            reward_history.append(reward)
            next_state_history.append(next_state)
            done_history.append(done)
            log_prob_history.append(log_prob)
            rewards+=reward
            if done == True:
                break
            state_primary=next_state

        history_reward.append(rewards)
        trpo_agent.update(state_history,action_history,reward_history,next_state_history,done_history,log_prob_history)
        if alt % 200 ==0:
            print(f'第{alt}次循环的回报为{rewards}')

    smooth_reward=curve_smooth(history_reward,50)
    alt=[i for i in range(len(history_reward))]
    plt.title('TRPO in Pendulum-v1')
    plt.xlabel('迭代次数')
    plt.ylabel('回报')
    plt.plot(alt,history_reward,color='lightblue',alpha=0.5,lw=1.5,label='原始回报')

    alt_smooth=[i for i in range(len(smooth_reward))]
    plt.plot(alt_smooth,smooth_reward,color='blue',lw=2,label='平滑处理')
    plt.legend()
    plt.show()
