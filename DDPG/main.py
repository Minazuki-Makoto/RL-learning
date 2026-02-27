import matplotlib.pyplot as plt
import numpy as np
import torch
from env import  env
from data import T_t,price_t
import random
from DDPG import ddpg_agent
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_curve(data, window=20):
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


if __name__ == '__main__':
    t=12
    T_primary=T_t(12)-4
    trans_MW_t_start=15
    trans_MW_t_end=22
    trans_alltime=2
    trans_WM_P_set=0.4
    trans_load_ws=0.5
    trans_DIS_t_start=19
    trans_DIS_t_end=22
    trans_DIS_P_set=0.5
    con_load_light_ws=2.0
    con_load_light_min=0.3
    con_load_light_set=0.5
    con_load_Humid_ws=1.1
    con_load_Humid_min=0.2
    con_load_Humid_set=0.3
    HVAC_p_set=2.0
    T_best=24
    HVAC_ws=0.05
    alpha=0.05
    beta=0.85
    error=1.5
    loss=0.03
    P_set=7.5
    energy_eta=0.95
    t_get=8
    t_leave=22
    SOC=60
    SOC_primary=40
    anxiety=0.05
    damage=0.01
    punish=0.5
    ESS_P_set=10
    SOC_max=30
    SOC_min=15
    SOC_initial=21
    energy_convert=0.95
    PV_P_set=4
    state_dim=8
    hidden_dim=128
    action_dim=8
    eps=0.1
    gamma=0.96
    tau=0.005

    a=0.2
    batch_size=64
    buffer_size=10000
    history_rewards=[]

    noise_std=0.01
    SEED=40
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    environment=env(t,price_t,T_t,T_primary,
    trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
    con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
    HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
    P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
    ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,PV_P_set)

    Ddpg_agent=ddpg_agent(state_dim,hidden_dim,action_dim,gamma,noise_std,tau,buffer_size,batch_size)

    for alt in range(15000):
        state=environment.reset()
        rewards=0
        while True:
            action=Ddpg_agent.choose_action(state)
            next_state,reward,done=environment.step(action)
            Ddpg_agent.buffer.push(state,action,reward,next_state,done)
            rewards+=reward
            if done:
                break
            state=next_state

        Ddpg_agent.update()
        history_rewards.append(rewards)
        if alt % 200 ==0:
            print(f'第{alt}次更新的回报为{rewards*50}')

    history_rewards = [history_rewards[i] * 50 for i in range(len(history_rewards))]
    alt = [i for i in range(len(history_rewards))]
    smooth_rewards = smooth_curve(history_rewards, window=50)
    smooth_alt=[i for i in range(len(smooth_rewards))]
    plt.figure(figsize=(18, 16))
    plt.grid(True)
    plt.title('DDPG算法下的HEMS问题')
    plt.xlabel('迭代次数')
    plt.ylabel('价值回报')
    plt.xlim(0, len(history_rewards) + 1)
    plt.plot(alt, history_rewards, color='lightgreen', lw=1, alpha=0.5, label='原始回报')
    plt.plot(smooth_alt, smooth_rewards, color='green', lw=2, label='平滑回报')

    plt.legend()
    plt.show()
