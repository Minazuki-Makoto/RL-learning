import numpy as np
from data import price_t,T_t

class env():
    def __init__(self,t,price_t,T_t,T_primary,
                 trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
                 con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
                 HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
                 P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
                 ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,
                 PV_P_set):
        self.t=t
        self.price_t=price_t
        self.T_t=T_t
        self.T_primary=T_primary

        self.trans_MW_t_start=trans_MW_t_start
        self.trans_MW_t_end=trans_MW_t_end
        self.trans_alltime=trans_alltime
        self.MW_remain_time=trans_alltime
        self.DIS_remain_time=trans_alltime
        self.trans_WM_P_set=trans_WM_P_set
        self.trans_load_ws=trans_load_ws
        self.trans_DIS_t_start=trans_DIS_t_start
        self.trans_DIS_t_end=trans_DIS_t_end
        self.trans_DIS_P_set=trans_DIS_P_set

        self.con_load_light_ws=con_load_light_ws
        self.con_load_Humid_ws=con_load_Humid_ws
        self.con_load_light_min=con_load_light_min
        self.con_load_Humid_min=con_load_Humid_min
        self.con_load_Humid_set=con_load_Humid_set
        self.con_load_light_set=con_load_light_set

        self.HVAC_p_set=HVAC_p_set
        self.T_best=T_best
        self.HVAC_ws=HVAC_ws
        self.alpha=alpha #表示输出功率P与环境温度下降的关系
        self.beta=beta   #表示热交换系数
        self.error=error
        self.loss=loss

        self.EV_P_set=P_set
        self.energy_eta=energy_eta
        self.connect=0
        self.t_get=t_get
        self.t_leave=t_leave
        self.SOC=SOC
        self.SOC_primary=SOC_primary
        self.EV_remain_SOC=SOC_primary
        self.anxiety=anxiety
        self.damage=damage
        self.punish=punish

        self.ESS_P_set=ESS_P_set
        self.SOC_max=SOC_max
        self.SOC_min=SOC_min
        self.SOC_initial=SOC_initial
        self.ESS_remain_SOC=SOC_initial
        self.energy_convert=energy_convert

        self.PV_P_set=PV_P_set

    def set_load(self):
        return 0.30

    def transform_MW_load(self, t, action):
        hour = t % 24

        # 不在时间窗 → 强惩罚
        if not (self.trans_MW_t_start <= hour <= self.trans_MW_t_end):
            return -0.5 * action, 0, self.MW_remain_time

        # 任务已完成
        if self.MW_remain_time <= 0:
            return -0.5 * action, 0, 0

        if action == 1:
            self.MW_remain_time -= 1
            P_out = self.trans_WM_P_set
            reward = 0  # 完成任务奖励
        else:
            P_out = 0
            reward = 0 # 不执行任务惩罚

        # 任务没完成 → episode 末尾惩罚
        if hour == self.trans_MW_t_end and self.MW_remain_time > 0:
            reward = -10 * self.MW_remain_time

        return reward, P_out, self.MW_remain_time

    def DISWH(self, t, action):
        hour = t % 24
        #  不在允许时间窗 → 强惩罚 + 不允许运行
        if not (self.trans_DIS_t_start <= hour <= self.trans_DIS_t_end):
            return -0.5 * action, 0, self.DIS_remain_time
        # 任务已完成 → 禁止再运行（防止策略乱开）
        if self.DIS_remain_time <= 0:
            return -0.5* action, 0, 0
        # 执行任务
        if action == 1:
            self.DIS_remain_time -= 1
            P_out = self.trans_DIS_P_set  # 恒功率
            reward = 0  # 执行任务奖励
        else:
            P_out = 0
            reward = 0

        # 4️⃣ 时间窗结束但任务没完成 → 巨大惩罚（核心）
        if hour == self.trans_DIS_t_end and self.DIS_remain_time > 0:
            reward = - 10 * self.DIS_remain_time
        return reward, P_out, self.DIS_remain_time

    def controlable_load_Light(self,t,action):
        hour=t % 24
        if  6<= hour <=22:
            P_out = (self.con_load_light_set - self.con_load_light_min) * (1.0 + action) / 2.0 + self.con_load_light_min
            reward=-self.con_load_light_ws*(1.0-(P_out/self.con_load_light_set))**2.0
        else :
            reward=0
            P_out=0
        return reward,P_out

    def controlable_load_Humid(self,t,action):
        hour=t % 24
        if 13 <= hour <=22:
            P_out = (self.con_load_Humid_set - self.con_load_Humid_min) * (1.0 + action) / 2.0 + self.con_load_Humid_min
            reward=-self.con_load_Humid_ws*(1.0-(P_out/self.con_load_Humid_set))**2.0
        else:
            reward=0
            P_out=0
        return reward,P_out

    def HVAC(self,t,action):
        hour=t % 24
        T_out=self.T_t(hour)
        u=(action+1)/2.0
        P_out = u * self.HVAC_p_set
        self.T_primary=self.T_primary*self.beta+(1-self.beta)*(T_out-self.alpha*P_out)
        T_in=self.T_primary
        if 8 <= hour <= 23 and abs(self.T_primary-self.T_best)<=self.error:
            rewards = -self.HVAC_ws*(self.T_primary-self.T_best)**2.0
        elif 8 <= hour <= 23 and abs(self.T_primary-self.T_best)>self.error:
            rewards = -(self.HVAC_ws)*(self.T_primary-self.T_best)**2.0-self.loss
        else:
            rewards= -0.5*(P_out)
        return rewards,P_out,T_in

    def  EV(self,t,action):
        hour= t % 24
        if  0 <= hour <=8 or 20 <= hour <=23:
            self.connect=1
        else:
            self.connect=0
        connect=self.connect
        P_out = action * self.EV_P_set * connect
        if connect==1 :
            EV_add=P_out*self.energy_eta if P_out>0 else P_out/self.energy_eta
            self.EV_remain_SOC+=EV_add
        if connect ==1:
            if self.EV_remain_SOC < 10:
                extra_punish=-10-self.punish*(10-self.EV_remain_SOC)
            elif self.EV_remain_SOC>65:
                extra_punish=-10-self.punish*(self.EV_remain_SOC-65)
            else:
                extra_punish=0
        if connect==1 :
            if self.EV_remain_SOC <=self.SOC:
                dam=-abs(self.damage*P_out)
                anx=-self.anxiety*(self.SOC-self.EV_remain_SOC)
            else:
                anx=0
                dam=-abs(self.damage*self.EV_remain_SOC)
        if connect==0:
            anx=0
            extra_punish = 0
            dam=0
        rewards=anx+dam+extra_punish
        punishment=0
        if hour ==self.t_get:
            punishment= -(self.punish+5)*(self.SOC-self.EV_remain_SOC)   if self.EV_remain_SOC<self.SOC else 0
        rewards+=punishment
        EV_remain_SOC=self.EV_remain_SOC
        return rewards,P_out,connect,EV_remain_SOC

    def ESS(self,t,action):
        hour=t % 24
        P_out=action*self.ESS_P_set
        self.ESS_remain_SOC+=P_out*self.energy_convert
        if self.ESS_remain_SOC <=self.SOC_min :
            extra_punish=-self.punish*(self.SOC_min-self.ESS_remain_SOC)-5
        elif  self.ESS_remain_SOC>self.SOC_max:
            extra_punish=-self.punish*(self.ESS_remain_SOC-self.SOC_max)-5
        else:
            extra_punish=0
        dam=-abs(self.damage*P_out)
        rewards=dam+extra_punish
        remain_SOC=self.ESS_remain_SOC
        return rewards,P_out,remain_SOC

    def PV(self,t,action):
        hour = t % 24
        if 7 <= hour <= 19:
            P_out=-(action+1)/2*self.PV_P_set
        else:
            P_out=0
        return P_out

    def reset(self):
        # 时间：从12点开始
        self.t = 12

        # MW任务
        self.MW_remain_time= self.trans_alltime

        #DIS任务
        self.DIS_remain_time=self.trans_alltime

        # HVAC 初始温度,比仿真开始时间低四度
        self.T_primary = self.T_t(12)-4

        #初始状态EV并没有并网
        self.connect=0

        # EV 初始 SOC
        self.EV_remain_SOC = self.SOC_primary

        # ESS 初始 SOC
        self.ESS_remain_SOC = self.SOC_initial

        return self.get_state()


    def get_state(self):
       hour=self.t % 24

       state=np.array([hour,  #时间
       self.price_t(hour),   #电价
       self.T_primary,      #室内温度
       self.MW_remain_time, #MW剩余工作时间
       self.DIS_remain_time,#DIS剩余工作时间
       self.connect,        #EV是否链接
       self.EV_remain_SOC,  #EV剩余电量
       self.ESS_remain_SOC],dtype=np.float32) #ESS剩余电量

       return state

    def step(self,action):
        #动作空间
        action_MW= int(action[0])
        action_DIS=int(action[1])
        action_light=action[2]
        action_humid=action[3]
        action_HVAC=action[4]
        action_EV=action[5]
        action_ESS=action[6]
        action_PV=action[7]

        #各电器的属性
        P_set=self.set_load()
        MW_reward,MW_P_out,MW_t_remain=self.transform_MW_load(self.t,action_MW)
        DIS_reward,DIS_P_out,DIS_remain=self.DISWH(self.t,action_DIS)
        L_reward,L_P_out=self.controlable_load_Light(self.t,action_light)
        HM_reward,HM_P_out=self.controlable_load_Humid(self.t,action_humid)
        HVAC_reward,HVAC_P_out,HVAC_T_in=self.HVAC(self.t,action_HVAC)
        EV_reward, EV_P_out, EV_connect, EV_remain_SOC=self.EV(self.t,action_EV)
        ESS_reward, ESS_P_out, ESS_remain_SOC=self.ESS(self.t,action_ESS)
        PV_P_out=self.PV(self.t,action_PV)

        '''功率之和'''
        P_all=P_set+MW_P_out+DIS_P_out+L_P_out+HM_P_out+HVAC_P_out+EV_P_out+ESS_P_out+PV_P_out
        '''当前时刻电价'''
        price_now=self.price_t(self.t)
        '''回报'''
        cost=-price_now*P_all
        rewards=cost+MW_reward+DIS_reward+L_reward+HM_reward+HVAC_reward+EV_reward+ESS_reward
        '''下一个状态'''
        rewards=rewards/50
        self.t+=1
        next_state=self.get_state()
        if self.t >=36:
            done=True
        else:
            done=False

        return next_state,rewards,done