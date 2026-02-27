import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(0)

class CliffEnv():
    def __init__(self,row,col):
        self.row=row
        self.col=col
        self.env=self.init()


    def move(self,s_now,action):
        x_now=s_now[0]
        y_now=s_now[1]
        x_after=s_now[0]+action[0]
        y_after=s_now[1]+action[1]
        return x_after,y_after

    def init(self):
            #每一个位置由四个元素组成（对应四个动作）
        env=[[]for i in range(self.row*self.col)]
        choice=[[0,1],[1,0],[0,-1],[-1,0]]
        for i in range(self.row):
            for j in range(self.col):
                for k in range(4):
                    s_list=[]
                    idx=i*self.col+j
                    s_now=[i,j]
                    if i==self.row-1 and 0<j<=self.col-1:#处于终点或者悬崖时无法进行移动
                        s_list=[[i,j],0,[i,j],True]
                        env[idx].append(s_list)
                        continue
                    action = choice[k]
                    x_after,y_after=self.move(s_now,action)
                    x_next=min(self.row-1,max(0,x_after))
                    y_next=min(self.col-1,max(0,y_after))
                    if x_next==self.row-1 and  0<y_next<self.col-1:
                        s_list=[[i,j],-100,[x_next,y_next],True]
                        env[idx].append(s_list)
                        continue
                    reward=-1
                    s_list=[[i,j],reward,[x_next,y_next],False]
                    env[idx].append(s_list)
        return env

class CliffWalking():
    def __init__(self,Env,gamma,eplision,alpha,N):
        self.Env=Env
        self.gamma=gamma
        self.eplision=eplision
        self.history_reward=[]
        self.Q_table=self.init()
        self.best_strategy=[]
        self.alpha=alpha
        self.dic={}
        self.keys=[]
        self.N=N

    def init(self):
        table=[]
        for i in range(self.Env.row*self.Env.col):
            table.append([0]*4)
        return table

    def choose_action(self,s,cnt):
        eps = max(0.001, self.eplision / cnt)
        if np.random.rand()<eps:
            return np.random.randint(4)
        else:
            return np.argmax(self.Q_table[s])

    def Q_table_valuealtered(self,s,a,s_next):
        past_position,reward,next_position,done=self.Env.env[s][a]
        next_maxvalue=np.max(self.Q_table[s_next])
        err=(reward+self.gamma*next_maxvalue-self.Q_table[s][a])*self.alpha
        self.Q_table[s][a]+=err

    def Q_learning_run(self):
        cnt=1
        for alt in range(2000):
            s_primary = (self.Env.row - 1) * self.Env.col
            a = self.choose_action(s_primary, cnt)
            rewards=0
            place=[]
            while 1:
                past_position,reward,next_position,done=self.Env.env[s_primary][a]
                place.append(past_position)
                rewards+=reward
                x_next,y_next=next_position
                s_next=x_next*self.Env.col+y_next
                key = (s_primary, a)
                if key not in self.dic:
                    self.keys.append(key)
                self.dic[key] = [reward, s_next]
                self.Q_table_valuealtered(s_primary, a, s_next)
                if done==True:
                    break
                else:
                    s_primary=s_next
                    a=self.choose_action(s_primary,cnt)
            if alt > 5:
                for i in range(self.N):
                    idx = np.random.randint(len(self.keys))
                    random_key = self.keys[idx]
                    s_primary, a = random_key
                    reward, s_next = self.dic[random_key]
                    self.Q_table_valuealtered(s_primary, a, s_next)
            self.history_reward.append(rewards)
            self.best_strategy.append(place)
            cnt+=1


if __name__ == '__main__':
    cliffenv=CliffEnv(4,12)
    print('悬崖环境已搭建完成')
    cliff_walking=CliffWalking(cliffenv,0.9,0.1,0.03,100)
    cliff_walking.Q_learning_run()
    history_value=cliff_walking.history_reward
    idx=np.argmax(history_value)
    best_way=cliff_walking.best_strategy[idx]
    print('最优策略为：')
    for i in range(len(best_way)):
        print(f'{best_way[i]}-->')

    max_value=np.max(history_value)
    min_value=np.min(history_value)
    alt=[i for i in range(2000)]
    plt.figure(figsize=(18,14))
    plt.xlim(0,1000)
    plt.ylim(min_value-1,max_value+10)
    plt.xlabel('迭代次数')
    plt.ylabel('价值回报')
    plt.title('悬崖漫步环境下的Q——learning学习曲线')
    plt.plot(alt,history_value,lw=1.5,color='b')
    plt.show()