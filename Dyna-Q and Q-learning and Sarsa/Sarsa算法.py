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
                    x_next=min(3,max(0,x_after))
                    y_next=min(11,max(0,y_after))
                    if x_next==self.row-1 and  0<y_next<self.col-1:
                        s_list=[[i,j],-100,[x_next,y_next],True]
                        env[idx].append(s_list)
                        continue
                    reward=-1
                    s_list=[[i,j],reward,[x_next,y_next],False]
                    env[idx].append(s_list)
        return env

class CliffWalking():
    def __init__(self,Env,gamma,alpha,eplision):
        self.Env=Env
        self.gamma=gamma
        self.alpha=alpha
        self.eplision=eplision
        self.pi=[[0.25,0.25,0.25,0.25] for i in range(self.Env.row*self.Env.col)]
        self.Q_table=self.init()
        self.loss=[]
        self.way=[]
        self.action=[]
    def init(self):
        Q_table=[]
        for i in range(self.Env.row*self.Env.col):
            Q_table.append([0,0,0,0])
        return Q_table

    def choose_action(self,s):
        if np.random.rand()<self.eplision:
            return np.random.randint(4)
        else:
            return np.argmax(self.Q_table[s])

    def value_alter(self,s,s_next,a,a_next):
        past_position,reward,next_position,done=self.Env.env[s][a]
        error=self.alpha*(reward+self.gamma*self.Q_table[s_next][a_next]-self.Q_table[s][a])
        self.Q_table[s][a]+=error

    def run(self):
        start_place=(self.Env.row-1)*self.Env.col+0
        for alt in range(10000):
            s=start_place
            a=self.choose_action(s)
            all_loss=0
            policy=[]
            position=[]
            while 1:
                past_position, reward, new_position, done = self.Env.env[s][a]
                position.append(s)
                policy.append(a)
                x_next,y_next=new_position
                all_loss+=reward
                s_next=x_next*self.Env.col+y_next
                a_next=self.choose_action(s_next)
                self.value_alter(s,s_next,a,a_next)
                self.action
                if done:
                    break
                s=s_next
                a=a_next
            self.loss.append(all_loss)
            self.way.append(position)
            self.action.append(policy)

def department(s):
    y_position=s%col
    x_position=int(s/col)
    return [x_position,y_position]

if __name__=='__main__':
    row=4
    col=12
    Env=CliffEnv(4,12)
    cliffwalking=CliffWalking(Env,0.9,0.05,0.1)
    cliffwalking.run()
    alt=[i for i in range(10000)]
    r_min=min(cliffwalking.loss)
    r_max=max(cliffwalking.loss)
    argmax=np.argmax(cliffwalking.loss)
    policy_best=cliffwalking.action[argmax]
    way_best=cliffwalking.way[argmax]
    print('最优路径为：')
    for i in range(len(way_best)):
        print(f'{department(way_best[i])}-->')
    plt.figure(figsize=(18,14))
    plt.title('悬崖漫步r曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('reward')
    plt.grid(True)
    plt.xlim(0,5000)
    plt.ylim(r_min-1,r_max+10)
    plt.plot(alt,cliffwalking.loss,'b',lw=1.5)
    plt.show()