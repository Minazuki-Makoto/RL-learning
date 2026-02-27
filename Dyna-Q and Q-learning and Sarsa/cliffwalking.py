import numpy as np

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
    def __init__(self,Env,theta,gamma):
        self.Env=Env
        self.V=[0]*self.Env.row*self.Env.col
        self.pi=[[0.25,0.25,0.25,0.25] for i in range(self.Env.row*self.Env.col)]
        self.theta=theta
        self.gamma=gamma

    def policy_envaluation(self):
        while True:
            max_diff=0
            new_value=[0]*self.Env.row*self.Env.col
            for s in range(self.Env.row*self.Env.col):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    past_position, reward, new_position, done = self.Env.env[s][a]
                    x_next,y_next=new_position
                    qsa+=(reward+1*self.gamma*self.V[x_next*self.Env.col+y_next]*(1-done))
                    qsa_list.append(self.pi[s][a]*qsa)
                new_value[s]=sum(qsa_list)
                max_diff=max(max_diff,abs(new_value[s]-self.V[s]))
            self.V=new_value
            if max_diff<self.theta:
                break

    def policy_improvement(self):
        for s in range(self.Env.row*self.Env.col):
            qsa_list=[]
            for a in range(4):
                qsa=0
                past_position, reward, new_position, done = self.Env.env[s][a]
                x_next, y_next = new_position
                qsa+=(reward+self.gamma*self.V[x_next*self.Env.col+y_next]*(1-done))
                qsa_list.append(qsa)
            maxq=max(qsa_list)
            cnt=qsa_list.count(maxq)
            self.pi[s]=[1/cnt if q==maxq else 0 for q in qsa_list]
        print('策略提升完成')
        return self.pi

    def run(self):
        while True:
            self.policy_envaluation()
            old_pi=self.pi.copy()
            new_pi=self.policy_improvement()
            if old_pi==new_pi:
                break

if __name__=='__main__':
    cliff=CliffEnv(4,12)
    cliff_Env=cliff.init()
    cliff_walking=CliffWalking(cliff,0.001,0.9)
    cliff_walking.run()
    print(cliff_walking.pi)
    print(cliff_walking.V)




