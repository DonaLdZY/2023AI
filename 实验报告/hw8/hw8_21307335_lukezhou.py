import numpy as np
from grid_env import MiniWorld
import copy as cp
LAMDA=0.9
R=-1
class DPAgent:
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

    def iteration(self, threshold=1e-3):
        # 初始化值函数和策略
        final=[env._xy_to_state(p) for p in env.ends]
        right=final[0]
        trap=final[1]
        block=[env._xy_to_state(p) for p in env.blocks]
        values = np.zeros([self.n_state])
        values[trap]=-10
        policy = np.full((self.n_state, self.n_action),1/4)
        action=[[0,1],[1,0],[0,-1],[-1,0]]
        P=np.zeros([self.n_state, self.n_action])
        for i in range(self.n_state):
            if(i in final):
                policy[i]=[0,0,0,0]
                continue
            time=4
            for j in range(self.n_action):
                (x,y)=env._state_to_xy(i)
                (x,y)=(x+action[j][0],y+action[j][1])
                if x>=0 and x<=5 and y<=5 and y>=0 and (x,y) not in env.blocks:
                    P[i][j]=(env._xy_to_state(x,y))
                else:
                    P[i][j]=-1000
                    policy[i][j]=0
                    time=time-1
            for j in range(self.n_action):
                if policy[i][j]!=0:
                    policy[i][j]=1/time
        while True:
            # 策略评估
            values = np.zeros([self.n_state])
            values[final[0]]=-10
            while True:
                delta = 0
                changed_values=cp.deepcopy(values)
                for s in range(self.n_state):
                    if s in final:
                        continue
                    changed_values[s]=R
                    for a in range(self.n_action):
                        u=P[s][a]
                        if u != -1000:
                            changed_values[s] =changed_values[s]+policy[s][a] * (LAMDA * values[int(u)])
                    delta = max(delta, abs(values[s]-changed_values[s]))
                values=changed_values
                if delta < threshold:
                    break
            env.show_values(values, sec=3)
            # 策略改进
            policy_stable = True
            for s in range(self.n_state):
                old_action =policy[s]
                action_values = np.zeros([self.n_action])
                maximum=-10000
                flag=[0,0,0,0]
                for a in range(self.n_action):
                    action_values[a] = (R + LAMDA * values[int(P[s][a])]) if P[s][a]!=-1000 else -2000
                    if maximum<action_values[a]:
                        maximum=action_values[a]
                        flag=[0,0,0,0]
                        flag[a]=1
                    elif maximum==action_values[a]:
                        flag[a]=1
                #选择最佳动作
                all_direction=sum(flag)
                best_action=np.zeros(self.n_action)
                for a in range(self.n_action):
                    best_action[a]=1/all_direction if flag[a] else 0
                if old_action.all() != best_action.all():
                    policy_stable = False
                policy[s] = best_action
            if policy_stable:
                break
        return values, policy


 
if __name__ == "__main__":
    env = MiniWorld()
    agent = DPAgent(env)

    values, policy = agent.iteration(threshold=1e-3)

    env.show_values(values, sec=3)  # 将values渲染成颜色格子, 显示3秒
    env.show_policy(policy)  # 在终端打印每个状态下的动作


