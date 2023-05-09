
    
import numpy as np
from grid_env import MiniWorld


class DPAgent:
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

    #由状态state，进行action -> 返回[(next_state,p,reward) ...]
    def go(self,state,action): 
        x, y = env._state_to_xy(state)
        # 运动，并且边缘检测
        if action == 3:
            next_state = (max(x - 1, 0), y)
        elif action == 1:
            next_state = (min(x + 1, env.n_height - 1), y)
        elif action == 2:
            next_state = (x, max(y - 1, 0))
        elif action == 0:
            next_state = (x, min(y + 1, env.n_width - 1))
        else:
            raise ValueError('Invalid action')
        # 障碍检测
        if next_state in env.blocks:
            next_state = (x,y)
        nx=env._xy_to_state(next_state)
        reward=env.R[nx]
        return [(nx,1,reward)] #在本题中往一个方向行动100%有下个状态
    
    def iteration(self, threshold=1e-3):
        gamma=0.9
        values = np.zeros([self.n_state])
        policy = np.zeros([self.n_state, self.n_action])
        
        # 价值迭代
        count=0 #计数器
        maxc=self.n_state**2 #最大迭代次数
        while True:
            count+=1
            delta = 0
            values_new = np.zeros([self.n_state])

            for s in range(len(values)):
                if (env._state_to_xy(s) in env.blocks or env._state_to_xy(s) in env.ends):
                    continue
                # [self.go(s,a) for a in range(len(env.action_meaning))]一个 [行动，行动可能的下一个状态] * (概率、下一个状态、行动期望回报)
                # nx 某个行动对应的所有可能的下一个状态
                q_values=[ sum([ p*(reward+ gamma*values[next_state]) for (next_state,p,reward) in nx ]) for nx in [self.go(s,a) for a in range(len(env.action_meaning))] ] 
                values_new[s]=max(q_values)
                delta = max(delta, abs(values[s]-values_new[s])) #最大的迭代变化
            values=values_new #迭代完后更新总体
            if (delta<threshold or count>=maxc):
                break
        # 生成最优策略
        for s in range(len(values)):
            if (env._state_to_xy(s) in env.blocks or env._state_to_xy(s) in env.ends):
                continue
            q_values=[ sum([ p*(reward+ gamma*values[next_state]) for (next_state,p,reward) in nx ]) for nx in [self.go(s,a) for a in range(len(env.action_meaning))] ]
            
            maxq=max(q_values)
            t=1/sum([q_values[i]==maxq for i in range(4)]) #有多少个多少个取值最大的动作
            for i in range(len(env.action_meaning)):
                if (q_values[i]==maxq):
                    policy[s][i]=t;
            #policy[s][np.argmax(q_values)]=1;
        return values, policy


if __name__ == "__main__":
    env = MiniWorld()
    agent = DPAgent(env)

    values, policy = agent.iteration(threshold=1e-3)

    env.show_values(values, sec=3)  # 将values渲染成颜色格子, 显示3秒
    env.show_policy(policy)  # 在终端打印每个状态下的动作
