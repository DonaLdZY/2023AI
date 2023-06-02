import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import time
import random
import gym
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#----网络----
class QNet(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super().__init__()
        self.nw=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )
    def forward(self, x):
        x=torch.Tensor(np.array(x)).to(device)
        #print(type(x),x.shape)
        return self.nw(x)
#----经验回放池----
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=[]
        self.capacity=capacity
    def __len__(self):
        return len(self.buffer)
    def len(self):
        return len(self.buffer)
    def push(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    def sample(self,n):
        index = np.random.choice(len(self.buffer), n)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)
    def clean(self):
        self.buffer.clear()
#----超参数----
#网络的输入隐藏输出层维度
input_dim=4
hidden_dim=64
output_dim=2
#经验回放池容量
buffer_capacity=4096
gamma=0.99
#epsilon探索率
eps_max=0.02
eps_decay=0.9999
eps_min=0.01
#训练参数
batch_size=256
episodes=1000
max_step=500
update_target=100
#----Agent-----
class MyAgent:

    def __init__(self):
        self.eval_net=QNet(input_dim,hidden_dim,output_dim).to(device)
        self.target_net=QNet(input_dim,hidden_dim,output_dim).to(device) 
        self.buffer=ReplayBuffer(buffer_capacity)
        self.gamma=gamma

        #损失函数
        self.criterion = nn.MSELoss()
        #Adam 动态学习率(加快收敛速度)+惯性梯度(避免local minimal)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=0.0003, weight_decay=1e-5)
        
        self.eps_max=eps_max
        self.eps_decay=eps_decay
        self.eps_min=eps_min
        self.eps=eps_max
        self.learn_step=0
        self.batch_size=batch_size
        self.update_target=update_target

    def get_action(self, state, eval_mode=False):
        if (np.random.uniform()<=self.eps) and (not eval_mode):
            action=np.random.randint(0,2) #随机探索
            #print(action)
            return action
        else:
            action_values = self.eval_net(state).detach().clone().cpu().numpy()
            action=np.argmax(action_values)
            return action
    def store_transition(self, *transition):
        self.buffer.push(*transition)
    def learn(self):
        #更新eps
        self.eps=max(self.eps*self.eps_decay,self.eps_min)
        
        if self.learn_step%self.update_target==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step+=1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        actions = torch.LongTensor(actions).to(device)  # LongTensor to use gather latter
        dones = torch.FloatTensor(dones).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        # 计算评估网络对当前状态和动作的价值估计
        q_eval = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 计算目标网络对下一个状态的最大价值估计，不计算梯度
        q_next = self.target_net(next_states).max(1)[0].detach()
        # 计算目标价值，如果是终止状态，就只有奖励，否则还有折扣后的下一个状态的价值
        q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def load_model(self, file_name):
        self.eval_net.load_state_dict(torch.load(file_name + ".pth", map_location=device))
        self.target_net.load_state_dict(self.eval_net.state_dict())
    def save_model(self, file_name):
        torch.save(self.eval_net.state_dict(),file_name + ".pth")
    

if __name__ == '__main__':
    load_begin=True
    load_name = "hw10_21307303_liuzhuoyi"
    save_name = "hw10_21307303_liuzhuoyi_nxt"
    log_on=True
    log_name = "hw10_21307303_liuzhuoyi_log.txt"
    env = gym.make("CartPole-v1", render_mode="human")
    Rewards=[]
    #exec("from %s import MyAgent"%load_name)
    agent=MyAgent()
    if load_begin:
        agent.load_model(load_name)
    for t in range(episodes):
        state = env.reset(seed=int(time.time()))[0]
        #print(state)
        episode_reward = 0
        loss=0
        done = False
        step_cnt = 0
        while not done and step_cnt < max_step:
            step_cnt += 1 #步数+1
            env.render() 
            action = agent.get_action(state) #算动作
            next_state, reward, done, info, _ = env.step(action) #执行动作
            reward-=abs(next_state[0])/5
            agent.store_transition(state,action,reward,next_state,done) #装载轨迹
            #经验回放池装满了，就开学
            if agent.buffer.len() >= buffer_capacity:
                loss+=agent.learn()
                agent.save_model(save_name)
            #记录训练信息
            episode_reward += reward #回报
            #到下一个状态
            state = next_state
        Rewards.append(episode_reward)
        print(f"Episode: {t}, Reward: {episode_reward}, eps: {agent.eps}, loss: {loss}") 
        if (log_on):
            with open(log_name,'a+')as op:
                op.write(str(episode_reward)+" "+str(agent.eps)+" "+str(loss)+"\n")
    # 计算近10局的均值
    mean_rewards = [np.mean(Rewards[i-10:i]) for i in range(10, len(Rewards))]
    # 绘制曲线图
    plt.plot(range(10, len(Rewards)), mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward of Last 10 Episodes')
    plt.show()

