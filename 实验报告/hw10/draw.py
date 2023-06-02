import matplotlib.pyplot as plt
import numpy as np
import torch
if __name__=="__main__":
    Rewards=[]
    eps=[]
    loss=[]
    with open("hw10_21307303_liuzhuoyi_log.txt","r")as ip:
        for lines in ip.readlines():
            if lines.rstrip()!="":
                line=lines.rstrip().split(' ')
                #print(line)
                Rewards.append(float(line[0]))
                eps.append(float(line[1]))
                if line[2]=='0':
                    loss.append(0.0)
                else:
                    loss.append(float(line[2].split(',')[0][7::]))
    # 计算近10局的均值
    mean_rewards = [np.mean(Rewards[i-10:i]) for i in range(10, len(Rewards))]
    # 绘制曲线图
    fig,ax1 = plt.subplots()
    ax1.plot(range(10, len(Rewards)), mean_rewards,label="line 1")
    ax2 = ax1.twinx()
    ax2.plot(range(0,len(eps)) , loss,label="line 2",color="r")
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward of Last 10 Episodes')
    plt.show()