import matplotlib.pyplot as plt
import numpy as np
if __name__=="__main__":
    Rewards=[]
    eps=[]
    with open("实验报告\hw10\hw10_21307303_liuzhuoyi_log.txt","r")as ip:
        for lines in ip.readlines():
            if lines.rstrip()!="":
                line=lines.rstrip().split(' ')
                #print(line)
                Rewards.append(float(line[0]))
                eps.append(float(line[1]))
    # 计算近10局的均值
    mean_rewards = [np.mean(Rewards[i-10:i]) for i in range(10, len(Rewards))]
    # 绘制曲线图
    plt.plot(range(10, len(Rewards)), mean_rewards)
    plt.plot(range(0,len(eps)) , [i*500 for i in eps])
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward of Last 10 Episodes')
    plt.show()