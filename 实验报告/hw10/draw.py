import matplotlib.pyplot as plt
import numpy as np
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
    mean_rewards = [np.mean(Rewards[max(0,i-10):i]) for i in range(0, len(Rewards))]

    # 绘制曲线图
    fig,ax1 = plt.subplots()
    color1='b'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score',color=color1)

    color2='r'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss',color=color2)
    
    ax2.plot(range(0,len(loss)) ,loss , label="Loss" ,color=color2)
    ax1.plot(range(len(Rewards)), mean_rewards , label="Score" ,color=color1)

    plt.title('Reward & Loss')
    plt.show()