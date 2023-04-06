import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time

class GeneticAlgTSP:
# ——数据部分——
# data_name数据集的名字
# city_name各个城市的名字
# city_location各个城市的坐标
# city_num城市的个数
# ——超参数——
# capacity 种群上限、
# extinction 物种大灭绝周期
# patience 迭代p次没有改善的话退出迭代
# vary_rate 变异率
# ——模型参数——
# gen 已经迭代的次数
# population 种群个体信息
# 	每个个体基因前city_num位表示基因信息，最后一位存放适应度
# best 已经找到的最短路长
#  ——日志文件-——
# log_open 是否记录日志
# log_path 日志存放目录
# checkpoint  绘制总体优化图时的检查点间隔
# loppoint 往日志文件写入记录的周期
# drawpoint 生成当前最优路线图的周期
# min_log 种群中最差的适应值的日志
# max_log 种群中最好的适应值的日志
# log_point 存日志时的时间点
    def __init__(self, tsp_filename, logs=False):
        #加载数据
        self.data_name=tsp_filename.split('\\')[-1].split('.')[0]
        ct=0 #找到数据开始的地方
        with open(tsp_filename,'r') as inputs:
            for i in inputs.readlines():
                ct+=1
                if (i.rstrip('\n')=="NODE_COORD_SECTION"):
                    break
        df = pd.read_csv(tsp_filename, sep=" ", skiprows=ct, header=None)
        city = np.array(df[0][0:len(df)-1])
        self.city_name = city.tolist()
        city_x = np.array(df[1][0:len(df)-1])
        city_y = np.array(df[2][0:len(df)-1])
        self.city_location = list(zip(city_x, city_y))
        self.city_num=len(self.city_name)
        #设置超参数
        self.capacity=64
        self.patience=int(pow(self.city_num,1.4))
        self.extinction=20
        self.vary_rate=0.5
        #建立种群
        self.population=[]
        for i in range(self.capacity-len(self.population)):
            self.population.append(self.generate_individual())
            self.population[-1].append(self.fitness(self.population[-1]))
        self.population.sort(key=lambda item:item[self.city_num])
        self.gen=0
        self.best=self.population[0][-1]
        #记录初始信息
        self.log_open=logs
        self.checkpoint=100
        self.drawpoint=1000
        self.logpoint=100
        self.log_path='Reports\\hw5\\data\\'
        if self.log_open:
            self.drawmap()
            with open(self.log_path+str(self.data_name)+'_log.txt',"w") as outputs:
                outputs.write(str(0)+' '+str(self.best)+' '+str([i+1 for i in self.population[0][0:self.city_num]])+'\n')
            self.log_point=[0]
            self.min_log=[self.population[-1][-1]]
            self.max_log=[self.population[0][-1]]
        print("ready")

    def fitness(self,idvd): #计算适应值，越低越好
        ans=0
        for i in range(self.city_num):
            ans+=math.sqrt((self.city_location[idvd[i]][0]-self.city_location[idvd[i-1]][0])*(self.city_location[idvd[i]][0]-self.city_location[idvd[i-1]][0])+(self.city_location[idvd[i]][1]-self.city_location[idvd[i-1]][1])*(self.city_location[idvd[i]][1]-self.city_location[idvd[i-1]][1]))
        return ans
    def generate_individual(self): #生成一个随机的个体
        lst=list(range(0,self.city_num))
        random.shuffle(lst)
        x=random.randint(0,self.city_num)
        return lst[x:self.city_num]+lst[0:x]
    def vary1(self,ex): #基因突变(片段反转)
        x=random.randint(0,self.city_num)
        y=random.randint(0,self.city_num)
        if (x>y):
            x=x^y
            y=x^y
            x=x^y
        return ex[0:x]+list(reversed(ex[x:y]))+ex[y:self.city_num]
    def vary2(self,ex): #基因突变(片段位移)
        x=random.randint(0,self.city_num)
        y=random.randint(0,self.city_num)
        if (x>y):
            (x,y)=(y,x)
        temp=ex[0:x]+ex[y:self.city_num]
        cut=ex[x:y]
        z=random.randint(0,len(temp))
        return temp[0:z]+cut+temp[z:len(temp)]
    
    def fusion(self,father,mother): #杂交育种
        x=random.randint(0,self.city_num)
        y=random.randint(0,self.city_num)
        if (x>y):
            (x,y)=(y,x)
        boy=father[0:x]+father[y:self.city_num]
        boyc=mother[x:y]
        girl=mother[0:x]+mother[y:self.city_num]
        girlc=father[x:y]
        for i in range(len(boy)):
            while (boy[i] in boyc):
                boy[i]=girlc[boyc.index(boy[i])]
        for i in range(len(girl)):
            while (girl[i] in girlc):
                girl[i]=boyc[girlc.index(girl[i])]    
        boys=boy[0:x]+boyc+boy[x:]
        girls=girl[0:x]+girlc+girl[x:]
        boys.append(self.fitness(boys))
        girls.append(self.fitness(girls))
        self.population.append(boys)
        self.population.append(girls)

    def iterate(self, num_iterations): #迭代
        begin=time.time()
        pc=0
        for _ in range(1,num_iterations):
            pc+=1
            #物种大灭绝
            if pc%self.extinction==0:
                self.population=self.population[0:self.capacity//6]
                for i in range(self.capacity-len(self.population)):
                    self.population.append(self.generate_individual())
                    self.population[-1].append(self.fitness(self.population[-1]))
            #随机两两杂交
            for i in range(len(self.population)//2):
                x=random.randint(0,self.capacity-1)
                if pc>5*self.extinction:
                    x=random.randint(0,min(2,self.capacity-1))
                y=random.randint(0,self.capacity-1)
                if x!=y:
                    self.fusion(self.population[x],self.population[y])
            #突变
            for item in range(len(self.population)): 
                if (random.random()<self.vary_rate):
                    self.population.append(self.vary1(self.population[item]))
                    self.population[-1].append(self.fitness(self.population[-1]))
                    self.population.append(self.vary2(self.population[item]))
                    self.population[-1].append(self.fitness(self.population[-1]))
            #优胜劣汰
            self.population.sort(key=lambda item:item[self.city_num]) 
            self.population=self.population[0:self.capacity]
            self.gen+=1
            #检测优化成功
            if (self.population[0][-1]<self.best):
                self.best=self.population[0][-1]
                pc=0
            if (pc>self.patience):
                break
            #日志记录
            self.updatelog()

        end=time.time()
        if self.log_open:
            self.updatelog(True)
            with open(self.log_path+str(self.data_name)+'_log.txt',"a+") as outputs:
                outputs.write('time cost : '+str(end-begin))
        print("Finishing at",self.gen,"th gen")
        return [self.population[1][-1],[i+1 for i in self.population[0][0:self.city_num]]]
    

    def updatelog(self,force=False):
        if not self.log_open:
            return
        if self.gen%self.drawpoint==0 or force:
            self.drawmap()
        if self.gen%self.checkpoint==0 or force:
            self.log_point.append(self.gen)
            self.min_log.append(self.population[-1][-1])
            self.max_log.append(self.population[0][-1])
            self.plot_diagram()
        if self.gen%self.logpoint==0 or force:
            with open(self.log_path+str(self.data_name)+'_log.txt',"a+") as outputs:
                outputs.write(str(self.gen)+': '+str(self.best)+' '+str([i+1 for i in self.population[0][0:self.city_num]])+'\n')
    def drawmap(self): #绘制当前路线
        plt.clf()
        xs=[self.city_location[i][0] for i in self.population[0][0:len(self.population[0])-1]]
        ys=[self.city_location[i][1] for i in self.population[0][0:len(self.population[0])-1]]
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs,ys,color='b')
        plt.scatter(xs,ys,color='r',s=10)
        plt.title('gen '+str(self.gen)+'\nShortest Path: '+str(self.best))
        plt.savefig(self.log_path+str(self.data_name)+'_gen'+str(self.gen)+'.jpg')
    def plot_diagram(self): #绘制数据迭代表
        plt.clf()
        # for i in range(len(self.log_point)):
        #     if self.max_log[i]!=self.min_log[i]:
        #         plt.plot([self.log_point[i], self.log_point[i]], [self.max_log[i],self.min_log[i]], color='b')
        for i in range(1,len(self.log_point)):
            plt.plot([self.log_point[i-1], self.log_point[i]], [self.max_log[i-1],self.max_log[i]], color='r')
        plt.title('gen '+str(self.gen)+'\nShortest Path: '+str(self.best))
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.savefig(self.log_path+str(self.data_name)+'_overall'+'.jpg')

def set_random():
    t = int( time.time() * 1000.0 )
    random.seed( ((t & 0xff000000) >> 24) +((t & 0x00ff0000) >>  8) +((t & 0x0000ff00) <<  8) +((t & 0x000000ff) << 24)   )
if __name__ == "__main__":
    set_random()
    T = 300000
    # tsp = GeneticAlgTSP('实验课\\Homework\\hw5\\wi29.tsp',logs=True)
    # tour = tsp.iterate(T)  # 对算法迭代T次
    # print(tour)  # 打印路径(以列表的形式)
    # tsp = GeneticAlgTSP('实验课\\Homework\\hw5\\dj38.tsp',logs=True)
    # tour = tsp.iterate(T)  # 对算法迭代T次
    # print(tour)  # 打印路径(以列表的形式)
    tsp = GeneticAlgTSP('实验课\\Homework\\hw5\\qa194.tsp',logs=False)
    tour = tsp.iterate(T)  # 对算法迭代T次
    print(tour)  # 打印路径(以列表的形式)

