import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import os
class GeneticAlgTSP:
    #---data---
    #data_name
    #model_path
    #log_path
    #city_name
    #city_location
    #city_num
    #---hyper parameter---
    #capacity
    #max_gen
    #vary_rate
    #patience
    #---parameter--
    #self.gen
    #self.population
    #self.best
    def __init__(self, tsp_filename):
        #load date
        self.data_name=tsp_filename.split('\\')[-1].split('.')[0]
        self.model_path='\Genetic_model\\'+self.data_name+'.model'
        self.log_path='\Genetic_model\\'+self.data_name+'.txt'
        ct=0 #找到数据开始的地方
        with open(tsp_filename,'r') as inputs:
            for i in inputs.readlines():
                ct+=1
                if (i.rstrip('\n')=="NODE_COORD_SECTION"):
                    break
        df = pd.read_csv(tsp_filename, sep=" ", skiprows=ct, header=None)
        city = np.array(df[0][0:len(df)-2])
        self.city_name = city.tolist()
        city_x = np.array(df[1][0:len(df)-2])
        city_y = np.array(df[2][0:len(df)-2])
        self.city_location = list(zip(city_x, city_y))
        self.city_num=len(self.city_name)
        #set hyper parameters
        self.capacity=min(max(64,self.city_num),1024)
        self.epoch=int(self.city_num*math.sqrt(self.city_num))
        self.vary_rate=0.5
        self.patience=1000000
        #load model
        self.population=[]
        if os.path.exists(self.model_path):
            print("start load model")
            with open(self.model_path,'r')as inputs:
                for individuals in inputs.readlines():
                    self.population.append(individuals.strip().split(' '))
                    for i in range(len(self.population[-1])-1):
                        self.population[-1][i]=int(self.population[-1][i])
                    self.population[-1][len(self.population[-1])-1]=float(self.population[-1][len(self.population[-1])-1])
                self.gen=self.population[-1][0]
                self.population.pop()
        for i in range(self.capacity-len(self.population)):
            self.population.append(self.generate_individual())
            self.population[-1].append(self.fitness_function(self.population[-1]))
        self.population.sort(key=lambda item:item[self.city_num])
        self.best=self.population[0][-1]
        #print(self.population)

    def fitness_function(self,idvd):
        ans=0
        for i in range(self.city_num):
            ans+=math.sqrt(pow(self.city_location[idvd[i]][0]-self.city_location[idvd[i-1]][0],2)+pow(self.city_location[idvd[i]][1]-self.city_location[idvd[i-1]][1],2))
        return ans
    def generate_individual(self):
        lst=list(range(0,self.city_num))
        random.shuffle(lst)
        return lst
    def vary(self,ex):
        x=random.randint(0,self.city_num)
        y=random.randint(0,self.city_num)
        if (x>y):
            x=x^y
            y=x^y
            x=x^y
        return ex[0:x]+list(reversed(ex[x:y]))+ex[y:self.city_num]
    def fusion(self,father,mother):
        x=random.randint(0,self.city_num)
        y=random.randint(0,self.city_num)
        if (x>y):
            x=x^y
            y=x^y
            x=x^y
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
        boys=boy[0:x]+boyc[:]+boy[x:]
        if (random.random()<self.vary_rate):
            boys=self.vary(boys)
        girls=girl[0:x]+girlc[:]+girl[x:]
        if (random.random()<self.vary_rate):
            girls=self.vary(girls)
        boys.append(self.fitness_function(boys))
        girls.append(self.fitness_function(girls))
        self.population.append(boys)
        self.population.append(girls)
    def iterate(self, num_iterations):
        pc=0
        for _ in range(num_iterations):
            lst=list(range(0,self.capacity))
            random.shuffle(lst)
            for i in range(0,(self.capacity-1)&1,2):
                self.fusion(self.population[lst[i]],self.population[lst[i+1]])
            self.population.sort(key=lambda item:item[self.city_num]) 
            self.population=self.population[0:self.capacity]
            pc+=1
            if (self.population[0][-1]<self.best):
                self.best=self.population[0][-1]
                pc=0
            if (pc>self.patience):
                print("at",_,"'s gen break")
                break
        xs=[self.city_location[i][0] for i in self.population[0][0:len(self.population[0])-1]]
        ys=[self.city_location[i][1] for i in self.population[0][0:len(self.population[0])-1]]
        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs,ys,color='b',marker='*')
        plt.show()
        return self.population[0]
        pass


if __name__ == "__main__":
    tsp = GeneticAlgTSP('实验课\Homework\hw5\qa194.tsp')
    T = 1000000
    tour = tsp.iterate(T)  # 对算法迭代10次
    print(tour)  # 打印路径(以列表的形式)
