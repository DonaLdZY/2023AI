
import matplotlib.pyplot as plt
import random
import math

if __name__=='__main__':
    datas=[]
    with open('hw7_21307303_liuzhuoyi_log.txt','r')as ip:
        for line in ip.readlines():
            datas.append(line.rstrip().split(' '))
    for i in datas:
        print(i)
    x=[float(i[0]) for i in datas]
    tl=[float(i[1]) for i in datas]
    ta=[float(i[2]) for i in datas]
    vl=[float(i[3]) for i in datas]
    va=[float(i[4]) for i in datas]     
    datax=[]
    with open('hw7_21307303_liuzhuoyi_test_log.txt',"r")as ip:
         for line in ip.readlines():
            datax.append(line.rstrip().split(' '))
    x2=[float(i[0]) for i in datax]
    s=[float(i[1]) for i in datax]
    plt.clf()
    plt.plot(x[1:],ta[1:],color='g')
    plt.plot(x[1:],va[1:],color='b')  
    for i in range(1,len(s)):
        plt.plot([x2[i-1],x2[i]],[s[i-1],s[i-1]],color='r')
        plt.plot([x2[i],x2[i]],[s[i-1],s[i]],color='r')
    plt.plot([x2[-1],x[-1]],[s[-1],s[-1]],color='r')
    plt.title("accuracy")
    plt.savefig("accuracy.jpg")
    plt.clf()
    plt.plot(x[1:],tl[1:],color='g')
    plt.plot(x[1:],vl[1:],color='b')     
    plt.title("loss")
    plt.savefig("loss.jpg")
# def drawmap(self): #绘制当前路线
#         plt.clf()
#         xs=[self.city_location[i][0] for i in self.population[0][0:len(self.population[0])-1]]
#         ys=[self.city_location[i][1] for i in self.population[0][0:len(self.population[0])-1]]
#         xs.append(xs[0])
#         ys.append(ys[0])
#         plt.plot(xs,ys,color='b')
#         plt.scatter(xs,ys,color='r',s=10)
#         plt.title('gen '+str(self.gen)+'\nShortest Path: '+str(self.best))
#         plt.savefig(self.log_path+str(self.data_name)+'_gen'+str(self.gen)+'.jpg')
#     def plot_diagram(self): #绘制数据迭代表
#         plt.clf()
#         # for i in range(len(self.log_point)):
#         #     if self.max_log[i]!=self.min_log[i]:
#         #         plt.plot([self.log_point[i], self.log_point[i]], [self.max_log[i],self.min_log[i]], color='b')
#         for i in range(1,len(self.log_point)):
#             plt.plot([self.log_point[i-1], self.log_point[i]], [self.max_log[i-1],self.max_log[i]], color='r')
#         plt.title('gen '+str(self.gen)+'\nShortest Path: '+str(self.best))
#         plt.ylabel('cost')
#         plt.xlabel('iteration')
#         plt.savefig(self.log_path+str(self.data_name)+'_overall'+'.jpg')
    