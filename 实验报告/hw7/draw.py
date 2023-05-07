
import matplotlib.pyplot as plt
import random
import math

if __name__=='__main__':
    _name="SGDM"
    datas=[]
    with open(_name+'_log.txt','r')as ip:
        for line in ip.readlines():
            datas.append(line.rstrip().split(' '))
    x=[float(i[0]) for i in datas]
    tl=[float(i[1]) for i in datas]
    ta=[float(i[2]) for i in datas]
    vl=[float(i[3]) for i in datas]
    va=[float(i[4]) for i in datas]     
    datax=[]
    with open(_name+'_model-score.txt',"r")as ip:
         for line in ip.readlines():
            datax.append(line.rstrip().split(' '))
    begin=3
    x2=[float(i[0]) for i in datax]
    s=[float(i[1]) for i in datax]
    plt.clf()
    plt.plot(x[begin:],ta[begin:],color='g')
    plt.plot(x[begin:],va[begin:],color='b')  
    for i in range(5,len(s)):
        plt.plot([x2[i-1],x2[i]],[s[i-1],s[i-1]],color='r')
        plt.plot([x2[i],x2[i]],[s[i-1],s[i]],color='r')
    plt.plot([x2[-1],x[-1]],[s[-1],s[-1]],color='r')
    plt.title(_name+" valid_acc="+str(va[-1])+"  test_acc="+str(s[-1]))
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.savefig(_name+"accuracy.jpg")

    plt.clf()
    plt.plot(x[begin:],tl[begin:],color='g')
    plt.plot(x[begin:],vl[begin:],color='b')   
    plt.xlabel("epochs")  
    plt.ylabel("loss")
    plt.title(_name+" loss")
    plt.savefig(_name+"loss.jpg")
