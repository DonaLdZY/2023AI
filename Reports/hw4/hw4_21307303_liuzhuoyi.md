# 人工智能实验报告 第5周

姓名:刘卓逸  学号:21307303

### 一.实验题目

hw4 启发式搜索算法

### 二.实验内容

##### 1.算法原理

```
将初始状态加入openlist
while (openlist不为空且目标状态未进入closedlist) do
    在openlist中找到估值函数最小的状态x
    将x从openlist中移除并加入到closedlist
    从x扩展出下一步的状态
    若下一步的状态不在closedlist中
        将这个状态加入到openlist中
        并且尝试更新新状态的估值函数
            更新成功则更新标记
判断目标状态在closedlist中
    若存在，则找到了解，返回解
    否则无解
```

##### 2.实验过程与关键代码展示

###### (1)哈希

二维列表无法被自动哈希，所以无法直接作为字典的键，于是考虑写一个哈希函数

每一个数只可能是0~15,用4位即可表示，每个状态共16个数，故可以用一个64位无符号整数表示一个状态

具体哈希函数如下:

```python
def encode(puzzle): #编码
    ans=0
    for i in range(4):
        for j in range(4):
            ans+=puzzle[i][j]*(1<<(4*(i*4+j)))
    return ans
```

显然这个哈希函数是可逆的，可以通过将哈希值转换回二维列表，方便状态转移

```python
def decode(x): #解码
    puzzle=[ [0 for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            puzzle[i][j]=x&15
            x>>=4
    return puzzle
```

###### (2)估值函数

定义估值函数f(x)=g(x)+h(x)

g(x)为初始状态到目前状态所走的步数

h(x)为每一个数离它的目标位置的曼哈顿距离之和

```'python
def miraishi(puzzle:list): #未来视 估值函数
    ans=0
    for i in range(4):
        for j in range(4):
            ii=((puzzle[i][j]+15)%16)/4
            jj=(puzzle[i][j]+3)%4
            ans+=abs(i-ii)+abs(j-jj)
    return ans
```

###### (3)辅助的函数

找出0(空位)在哪的函数

```python
def findzero(puzzle:list):
    for i in range(4):
        for j in range(4):
            if puzzle[i][j]==0:
                return (i,j)
```

产生下一个状态的函数
'''python
def move(nw:list,xz,yz,xi,yi): #生成新的状态,并且[xz,yz]与[xi,yi]交换
    nxt=[[nw[i][j] for j in range(4)] for i in range(4)]
    temp=nxt[xz][yz]
    nxt[xz][yz]=nxt[xi][yi]
    nxt[xi][yi]=temp
    return nxt
'''

###### (4)测试用代码

```python
if __name__ == '__main__':
    # 可自己创建更多用例并分析算法性能

    puzzle1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    puzzle2 = [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]]
    sol1 = A_star(puzzle1)
    sol2 = A_star(puzzle2)
    print(sol1)
    print(sol2)
    puzzle3= [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]]
    sol3=A_star(puzzle3)
    print(sol3)
    puzzle4= [[0,5,15,14],[7,9,6,13],[1,2,12,10],[12,13,7,15]]
    sol4=A_star(puzzle4)
    print(sol4)
```

###### (5)核心代码-v1

在找出估值函数最小的状态时，用的方法是是暴力遍历整个openlist

```python
def A_star(puzzle):
    fa=[(-1,0),(1,0),(0,1),(0,-1)]
    ans=[]
    origin=encode(puzzle)
    finall=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])
    trap=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,15,14,0]])
    fr={origin:[0,0,0]} #from来自信息哈希值:[动了几步，动了什么，上一步的hash]
    op={origin:miraishi(puzzle)} #openinglist{哈希值:估值函数值}
    cl=set() #closedlist
    while len(op)>0 and (not (finall in cl)):
        nwh=0
        nwv=1145141919 #INf
        for (oph,opv) in op.items(): #寻找估值函数最小的
            if nwv>opv:
                nwh=oph
                nwv=opv
        cl.add(nwh) #加入closed
        del op[nwh] #从open中删除
        nw=decode(nwh) #解码
        (xz,yz)=findzero(nw) #找到0在哪
        for i in range(4):
            (xi,yi)=(xz+fa[i][0],yz+fa[i][1])
            if (xi>=0 and xi<=3 and yi>=0 and yi<=3): #开始扩展
                nxt=move(nw,xz,yz,xi,yi)
                nxth=encode(nxt)
                #print("go",nxt)
                if (nxth in cl):
                    continue
                if (nxth==trap):
                    print("fuck this")
                    return []
                if nxth in op.values():
                    if fr[nxth][0]>fr[nwh][0]+1:
                        fr[nxth]=[fr[nwh][0]+1,nw[xi][yi],nwh]
                        op[nxth]=fr[nxth][0]+miraishi(nxt)
                else:
                    fr[nxth]=[fr[nwh][0]+1,nw[xi][yi],nwh]
                    op[nxth]=fr[nxth][0]+miraishi(nxt)
    if not(finall in cl):
        print("No answer")
        return []
    i=finall
    while (i!=origin):
        ans.append(fr[i][1])
        i=fr[i][2]
    ans.reverse()
    return ans
```

但结果是跑task3就卡死了，一分多钟没有出结果

###### (6)核心代码-v2

优化思路:

1.将标记fr的值设为结构体增强可读性

2.一个状态的估值hx部分只需算一遍即可，储存下来

3.考虑最终步数应该不会特别特别大，所以考虑用桶以估值函数为键来装openlist

### 三.实验结果及分析

##### 1.实验结果展示示例
