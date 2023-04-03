def encode(puzzle): #编码
    ans=0
    for i in range(4):
        for j in range(4):
            ans+=puzzle[i][j]*(1<<(((i<<2)+j)<<2))
    return ans
def decode(x): #解码
    puzzle=[ [0 for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            puzzle[i][j]=x&15
            x>>=4
    return puzzle
def findzero(puzzle): #找0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j]==0:
                return (i,j)
def move(nw:list,xz,yz,xi,yi): #生成新的状态,并且[xz,yz]与[xi,yi]交换
    nxt=[[nw[i][j] for j in range(4)] for i in range(4)]
    nxt[xz][yz],nxt[xi][yi]=nxt[xi][yi],nxt[xz][yz]
    return nxt
def heuristic(puzzle): #启发式函数
    miracle=[0,1,4,5,6,7,8]
    #miracle=[0,1,4,5,7,8,9]
    ans=0
    for i in range(4):
        for j in range(4):
            if (puzzle[i][j]==0):
                continue
            ii=((puzzle[i][j]+15)&15)>>2
            jj=(puzzle[i][j]+3)&3
            ans+=miracle[abs(i-ii)+abs(j-jj)]
    return int(ans)

class Node:
    def __init__(self,gx,hx,action,parent):
        self.gx=gx
        self.hx=hx
        self.action=action
        self.parent=parent
    def fx(self):
        return int(self.hx+self.gx)
    def __lt__(self,other):
        return self.fx()<other.fx()

def A_star(puzzle):
    fa=[(-1,0),(0,-1),(1,0),(0,1)] #方向数组
    maxstep=512
    origin=encode(puzzle)
    finall=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]) #最终状态
    if (origin==finall):
        return []
    trap=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,15,14,0]]) #死胡同状态
    if (origin==trap):
        print("No answer")
        return []

    infos={ origin: Node(0,heuristic(puzzle),0,0) } #信息，gx已走步数、hx估值函数、action上一步走了什么、parent从什么状态转移来
    cl=set() #closedlist
    op=[ [] for i in range(maxstep) ] #openlist 桶排序,op[x]为所有估值函数为x为状态组成的列表
    op[infos[origin].fx()].append(origin)
    ops=infos[origin].fx()
    opset={origin} #是否在op中
    while (not (finall in cl)):
        while (ops<maxstep and len(op[ops])==0):
            ops+=1
        if (ops==maxstep):
            print("not found")
            return []
        nwh=op[ops][len(op[ops])-1]
        op[ops].pop() #从op中删除
        if (nwh in cl):
            continue
        opset.remove(nwh)
        cl.add(nwh) #加入closed
        nw=decode(nwh) #解码
        (xz,yz)=findzero(nw) #找到0在哪
        for i in range(4):
            (xi,yi)=(xz+fa[i][0],yz+fa[i][1])
            if (xi>=0 and xi<=3 and yi>=0 and yi<=3): #开始扩展
                nxt=move(nw,xz,yz,xi,yi)
                nxth=encode(nxt)
                if (nxth in cl):
                    continue
                if (nxth==trap):
                    print("No answer")
                    return []
                if nxth in opset:
                    if infos[nxth].gx>infos[nwh].gx+1:
                        infos[nxth].gx=infos[nwh].gx+1
                        op[infos[nxth].fx()].append(nxth)
                        if infos[nxth].fx()<ops:
                            ops=infos[nxth].fx()
                else:
                    infos[nxth]=Node(infos[nwh].gx+1,heuristic(nxt),nw[xi][yi],nwh)
                    op[infos[nxth].fx()].append(nxth)
                    opset.add(nxth)
                    if infos[nxth].fx()<ops:
                        ops=infos[nxth].fx()
    i=finall
    ans=[]
    while (i!=origin):
        ans.append(infos[i].action)
        i=infos[i].parent
    return ans[::-1]

def IDA_star(puzzle):
    pass

if __name__ == '__main__':
    import time
    print("go task0")
    puzzle = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task1")
    puzzle = [[1, 2, 4, 8], [5, 7, 11, 10], [13, 15, 0, 3], [14, 6, 9, 12]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task2")
    puzzle = [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task3 (2)")
    puzzle = [[14, 10, 6, 0],[4, 9 ,1 ,8],[2, 3, 5 ,11],[12, 13, 7 ,15]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task4 (4)")
    puzzle = [[6, 10, 3, 15],[14, 8, 7, 11], [5, 1, 0, 2],[13, 12, 9, 4]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task5 (1)")
    puzzle = [[11, 3, 1, 7],[4, 6, 8, 2], [15, 9, 10, 13],[14, 12, 5, 0]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task6 (3)")
    puzzle = [[0, 5, 15, 14],[7, 9, 6, 13], [1, 2, 12, 10],[8, 11, 4, 3]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)
