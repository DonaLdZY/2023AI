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
def heuristic(puzzle): #启发式函数
    ans=0
    for i in range(4):
        for j in range(4):
            if (puzzle[i][j]==0):
                continue
            ii=((puzzle[i][j]+15)%16)//4
            jj=(puzzle[i][j]+3)%4
            ans+=abs(i-ii)+abs(j-jj)
    return ans
def findzero(puzzle): #找0
    for i in range(4):
        for j in range(4):
            if puzzle[i][j]==0:
                return (i,j)
def move(nw:list,xz,yz,xi,yi): #生成新的状态,并且[xz,yz]与[xi,yi]交换
    nxt=[[nw[i][j] for j in range(4)] for i in range(4)]
    nxt[xz][yz],nxt[xi][yi]=nxt[xi][yi],nxt[xz][yz]
    return nxt
class Node:
    def __init__(self,state,gx,hx,action,parent):
        self.state=state
        self.gx=gx
        self.hx=hx
        self.action=action
        self.parent=parent
    def fx(self):
        return int(2*self.hx+self.gx)
    def __lt__(self,other):
        return self.fx()<other.fx()

def A_star(puzzle):
    fa=[(-1,0),(1,0),(0,1),(0,-1)] #方向数组
    origin=encode(puzzle)
    finall=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]) #最终状态
    trap=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,15,14,0]]) #死胡同状态
    temp=Node(origin,0,heuristic(puzzle),0,0)
    infos={ origin: temp } #信息，gx已走步数、hx估值函数、action上一步走了什么、parent从什么状态转移来
    cl=set() #closedlist
    op=[ [] for i in range(11451) ] #openlist op[x]为所有估值函数为x为状态组成的列表
    opset={origin}
    op[int(infos[origin].fx())].append(origin)
    ops:int=infos[origin].fx()
    count=0
    while (not (finall in cl)):
        count+=1
        if count%100000==0:
            print(count,ops)
        while (ops<11451 and len(op[int(ops)])==0):
            ops+=1
        if (ops==11451):
            break
        nwh=op[int(ops)][len(op[int(ops)])-1]
        op[ops].pop() #从op中删除
        if (nwh in cl):
            continue
        nw=decode(nwh) #解码
        (xz,yz)=findzero(nw) #找到0在哪
        #print(nwh)
        opset.remove(nwh)
        cl.add(nwh) #加入closed
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
                if not nxth in opset:
                #     if infos[nxth].gx>infos[nwh].gx+1:
                #         infos[nxth].gx=infos[nwh].gx+1
                #         op[infos[nxth].fx()].append(nxth)
                #         if infos[nxth].fx()<ops:
                #             ops=infos[nxth].fx()
                # else:
                    infos[nxth]=Node(nxth,infos[nwh].gx+1,heuristic(nxt),nw[xi][yi],nwh)
                    op[infos[nxth].fx()].append(nxth)
                    opset.add(nxth)
                    if infos[nxth].fx()<ops:
                        ops=infos[nxth].fx()
    if not(finall in cl):
        print("No answer")
        return []
    i=finall
    ans=[]
    while (i!=origin):
        ans.append(infos[i].action)
        i=infos[i].parent
    ans.reverse()
    return ans

def IDA_star(puzzle):
    pass
if __name__ == '__main__':
    print("go task1")
    puzzle1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    sol1 = A_star(puzzle1)
    print(len(sol1))
    print(sol1)

    print("go task2")
    puzzle2 = [[5, 1, 3, 4], [2, 7, 8, 12], [9, 6, 11, 15], [0, 13, 10, 14]]
    sol2 = A_star(puzzle2)
    print(len(sol2))
    print(sol2)

    print("go task3")
    puzzle3= [[14,10,6,0],[4,9,1,8],[2,3,5,11],[12,13,7,15]]
    sol3=A_star(puzzle3)
    print(len(sol3))
    print(sol3)
    print("go task4")
    puzzle4= [[0,5,15,14],[7,9,6,13],[1,2,12,10],[12,13,7,15]]
    sol4=A_star(puzzle4)
    print(sol4)
