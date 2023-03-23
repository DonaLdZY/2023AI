def encode(puzzle): #编码
    ans=0
    for i in range(4):
        for j in range(4):
            ans+=puzzle[i][j]*(1<<(4*(i*4+j)))
    return ans
def decode(x): #解码
    puzzle=[ [0 for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            puzzle[i][j]=x&15
            x>>=4
    return puzzle
def findzero(puzzle:list):
    for i in range(4):
        for j in range(4):
            if puzzle[i][j]==0:
                return (i,j)
def move(nw:list,xz,yz,xi,yi): #生成新的状态,并且[xz,yz]与[xi,yi]交换
    nxt=[[nw[i][j] for j in range(4)] for i in range(4)]
    temp=nxt[xz][yz]
    nxt[xz][yz]=nxt[xi][yi]
    nxt[xi][yi]=temp
    return nxt
def miraishi(puzzle:list): #未来视 估值函数
    ans=0
    for i in range(4):
        for j in range(4):
            ii=((puzzle[i][j]+15)%16)//4
            jj=(puzzle[i][j]+3)%4
            ans+=abs(i-ii)+abs(j-jj)
    return ans
def A_star(puzzle):
    fa=[(-1,0),(1,0),(0,1),(0,-1)]
    ans=[]
    origin=encode(puzzle)
    finall=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])
    trap=encode([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,15,14,0]])
    fr={origin:[0,0,0]} #from来自信息哈希值:[动了几步，动了什么，上一步的hash]
    op={origin:2*miraishi(puzzle)} #openinglist{哈希值:估值函数值}
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
                        op[nxth]=fr[nxth][0]+2*miraishi(nxt)
                else:
                    fr[nxth]=[fr[nwh][0]+1,nw[xi][yi],nwh]
                    op[nxth]=fr[nxth][0]+2*miraishi(nxt)
                    if (nxth==finall):
                        cl.add(finall)
    if not(finall in cl):
        print("No answer")
        return []
    i=finall
    while (i!=origin):
        ans.append(fr[i][1])
        i=fr[i][2]
    ans.reverse()
    return ans
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

    print("go task3")
    puzzle = [[14, 10, 6, 0],[4, 9 ,1 ,8],[2, 3, 5 ,11],[12, 13, 7 ,15]]
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task4")
    puzzle = [[6, 10, 3, 15],[14, 8, 7, 11], [5, 1, 0, 2],[13, 12, 9, 4]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task5")
    puzzle = [[11, 3, 1, 7],[4, 6, 8, 2], [15, 9, 10, 13],[14, 12, 5, 0]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)

    print("go task6")
    puzzle = [[0, 5, 15, 14],[7, 9, 6, 13], [1, 2, 12, 10],[8, 11, 4, 3]] 
    start=time.time()
    sol = A_star(puzzle)
    end=time.time()
    print("time:",end-start)
    print(len(sol),sol)
