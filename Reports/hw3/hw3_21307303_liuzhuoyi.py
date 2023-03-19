def isvariate(f:str): #判断是否是变量
    if f in ['xx','yy','zz','uu','vv','ww']:
        return True
    return False
def diff(f1:list,f2:list): #找出不匹配项
    for i in range(len(f1)):
        if f1[i]!=f2[i]:
            return (f1[i],f2[i])
def peel(f:str): #分离外层谓词名与内容
    nl=0
    while nl<len(f) and (f[nl].isalpha() or f[nl]=='~'):
        nl+=1
    name=f[0:nl]
    if nl<len(f):
        f=f[nl+1:len(f)]
        f=f[0:len(f)-1]
    else:
        f=""
    return (name,f)
def MGU(f1:str, f2:str): #归一化
    (f1name,f1in)=peel(f1) #拆出最外层的谓词
    (f2name,f2in)=peel(f2)
    if (f1name != f2name): #最外层谓词不同那不考虑
        return {}
    f1item=f1in.split(',') #分离项
    f2item=f2in.split(',')
    ans={}
    while (f1item != f2item): #没有归一完成
        (f1x,f2x)=diff(f1item,f2item) #找出不同项
        while (f1x[-1]==')' and f2x[-1]==')'): #两个的外层还有函数，剥开函数
            (n1,f1x)=peel(f1x)
            (n2,f2x)=peel(f2x)
            if (n1!=n2): #若外层函数名不同那肯定不能归一直接结束
                return {}
        if (isvariate(f1x) and (not f1x in f2x)): #f1是变量f2是项的情况
            if f1x in ans: #f1已经被替换过（这可能吗，匹配完不都是直接消失的吗）
                return {}
            ans[f1x]=f2x #写入字典
            f1item=[x.replace(f1x,f2x) for x in f1item] #两个原子命题与字典进行全替换
            f2item=[x.replace(f1x,f2x) for x in f2item]
            ans=dict((k,v.replace(f1x,f2x)) for k,v in ans.items())
        elif (isvariate(f2x) and (not f2x in f1x)): #f2是变量f1是项的情况
            if f2x in ans:
                return {}
            ans[f2x]=f1x
            f1item=[x.replace(f2x,f1x) for x in f1item]
            f2item=[x.replace(f2x,f1x) for x in f2item]
            ans=dict((k,v.replace(f2x,f1x)) for k,v in ans.items())
        else: #两个常量的情况，寄
            return{}
    return ans
def check(a:tuple,b:tuple):
    '''
    查询a与b能否归结
    return (key,chg,from)
        key:归结的项
        chg:归一化的变量替换
        from:0:归结项在A中取否, 1:在B中取否, 2无法归结
    '''
    for ai in a:
        (ainame,aitem)=peel(ai)
        for bi in b:
            (biname,bitem)=peel(bi)
            if (ainame=='~'+biname): #谓词相同且互补
                if (ai=='~'+bi): #整条式子完全相同，不需要进行归一化
                    return (bi,{},0) 
                chg=MGU(ai[1:len(ai)],bi) #进行归一化，返回归一化所需变量替换
                if (chg!={}): #如果归一化成功
                    return (bi,chg,0) #可以归结
            if ('~'+ainame==biname):
                if ('~'+ai==bi):
                    return (ai,{},1)
                chg=MGU(ai,bi[1:len(bi)])
                if (chg!={}):
                    return (ai,chg,1)
    return ("",{},2) #扫完了，无法归结

def chgs(a:str,chg:dict): #将字符串a按照chg进行所有的变量替换
    for (key,val) in chg.items():
        a=a.replace(key,val)
    return a

def fusion(a:tuple,b:tuple,key:str,chg:dict):
    key=chgs(key,chg)
    ans=[]
    for j in a:
        i=chgs(j,chg)
        if i!=key and not i in ans:
            ans.append(i)
    for j in b:
        i=chgs(j,chg)
        if i!='~'+key and not i in ans:
            ans.append(i)
    return tuple(ans)
def order(table:list,fa:list,origin:int,x:int): #table过程 fa推理来源 origin原始条件 x最终结果
    temp=[i<=origin for i in range(x+1)]
    realnum=[0 for i in range(x+1)]
    temp[x]=True
    bfs=[x,]
    bi=0
    while (bi<len(bfs)):
        k=bfs[bi]
        if k<=origin:
            bi+=1
            continue
        if (not temp[fa[k][0]]):
            bfs.append(fa[k][0])
            temp[fa[k][0]]=True
        if (not temp[fa[k][1]]):
            bfs.append(fa[k][1])
            temp[fa[k][1]]=True
        bi+=1
    ex=[]
    for i in range(1,x+1):
        if temp[i]:
            if fa[i][0]>0 or fa[i][1]>0:
                ex.append(str(len(ex)+1)+' R['+str(realnum[fa[i][0]])+','+str(realnum[fa[i][1]])+']'+table[i])
            else:
                ex.append(str(len(ex)+1)+' '+table[i])
            realnum[i]=len(ex)
    return ex
def ResolutionFOL(KB):
    ans=[""]
    a=list(KB)
    ai=len(a)
    ais=len(a)
    fa=[(0,0) for i in range(ai+1)]
    for i in range(ais):
        ans.append(str(a[i]))
    i=0
    while i<ai:
        j=0
        while j<ai:
            (key,chg,fr)=check(a[i],a[j])
            if (fr!=2):
                aadd=()
                if fr:
                    aadd=fusion(a[i],a[j],key,chg)
                else:
                    aadd=fusion(a[j],a[i],key,chg)
                #print(a[i],a[j],'=>',fr,key,chg,aadd)
                if (aadd in a or len(aadd)>=max(len(a[i]),len(a[j]))):
                    j+=1
                    continue
                a.append(aadd)
                ai+=1
                s='{'
                for (key,value) in chg.items():
                    s+=key+'='+value+','
                s=s[:len(s)-(0 if (s[len(s)-1]=='{') else 1 )]+'}: '+str(aadd)
                ans.append(s)
                fa.append((i+1,j+1))
                if aadd==():
                    return order(ans,fa,ais,ai)
            j+=1
        i+=1
    return order(ans,fa,ai,ai)
def ResolutionProp(KB):
    ans=[""]
    a=list(KB)
    ai=len(a)
    ais=len(a)
    fa=[(0,0) for i in range(ai+1)]
    for i in range(ais):
        ans.append(str(a[i]))
    i=0
    while i<ai:
        j=0
        while j<ai:
            (key,chg,fr)=check(a[i],a[j])
            if (fr!=2):
                aadd=()
                if fr:
                    aadd=fusion(a[i],a[j],key,chg)
                else:
                    aadd=fusion(a[j],a[i],key,chg)
                #print(a[i],a[j],'=>',fr,key,chg,aadd)
                if (aadd in a or len(aadd)>=max(len(a[i]),len(a[j]))):
                    j+=1
                    continue
                a.append(aadd)
                ai+=1
                s=': '+str(aadd)
                ans.append(s)
                fa.append((i+1,j+1))
                if aadd==():
                    return order(ans,fa,ais,ai)
            j+=1
        i+=1
    return order(ans,fa,ai,ai)

if __name__ == '__main__':
    print("----test1----")
    KB1 = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    result1 = ResolutionProp(KB1)
    for r in result1:
        print(r)
    print("----test2----")
    print(MGU('P(xx,a)', 'P(b,yy)'))
    print(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))
    print("----test3----")
    KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    result2 = ResolutionFOL(KB2)
    for r in result2:
        print(r)
    print("----test3 Pro----")
    KB3={('A(tony)',),('A(mike)',),('A(john)',),('L(tony,rain)',),('L(tony,snow)',),('~A(xx)','S(xx)','C(xx)'),('~C(yy)','~L(yy,rain)'),('L(zz,snow)','~S(zz)'),('~L(tony,uu)','~L(mike,uu)'),('L(tony,vv)','L(mike,vv)'),('~A(ww)','~C(ww)','S(ww)')}
    result3 = ResolutionFOL(KB3)
    for r in result3:
        print(r)
    print("----test3 Alter----")
    KB4={('F(xx)',), ('~F(fgo)','C(yy)'), ('~C(pcr)','D(ygo)'),('~F(csgo)','C(zz)'),('D(duel)',)}
    result4 = ResolutionFOL(KB4)
    for r in result4:
        print(r)