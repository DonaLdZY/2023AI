class task1:
    def check(a:tuple,b:tuple):
        for i in a:
            for j in b:
                if ('~'+i==j):
                    return (i,1)
                if ('~'+j==i):
                    return (j,0)
        return ('',0)
    def fusion(a:tuple,b:tuple,key:str):
        ans=[]
        for i in a:
            if i!=key:
                ans.append(i)
        for i in b:
            if i!='~'+key:
                ans.append(i)
        return tuple(ans)
    def order(self,ans:list,fa:list,origin:int,x:int,temp:list):
        if x<=origin or temp[x]==1:
            print(x,origin,temp[x])
            return x
        print("go")
        lf=self.order(ans,fa,origin,fa[x][0],temp)
        rf=self.order(ans,fa,origin,fa[x][1],temp)
        ans.append(str(len(ans)+1)+' R['+str(lf)+','+str(rf)+']: '+str(f[x][2]))
        return x
def ResolutionProp(KB):
    ans=[]
    a=list(KB)
    ai=len(a)
    ais=len(a)
    fa=[]
    fa=[(0,0,"") for i in range(ai+1)]
    for i in range(ai):
        ans.append(str(i+1)+' '+str(a[i]))
    i=1
    while (i<ai):
        j=i+1
        while j<ai:
            (key,fr)=task1.check(a[i],a[j])
            if (key!=''):
                aadd=()
                if fr:
                    aadd=task1.fusion(a[i],a[j],key)
                else:
                    aadd=task1.fusion(a[j],a[i],key)
                a.append(aadd)
                ai+=1
                fa.append((i+1,j+1,str(aadd)))
                ans.append(str(ai)+' R['+str(i+1)+','+str(j+1)+']: '+str(aadd))
                if aadd==():
                    temp=[0 for i in range(ai+1)]
                    task1.order(ex,fa,ais,ai,temp)
                    return ans
            j+=1
        i+=1
    return ans

# def ResolutionProp(KB):
#     ans=[]
#     a=list(KB)
#     ai=len(a)
#     ais=len(a)
#     fa=[]
#     fa=[(0,0,"") for i in range(ai+1)]
#     for i in range(ai):
#         ans.append(str(i+1)+' '+str(a[i]))
#     i=1
#     while (i<ai):
#         j=i+1
#         while j<ai:
#             (key,fr)=task1.check(a[i],a[j])
#             if (key!=''):
#                 aadd=()
#                 if fr:
#                     aadd=task1.fusion(a[i],a[j],key)
#                 else:
#                     aadd=task1.fusion(a[j],a[i],key)
#                 a.append(aadd)
#                 ai+=1
#                 fa.append((i+1,j+1,str(aadd)))
#                 ans.append(str(ai)+' R['+str(i+1)+','+str(j+1)+']: '+str(aadd))
#                 if aadd==():
#                     temp=[0 for i in range(ai+1)]
#                     task1.order(ex,fa,ais,ai,temp)
#                     return ans
#             j+=1
#         i+=1
#     return ans

def isvariate(f:str): #判断是否是变量
    if f in ['xx','yy','zz','uu','vv','ww']:
        return True
    return False
def diff(f1:list,f2:list): #找出不匹配项
    for i in range(len(f1)):
        if f1[i]!=f2[i]:
            return (f1[i],f2[i])
def peel(f:str): #去掉最外层谓词
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
def MGU(f1:str, f2:str):
    (f1name,f1in)=peel(f1)
    (f2name,f2in)=peel(f2)
    if (f1name != f2name):
        return {}
    f1item=f1in.split(',')
    f2item=f2in.split(',')
    ans={}
    while (f1item != f2item):
        (f1x,f2x)=diff(f1item,f2item)
        while (f1x[-1]==')' and f2x[-1]==')'):
            (n1,f1x)=peel(f1x)
            (n2,f2x)=peel(f2x)
            if (n1!=n2):
                return {}
        if (isvariate(f1x) and (not f1x in f2x)):
            if f1x in ans:
                return {}
            ans[f1x]=f2x
            f1item=[f1item[i].replace(f1x,f2x) for i in range(len(f1item))]
            f2item=[f2item[i].replace(f1x,f2x) for i in range(len(f2item))]
            ans=dict((k,v.replace(f1x,f2x)) for k,v in ans.items())
        elif (isvariate(f2x) and (not f2x in f1x)):
            if f2x in ans:
                return {}
            ans[f2x]=f1x
            f1item=[f1item[i].replace(f2x,f1x) for i in range(len(f1item))]
            f2item=[f2item[i].replace(f2x,f1x) for i in range(len(f2item))]
            ans=dict((k,v.replace(f2x,f1x)) for k,v in ans.items())
        else:
            return{}
    return ans
def check(a:tuple,b:tuple):
    if (len(a)>1 and len(b)>1):
        return ("",{},2)
    for ai in a:
        (ainame,aitem)=peel(ai)
        for bi in b:
            (biname,bitem)=peel(bi)
            if (ainame=='~'+biname):
                if (ai=='~'+bi):
                    return (bi,{},0)
                chg=MGU(ai[1:len(ai)],bi)
                if (chg!={}):
                    return (bi,chg,0)
            if ('~'+ainame==biname):
                if ('~'+ai==bi):
                    return (ai,{},1)
                chg=MGU(ai,bi[1:len(bi)])
                if (chg!={}):
                    return (ai,chg,1)
    return ("",{},2)
def chgs(a:str,chg:dict):
    for (key,val) in chg.items():
        #print("changing",key,val)
        a=a.replace(key,val)
    return a
def fusion(a:tuple,b:tuple,key:str,chg:dict):
    key=chgs(key,chg)
    ans=[]
    for j in a:
        i=chgs(j,chg)
        if i!=key:
            ans.append(i)
    for j in b:
        i=chgs(j,chg)
        if i!='~'+key:
            ans.append(i)
    return tuple(ans)
def ResolutionFOL(KB):
    ans=[]
    a=list(KB)
    ai=len(a)
    ais=ai
    for i in range(ais):
        ans.append(str(i+1)+' '+str(a[i]))
    i=0
    while i<ai:
        j=i+1
        while j<ai:
            (key,chg,fr)=check(a[i],a[j])
            if (fr!=2):
                aadd=()
                if fr:
                    aadd=fusion(a[i],a[j],key,chg)
                else:
                    aadd=fusion(a[j],a[i],key,chg)
                #print(a[i],a[j],'=>',fr,key,chg,aadd)
                if (aadd in a):
                    j+=1
                    continue
                a.append(aadd)
                ai+=1
                if (chg=={}):
                    ans.append(str(ai)+' R['+str(i+1)+','+str(j+1)+']: '+str(aadd))
                else:
                    s=str(ai)+' R['+str(i+1)+','+str(j+1)+']{'
                    for (key,value) in chg.items():
                        s+=key+'='+value+','
                    s=s[:len(s)-1]+'}: '+str(aadd)
                    ans.append(s)
                if aadd==():
                    return ans
            j+=1
        i=i+1
    return ans


if __name__ == '__main__':
    # 测试程序
    KB1 = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    #result1 = ResolutionProp(KB1)
    result1 = ResolutionProp(KB1)
    for r in result1:
        print(r)
    # MGU('P', 'P')
    # print(MGU('P(xx,a)', 'P(b,yy)'))
    # print(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))

    # KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    # result2 = ResolutionFOL(KB2)
    # for r in result2:
    #     print(r)
