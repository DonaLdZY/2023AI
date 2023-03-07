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
def ResolutionProp(KB):
    ans=[]
    a=list(KB)
    ai=len(a)
    for i in range(ai):
        ans.append(str(i+1)+' '+str(a[i]))
    for i in range(ai):
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
                ans.append(str(ai)+' R['+str(i+1)+','+str(j+1)+']: '+str(aadd))
                if aadd==():
                    return ans
            j+=1
    return ans

def decoder(f:str): #将原子公式拆解成谓词名与项
    nl=0
    while nl<len(f) and f[nl].isalpha():
        nl+=1
    name=f[0:nl]
    if nl<len(f):
        f=f[nl+1:len(f)]
        f=f[0:len(f)-1]
        items=f.split(',')
    else:
        items=[]
    return (name,items)
def diff(f1:list,f2:list):
    for i in range(len(f1)):
        if f1[i]!=f2[i]:
            return (f1[i],f2[i])
def peel(f:str):
    nl=0
    while nl<len(f) and f[nl].isalpha():
        nl+=1
    name=f[0:nl]
    f=f[nl+1:len(f)]
    f=f[0:len(f)-1]
    return (name,f)
def peels(f1:str,f2:str):
    while (f1[-1]==')' and f2[-1]==')'):
        (temp,f1)=peel(f1)
        (temp,f2)=peel(f2)
    return (f1,f2)
def isvariate(f:str):
    if f in ['xx','yy','zz','uu','vv','ww']:
        return True
    return False
def MGU(f1:str, f2:str):
    (f1name,f1item)=decoder(f1)
    (f2name,f2item)=decoder(f2)
    ans={}
    k=0
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
            i=0
            while i<len(f1item):
                f1item[i]=f1item[i].replace(f1x,f2x)
                i+=1
        elif (isvariate(f2x) and (not f2x in f1x)):
            if f2x in ans:
                return {}
            ans[f2x]=f1x
            i=0
            while i<len(f2item):
                f2item[i]=f2item[i].replace(f2x,f1x)
                i+=1
        else:
            return{}
    return ans


def ResolutionFOL(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    return


if __name__ == '__main__':
    # 测试程序
    KB1 = {('FirstGrade',), ('~FirstGrade', 'Child'), ('~Child',)}
    result1 = ResolutionProp(KB1)
    for r in result1:
        print(r)
    MGU('P', 'P')
    print(MGU('P(xx,a)', 'P(b,yy)'))
    print(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))

    # KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    # result2 = ResolutionFOL(KB2)
    # for r in result2:
    #     print(r)
