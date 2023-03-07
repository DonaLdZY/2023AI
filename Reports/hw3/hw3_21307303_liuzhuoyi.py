def check(a:tuple,b:tuple):
    for i in a:
        for j in b:
            if ('~'+i==j):
                return (i,1)
            if ('~'+j==i):
                return (j,0)
    return ('',0)

def fushion(a:tuple,b:tuple,key:str):
    ans=[]
    for i in a:
        if i!=key:
            ans.append(i)
    for i in b:
        if i!='~'+key:
            ans.append(i)
    return tuple(ans)

def ResolutionProp(KB):
    """
    :param KB: set(tuple(str))
    :return: list[str]
    """
    ans=[]
    a=list(KB)
    ai=len(a)
    for i in range(ai):
        ans.append(str(i+1)+' '+str(a[i]))
        
    i=0
    while i+1<ai:
        j=i+1
        while j<ai:
            (share,fr)=check(a[i],a[j])
            if (share!=''):
                aadd=()
                if fr:
                    aadd=fushion(a[i],a[j],share)
                else:
                    aadd=fushion(a[j],a[i],share)
                a.append(aadd)
                ai+=1
                ans.append(str(ai)+' R['+str(i+1)+','+str(j+1)+']: '+str(aadd))
                if aadd==():
                    return ans
            j+=1
        i+=1
    return ans

def MGU(f1, f2):
    """
    :param f1: str
    :param f2: str
    :return: dict
    """
    return


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

    # print(MGU('P(xx,a)', 'P(b,yy)'))
    # print(MGU('P(a,xx,f(g(yy)))', 'P(zz,f(zz),f(uu))'))

    # KB2 = {('On(a,b)',), ('On(b,c)',), ('Green(a)',), ('~Green(c)',), ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    # result2 = ResolutionFOL(KB2)
    # for r in result2:
    #     print(r)
