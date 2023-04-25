def BinarySearch(nums, target):
    """
    :param nums: list[int]
    :param target: int
    :return: int
    """
    l=0
    r=len(nums)-1
    while l<=r:
        m=(l+r)//2
        if nums[m]==target:
            return m
        elif nums[m]>target:
            r=m-1
        else:
            l=m+1
    return -1
#end binarysearch

def MatrixAdd(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
#end matrixadd

def MatrixMul(A, B):
    """
    :param A: list[list[int]]
    :param B: list[list[int]]
    :return: list[list[int]]
    """
    C=[[0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j]+=A[i][k]*B[k][j]
    return C
#end matrixmul

def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    dict2=dict(zip(dict1.values(),dict1.keys()))
    return dict2
#end reversekeyvalue

if __name__ == "__main__":
    print("输出", BinarySearch([-1, 0, 3, 5, 9, 12], 9), "答案", 4)
    print("输出", MatrixAdd([[1,0],[0,1]], [[1,2],[3,4]]), "答案", [[2, 2], [3, 5]])
    print("输出", MatrixMul([[1,0],[0,1]], [[1,2],[3,4]]), "答案", [[1, 2], [3, 4]])
    print("输出", ReverseKeyValue({'Alice':'001', 'Bob':'002'}), "答案", {'001':'Alice', '002':'Bob'})