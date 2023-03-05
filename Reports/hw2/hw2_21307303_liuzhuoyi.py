class StuData:
    def __init__(self,filename:str):
        self.data =[]
        with open(filename,'r') as inputs:
            for student in inputs.readlines():
                #self.data.AddData(student.rstrip().split(' '))
                self.data.append(student.rstrip().split(' '))
                self.data[-1][-1]=int( self.data[-1][-1])
        print(self.data)
    
    def AddData(self,name,stu_num,gender,age):
        self.data.append([name,stu_num,gender,int(age)])
        print(self.data)

    def SortData(self,go):
        if go=='name':
            keys=0
        elif go=='stu_num':
            keys=1
        elif go=='gender':
            keys=2
        else:
            keys=3
        self.data.sort(key=lambda item:item[keys], reverse=False)
        print(self.data)

    def ExportFile(self,filename:str):
        with open(filename,'w') as outputs:
            for student in self.data:
                for data in student:
                    outputs.write(str(data)+" ")
                outputs.write('\n')


if __name__ == '__main__':
    # 测试程序
    s1 = StuData('student_data.txt')
    s1.AddData(name="Bob", stu_num="003", gender="M", age=20)
    s1.SortData('age')
    s1.ExportFile('new_stu_data.txt')