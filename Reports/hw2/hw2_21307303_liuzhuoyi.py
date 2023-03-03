class StuData:
    def __init__(self,filename:str):
        self.data = []
        with open(filename,'r') as data:
            for student in data.readlines().:
                self.data.append(student.split(' '))
        print(self.data)

    def AddData(self,name,stu_num,gender,age):
        self.data.append([name,stu_num,gender,age])

    def SortData(self):
        pass

    def ExportFile(self):
        pass


if __name__ == '__main__':
    # 测试程序
    s1 = StuData('C:\\Users\\DonaLdZY\\Documents\\Lesson\\2023AI\\Reports\\hw2\\student_data.txt')
    # s1.AddData(name="Bob", stu_num="003", gender="M", age=20)
    # s1.SortData('age')
    # s1.ExportFile('new_stu_data.txt')