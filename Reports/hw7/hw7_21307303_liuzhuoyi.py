_exp_name="hw7_21307303_liuzhuoyi"
#------ import ------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


#------ Dataset ------
class MyDataset(Dataset):
    def __init__(self,datas_dir="",label_dir=""):
        if datas_dir=="":
            return 
        self.datas=torch.load(datas_dir)
        self.label=torch.load(label_dir)

    def __len__(self):
        return self.label.size()[0]
    
    def __getitem__(self,idx):
        return self.datas[idx],self.label[idx]
#------ Dataloader ------
n_workers=8 #用于数据加载的子进程数

def collate_batch(batch):
    a,b=zip(*batch)
    return a,b

def get_dataloader(datas_dir,label_dir, batch_size, n_workers,valid_sept=0):
    dataset=MyDataset(datas_dir,label_dir)
    if valid_sept>0:
        #分割成trainset与validset
        validlen=int(valid_sept*len(dataset))
        lengths=[len(dataset)-validlen,validlen]
        trainset,validset=random_split(dataset,lengths)
        train_loader=DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True, #训练时将训练数据顺序打乱
            num_workers=n_workers, #用于数据加载的子进程数
            drop_last=True, #最后一个batch可能不满batch_size,抛弃掉
            pin_memory=True, #存在固定内存，加速
        )
        valid_loader=DataLoader(
            validset,
            batch_size=batch_size,
            shuffle=False, #验证时就没必要打乱了
            num_workers=n_workers,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader,valid_loader
    else:
        loader=DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=n_workers, 
            drop_last=True,
            pin_memory=True, 
        )
        return loader
#------ Model ------
#确保每次调用卷积算法返回确定性输出，即默认算法
torch.backends.cudnn.deterministic = True 
#固定网络结构的模型优化以提高效率，否则会花费时间在选择最合适算法上
torch.backends.cudnn.benchmark = False 

class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #input [1,28,28]
        self.cnn=nn.Sequential(
            nn.Conv2d(1,16,3,1,1), #[16,28,28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[16,14,14]

            nn.Conv2d(16,32,3,1,1), #[32,14,14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[32,7,7]
        )
        self.fc=nn.Sequential(
            nn.Linear(32*7*7,8*7*7),
            nn.ReLU(),
            nn.Linear(8*7*7,4*7*7),
            nn.ReLU(),
            nn.Linear(4*7*7,10),
        )
    def forward(self, x):
        if len(x.size())==3:
            x=x.unsqueeze(0)
        out=self.cnn(x)
        out=out.view(out.size()[0],-1)
        out=self.fc(out)
        return torch.softmax(out,dim=-1)

#------ hyper parameter ------
batch_size=16
n_epochs=32
patience=8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------ test ------
def test(data, labels, net):
    num_data = data.shape[0]
    num_correct = 0
    for i in range(num_data):
        feature = data[i]
        prob = net(feature).detach()
        dist = Categorical(prob)
        label = dist.sample().item()
        true_label = labels[i].item()
        if label == true_label:
            num_correct += 1

    return num_correct / num_data
def Testing():
    net= MyConvNet()
    test_data = torch.load('data\\hw7\\test_data.pth')
    test_labels = torch.load('data\\hw7\\test_labels.pth')
    net.load_state_dict(torch.load('hw7_21307303_liuzhuoyi.pth'))
    net.eval()
    return test(test_data, test_labels, net)

#------ main ------
from tqdm.auto import tqdm #进度条可视化

if __name__=="__main__":
    #模型实例化
    model=MyConvNet().to(device)
    #分类问题中,用cross-entropy来定义损失函数,用平方误差会导致离答案很远很近都梯度很小
    criterion = nn.CrossEntropyLoss()
    #Adam 动态学习率(加快收敛速度)+惯性梯度(避免local minimal)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    best_acc=0
    with open(f"./{_exp_name}_log.txt","w")as op:
        op.write("0 0 0 0\n")
    for epoch in range(n_epochs):
        #dataset/dataloader实例化
        train_loader,valid_loader=get_dataloader(
            datas_dir='data\\hw7\\train_data.pth',
            label_dir='data\\hw7\\train_labels.pth',
            batch_size=batch_size, 
            n_workers=n_workers,
            valid_sept=0.1
        )
        test_loader=get_dataloader(
            datas_dir='data\\hw7\\test_data.pth',
            label_dir='data\\hw7\\test_labels.pth',
            batch_size=batch_size, 
            n_workers=n_workers,
        )
        #----- Training ------
        #训练模式
        model.train() 
        #训练记录
        train_loss=[]
        train_accs=[]
        for batch in tqdm(train_loader):
            #数据
            imgs,labels=batch
            #模型预测
            # imgs.size()=torch.Size([16, 1, 28, 28])
            logits=model(imgs.to(device))
            #计算损失
            loss=criterion(logits,labels.to(device))
            #梯度归零
            optimizer.zero_grad()
            #反向传播 计算梯度
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            #更新参数
            optimizer.step()
            #计算准确率并记录
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        # 输出信息
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        with open(f"./{_exp_name}_log.txt","a") as op:
            op.write(f"{epoch + 1:03d} {train_acc:.5f} ")
        # ------ Validation ------
        #无梯度信息模式
        model.eval()
        #信息记录
        valid_loss=[]
        valid_accs=[]
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            # 验证时不需要计算梯度
            with torch.no_grad():
                logits = model(imgs.to(device))
            #求损失函数
            loss = criterion(logits, labels.to(device))
            # 求准确率
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            # 记录.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        # 求平均loss与平均准确率
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # 更新日志
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}",end="")
        if valid_acc > best_acc:
            print(" -> best") 
        else:
            print("")
        with open(f"./{_exp_name}_log.txt","a") as op:
            op.write(f"{valid_acc:.5f}    {Testing():.5f}\n")
   
        # 保存模型
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}.pth") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    print("final test acc = ",Testing())
    # 如果已经训练完成, 则直接读取网络参数. 注意文件名改为自己的信息
    