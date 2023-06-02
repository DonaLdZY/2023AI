#------ import env ------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
#确保每次调用卷积算法返回确定性输出，即默认算法
torch.backends.cudnn.deterministic = True 
#固定网络结构的模型优化以提高效率，否则会花费时间在选择最合适算法上
torch.backends.cudnn.benchmark = False 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------ hyper parameter for Data ------
batch_size=16
n_workers=0 #用于数据加载的子进程数 （实验得出越小越好）
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
def get_dataloader(datas_dir,label_dir,sept=0):
    dataset=MyDataset(datas_dir,label_dir)
    if sept>0:
        #分割成trainset与validset
        validlen=int(sept*len(dataset))
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
            shuffle=True, 
            num_workers=n_workers, 
            drop_last=True,
            pin_memory=True, 
        )
        return loader
    
#------ Model ------
class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #input [1,28,28]
        self.cnn=nn.Sequential(
            nn.Conv2d(1,64,3,1,1), #[4,28,28]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[64,14,14]
            
            nn.Conv2d(64,128,3,1,1), #[128,14,14]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0), #[128,7,7]
        )
        self.fc=nn.Sequential(
            nn.Linear(128*7*7,16*7*7),
            nn.ReLU(),
            nn.Linear(16*7*7,2*7*7),
            nn.ReLU(),
            nn.Linear(2*7*7,10)
        )
    def forward(self, x):
        if len(x.size())==3:
            x=x.unsqueeze(0)
        out=self.cnn(x)
        out=out.view(out.size()[0],-1)
        out=self.fc(out)
        return torch.softmax(out,dim=-1)
#------ hyper parameter for training------
n_epochs=64
patience=48
log_open=True
#------ filename ------
_exp_name="hw7_21307303_liuzhuoyi_v3"
modelfile=_exp_name+'.pth'
logfile=_exp_name+"_log.txt"
scorefile=_exp_name+"_model-score.txt"
train_datafile='data\\hw7\\train_data.pth'
train_labelfile='data\\hw7\\train_labels.pth'
#------ test ------
def test(data, labels, net):
    num_data = data.shape[0]
    num_correct = 0
    for i in range(num_data):
        feature = data[i].unsqueeze(0)
        prob = net(feature).squeeze(0).detach()
        label = torch.argmax(prob).item()
        true_label = labels[i].item()
        if label == true_label:
            num_correct += 1
    return num_correct / num_data
def Testing():
    net= MyConvNet()
    test_data = torch.load('data\\hw7\\test_data.pth')
    test_labels = torch.load('data\\hw7\\test_labels.pth')
    net.load_state_dict(torch.load(modelfile))
    return test(test_data, test_labels, net)
#------ main ------
if __name__=="__main__":
    from tqdm.auto import tqdm #进度条可视化
    #模型实例化
    model=MyConvNet().to(device)
    #分类问题中,用cross-entropy来定义损失函数,用平方误差会导致离答案很远很近都梯度很小
    criterion = nn.CrossEntropyLoss()
    #Adam 动态学习率(加快收敛速度)+惯性梯度(避免local minimal)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler=StepLR(optimizer,step_size=2,gamma=0.9)

    if log_open:
        with open(logfile,"w")as op:
            op.write("0 0 0 0 0\n")
        with open(scorefile,"w")as op:
            op.write("0 0\n")

    #dataset/dataloader实例化
    train_loader,valid_loader=get_dataloader(
        datas_dir=train_datafile,
        label_dir=train_labelfile,
        sept=0.167
    )
    best_acc=0

    for epoch in range(n_epochs):
        #----- Training ------
        #开启梯度信息
        model.train() 
        #训练记录
        train_loss=[]
        train_accs=[]

        for batch in tqdm(train_loader):
            #获取数据
            imgs,labels=batch
            #模型预测
            logits=model(imgs.to(device))
            #计算损失函数
            loss=criterion(logits,labels.to(device))
            #梯度清零
            optimizer.zero_grad()
            #反向传播，计算梯度
            loss.backward()
            #梯度裁剪，防止梯度爆炸
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            #更新参数
            optimizer.step()
            #计算准确率并记录
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        #求平均损失与准确率
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ------ Validation ------
        #关闭梯度信息
        model.eval()
        #验证记录
        valid_loss=[]
        valid_accs=[]

        for batch in tqdm(valid_loader):
            #获取数据
            imgs, labels = batch
            # 验证时不需要计算梯度
            with torch.no_grad():
                logits = model(imgs.to(device))
            #求损失函数
            loss = criterion(logits, labels.to(device))
            #计算准确率并记录
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        #求平均损失与准确率
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        # 更新日志
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}" + (" ->best " if (valid_acc>best_acc) else "") )

        if log_open:
            with open(logfile,"a") as op:
                op.write(f"{epoch + 1:03d} {train_loss:.5f} {train_acc:.5f} {valid_loss:.5f} {valid_acc:.5f}\n")
   
        # 保存模型
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), modelfile) # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0

            if log_open:
                with open(scorefile,"a") as op:
                    op.write(f"{epoch + 1:03d} {Testing():.5f}\n")
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
        scheduler.step()

    print("final test acc = ",Testing())
    