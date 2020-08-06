from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def get_transform():
    return transforms.Compose([
            # 图像缩放到32 x 32
            transforms.Resize(64),
            # 中心裁剪 32 x 32
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            # 对每个像素点进行归一化
            transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                 std=[0.2, 0.2, 0.2])
        ])

def get_dataset(batch_size=10, num_workers=1):
    data_transform = get_transform()
    # load训练集图片
    train_dataset = ImageFolder('./data/train', transform=data_transform)
    # load验证集图片
    test_dataset = ImageFolder('./data/test', transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader


#函数说明：定义模型的结构
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(10000, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 4)
        )

    def forward(self, x):
        output = self.model(x)
        return output

#函数说明：使用反向传播算法训练模型
def train(epoch):

    for i, (images, labels) in enumerate(data_train_loader):
        data = images.view(-1, 10000)

        logits = net(data)#输入数据，返回模型预测结果
        labels = labels.long()#将标签数据类型转换为long类型，否则会报错
        loss = criteon(logits, labels)#计算交叉熵损失

        optimizer.zero_grad()
        loss.backward()#反向传播
        optimizer.step()

        if i % 10 == 0:#每十组输出一次loss
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

#函数说明：模型测试，返回分类的正确率
def test():
    net.eval()
    total_correct = 0#正确分类的个数
    avg_loss = 0.0#平均损失
    for i, (images, labels) in enumerate(data_test_loader):
        data = images.view(-1, 10000)
        output = net(data)
        labels = labels.long()
        avg_loss += criteon(output, labels).sum()
        pred = output.detach().max(1)[1]

        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))

    return float(total_correct) / len(data_test)

#函数说明：读入人脸数据，返回一个一维向量和对应的标签
def load_faces(root_str):

    X = []  # 存放图像数据
    Label = []  # 存放标签
    transforms1 = transforms.Compose([
        # 图像缩放到100 x 100
        transforms.Resize(100),
        # 中心裁剪 100 x 100
        transforms.CenterCrop(100),

    ])
    directory = os.fsencode(root_str)
    for file in os.listdir(directory):#读取根目录下的文件夹
        filename = os.fsdecode(file)
        file_path = os.path.join(root_str, filename)#访问下一级目录
        #Label.extend([int(os.path.basename(file_path).replace('s', ''))] * 10)#存放标签
        for f in os.listdir(file_path):#读取图像

            Label.append(int(filename))
            infile = os.path.join(file_path, f)#读取图像地址
            img = Image.open(infile)#读取图像，存入img
            img = transforms1(img)
            img = np.array(img)
            im_vec = np.reshape(img, -1)#将图像变为一维向量
            X.append(im_vec)

    faces = np.array(X, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)

    return faces, idLabel  # 返回图像数据集和对应的标签


x_train, y_train = load_faces('./data/train')
x_test, y_test = load_faces('./data/test')
x_train = x_train.T
x_test = x_test.T
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
data_train = Data.TensorDataset(x_train, y_train)
data_test = Data.TensorDataset(x_test, y_test)
data_train_loader = Data.DataLoader(data_train, batch_size=8, shuffle=True)
data_test_loader = Data.DataLoader(data_test, batch_size=8)
net = Model()

criteon = nn.CrossEntropyLoss()#交叉熵
optimizer = optim.Adam(net.parameters(), lr=5e-6, weight_decay=0.0005)#优化器

index = []
accuracy = []
for epoch in range(1, 11):
    train(epoch)
    Accuracy = test() * 100
    index.append(epoch)
    accuracy.append(Accuracy)

#画出正确率变化的曲线
plt.plot(index, accuracy, color='red', linewidth=2.0)
plt.title('5 layers Model')
plt.xlabel('epoch')
plt.ylabel('accuracy(%)')
plt.show()
