import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import pandas as pd
import os
import torch.optim as optim
import time

#https://zhuanlan.zhihu.com/p/488193814, 20层的Resnet，加载ImageNet的Pretrain模型，91.28%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torchvision.__version__)
print(device)

train_batch_size = 128
test_batch_size = 128
num_workers = 0 #线程数
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
lr = 0.001
momentum = 0.9
Train_rate=0.8
EPOCH = 3#100
BATCH_SIZE = train_batch_size

#加载数据集
transform_train=transforms.Compose([transforms.RandomCrop(32, padding=4), #对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
transforms.RandomHorizontalFlip(), #以给定的概率随机水平翻转给定的PIL图像，默认值为0.5
transforms.ToTensor(), #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),#用平均值和标准偏差归一化张量图像，#(M1,…,Mn)和(S1,…,Sn)将标准化输入的每个通道
])
transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
##下载数据集并设置存放目录，训练与否，下载与否，数据预处理转换方式等参数，得到train_dataset、test_dataset
train_data = torchvision.datasets.CIFAR10(root='./cifar10/',train=True,transform=transform_train,download=True)
test_data = torchvision.datasets.CIFAR10(root='./cifar10/',train=False,transform=transform_test,download=True)
train_loader = DataLoader(train_data,batch_size=train_batch_size,shuffle=True,num_workers=num_workers)
test_loader = DataLoader(test_data,batch_size=test_batch_size,shuffle=False,num_workers=num_workers)
#展示数据集类型及数据集的大小
print(type(train_data))
print('训练样本数：',len(train_data), ', 测试样本数：',len(test_data))
class_label = train_data.classes
print('类别标签：',class_label)

#数据可视化
import matplotlib.pyplot as plt
import numpy as np
print('--------------查看一批训练样本------------')
plt.figure()
def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
examples = enumerate(train_loader)
idx, (examples_data, examples_target) = next(examples) #examples_target是标签列表,0-9表示不同的类别
imshow(torchvision.utils.make_grid(examples_data))
#用于具体查看examples，#打印其中一个图像数据对应的标签
print('一批训练样本_target.shape:{}'.format(examples_target.shape))
print('一批训练样本_target[0]:{}'.format(examples_target[0]))
print('一批训练样本_data.shape:{}'.format(examples_data.shape))

#构建网络
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)   size//2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)    #size 不变
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):#isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    def forward(self, x):
        return self._forward_impl(x)

model = ResNet(block=BasicBlock,layers=[3, 3, 3])
model = model.to(device)#网络模型搬移至指定的设备上
print('--------------查看网络结构-----------')
print(model)

print("training on ", device) #查看当前训练所用的设备
def train_model(model,traindataloader,train_rate,criterion,optimizer,num_epochs):
    #train_rate：训练集batchsize百分比
    #计算训练使用的batch数量
    #batch_num = len(traindataloader)/200
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate) #前train_rate的batch进行训练
    print("总批数：",batch_num,"， 训练批数：",train_batch_num,"， 训练批量大小：",BATCH_SIZE)
    #复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-' * 10)
        #每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step,(b_x,b_y) in enumerate(traindataloader):
            b_x, b_y=b_x.to(device),b_y.to(device)#将图像、标签搬移至指定设备上
            if step < train_batch_num: #前train_rate的batch进行训练
                model.train() #设置模型为训练模式
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y) #loss是一个batch的loss，每个样本的loss均值，
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0) #此处的b_x.size(0)=batch_size。此处相当于一个batch的loss
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            if step < batch_num and step >= train_batch_num:
                model.eval() #设置模型为评估模式
                output = model(b_x)
                pre_lab = torch.argmax(output,1)
                loss = criterion(output,b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        #计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))
        #拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60,time_use % 60))
    #使用最好模型的参数
    model.load_state_dict(best_model_wts)
    #组成数据表格train_process输出
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    return model,train_process

#--训练模型--
print('-----训练学习过程-------')

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=lr)
mynet,train_process = train_model(model,train_loader,Train_rate,criterion,optimizer,num_epochs=EPOCH )

#可视化模型训练过程中
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_process.epoch,train_process.train_loss_all,"ro-",label="Train loss")
plt.plot(train_process.epoch,train_process.val_loss_all,"bs-",label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(train_process.epoch,train_process.train_acc_all,"ro-",label="Train acc")
plt.plot(train_process.epoch,train_process.val_acc_all,"bs-",label="Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

#保存网络模型的参数至mymodel.pth文件
#torch.save(model.state_dict(),'mymodel.pth')

#测试模型
model.eval()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
num_correct = 0
with torch.no_grad():
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        out = model(img).to(device)
        #计算准确率
        _, pred = out.max(1)
        num_correct += (pred == label).sum()
        #计算各类别准确率
        c = (pred == label)
        for i in range(16):
            class_correct[label[i]] += c[i].item() #将True/False化为1/0
            class_total[label[i]] += 1
    print('--------------查看测试结果-----------')
    for i in range(10):
        print("accuracy of {}:{}%".format(classes[i],100*class_correct[i]/class_total[i]))
    print("测试集的精确率为:{}".format(num_correct / (len(test_loader) * test_batch_size)))




