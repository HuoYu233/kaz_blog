---
title: 小土堆深度学习学习笔记
mathjax: true
date: 2025/7/1 20:46:25
img: https://i1.hdslb.com/bfs/archive/ce2cdffabecfa83b56b588d7dbab21f90eb54281.png
excerpt: 小土堆深度学习学习笔记
---
## 准备

下载anaconda，管理库

下载cuda版本的pytorch，验证

```python
torch.cuda.is_available()
```

## 数据准备

### Dataset

自己的dataset，要继承torch.util.DataSet，重写一些方法

```python
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.my_list = os.listdir(self.path)

    def get_item(self, idx):
        img_name = self.my_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        img.show()

    def get_length(self):
        return len(self.my_list)
```

### DataLoader

```python
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# dataset 选定的数据集
# batch_size 每次选的数据数量
# shuffle 是否打乱
# num_workers 线程数 一般为0
# drop_last 是否将最后一次不足batch_size大小的数据舍去
for data in test_loader:
    imgs, targets = data
    print(imgs)
    print(targets)
```

## TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
import math
import numpy
from PIL import Image

writer = SummaryWriter("logs")

path = "../dataset/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(path)
img_array = numpy.array(img_PIL)
writer.add_image("test", img_array, 1, dataformats="HWC")
for i in range(-10, 10):
    writer.add_scalar("y=sin(x)", math.sin(i), i)

writer.close()
```

```bash
tensorboard --logdir=logs --port=6006
#命令行启动tensorboard
```

## Transform

torchvision里的transform.py，里面有很多类，来实例化对象处理图片

补充：`__init__`是实例化对象，构造函数，`__call__`让类的实力可以像函数一样被调用

```python
MyTensor = transforms.ToTensor()
tensor_img = MyTensor(img)  # 将图片转化为Tensor

trans_nor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
normal_img = trans_nor(tensor_img)

size = transforms.Resize((512, 512))
img_size = size(img)

size_2 = transforms.Resize(512)
trans_compose = transforms.Compose([size_2, MyTensor])
img_size_2 = trans_compose(img) #组合处理
```

## TorchVision的数据集

```python
tensor_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train = torchvision.datasets.CIFAR10("mydata", True, transform=tensor_transform, download=True)
test = torchvision.datasets.CIFAR10("mydata", False, transform=tensor_transform, download=True)
#true or false代表着是否是训练集
#每个都是(img, label)
```

## 神经网络

```python
class Kaz(nn.Module):
    def __init__(self):
        super(Kaz, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)
```

## 卷积

- **特征提取**：通过滑动窗口（卷积核）扫描输入数据，检测局部特征（如边缘、纹理、颜色等）。
- **参数共享**：同一个卷积核在整个输入上复用，减少参数量（相比全连接层）。
- **平移不变性**：无论特征出现在图像的哪个位置，卷积层都能检测到。

卷积操作调用F.conv2d，指定input，kernel，stride等

在神经网络中使用Conv2d添加卷积层，指定in_channel，out_channel等

## 池化

- **降维（下采样）**：减少特征图尺寸，降低计算量。
- **增强平移不变性**：对微小位置变化不敏感。
- **防止过拟合**：减少参数量的间接效果。

有最大池化和平均池化

池化的stride和卷积默认不一样，卷积默认1，池化默认kernel_size

具体还是参照pytorch文档

## 非线性激活

inplace选择是否修改input

组合模型模块用nn.Sequential()

## 损失函数和反向传播

```python
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))
loss3 = nn.CrossEntropyLoss()

result = loss3(x, y)
```

```python
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    output = huoyu(imgs)
    result = loss(output, targets)
    print(result)
    result.backward() #反向传播，生成梯度
```

## 优化器

```python
huoyu = Kaz()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(huoyu.parameters(), lr=0.01)
for i in range(20):
    total_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = huoyu(imgs)
        result = loss(output, targets)
        optim.zero_grad() # 梯度清零
        result.backward() # 生成梯度
        optim.step()  # 更新参数
        total_loss += result
    print("{0}轮的损失值:{1}".format(i, total_loss))
```

## 现有模型的使用和修改

```python
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.add_module('add_linear', nn.Linear(1000, 10))
vgg16.classifier.add_module('7', nn.Linear(1000, 10)) #classifier是vgg里面的一个模块
vgg16.classifier[6] = nn.Linear(4096, 10)
```

## 模型的保存和读取

```python
vgg16 = torchvision.models.vgg16()
# 方式1，模型结构+参数
torch.save(vgg16, 'vgg16.pth')
get_vgg16 = torch.load('vgg16.pth')
print(get_vgg16)

# 方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), 'vgg16_2.pth')
get_vgg16_2 = torch.load('vgg16_2.pth')
print(get_vgg16_2)  # 输出字典
v = torchvision.models.vgg16()
v.load_state_dict(get_vgg16_2)
print(v)
```

## 模型训练

```python
print(len(train_data))
print(len(test_data))
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 创建网络模型实例
huoyu = Kaz()
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(huoyu.parameters(), lr=learning_rate)

# 设置训练网络的参数
train_step = 0
test_step = 0
epoch = 10
sw = SummaryWriter("logs")

for i in range(epoch):
    print("----------第{0}轮训练开始----------".format(i+1))
    huoyu.train() # 只是一个标识，有Dropout，BatchNorm层才有作用
    for data in train_loader:
        imgs, targets = data
        outputs = huoyu(imgs)
        loss = loss_function(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_step += 1
        if train_step % 100 == 0 :
            print("{0}次训练的loss值:{1}".format(train_step, loss.item()))
            sw.add_scalar("train_loss", loss.item(), train_step)
    # 测试
    huoyu.eval() # 只是一个标识，有Dropout，BatchNorm层才有作用
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = huoyu(imgs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() # argmax(1)沿着列看
            # output[64, 10] 经过argmax(1)变成[64,1],与target比较变成[64, 1]的Bool，sum得到预测正确的个数
            total_accuracy += accuracy
    print("整体损失:{0}".format(total_loss))
    sw.add_scalar("test_loss", total_loss, test_step)
    print("整体正确率:{0}".format(total_accuracy/len(test_data)))
    sw.add_scalar("test_accuracy",total_accuracy/len(test_data), test_step)
    test_step += 1

sw.close()
```

## GPU训练

1. 把所有网络模型，损失函数，数据转化为.cuda()，只有N卡才可以 

```python
loss_function = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda()
```

2. 设置device

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
```

##  验证

```python
path = "../pic/dog.png"
image = Image.open(path)
image = image.convert("RGB")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()])

image = transform(image)

model = torch.load("model21.pth", map_location=torch.device("cpu")) #gpu训练好的模型，放在cpu上运行需要加map_location
# print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output.argmax(1))
```

