"""
Demo single-file script to train a ConvNet on CIFAR10 using SoftHebb, an unsupervised, efficient and bio-plausible
learning algorithm
"""
import math
import warnings

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import torchvision


class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            kernel_size: int,  # 卷积核尺寸
            stride: int = 1,  # 卷积步长
            padding: int = 0,  # 边缘填充
            dilation: int = 1,  # 空洞卷积参数
            groups: int = 1,  # 分组卷积参数，当前简化实现不支持groups>1
            t_invert: float = 12,  # 用于softmax温度调整的参数
    ) -> None:
        super(SoftHebbConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)  # 将kernel_size转换为(height, width)格式
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'  # 设置填充模式为reflect
        self.F_padding = (padding, padding, padding, padding)  # 四边填充相同的值
        # 初始化权重参数，权重范围基于He初始化方法
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)  # softmax温度调整参数

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # 对输入x进行填充
        # 执行卷积操作，获取加权输入
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        if self.training:
            # 计算后突触激活，进行塑性更新
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # 将非竞争维度展平为(OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            # 对每个批次元素和像素计算胜出神经元
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            flat_softwta_activs = -flat_softwta_activs  # 将所有后突触激活转为反Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # 每个像素的胜出神经元
            competing_idx = torch.arange(flat_weighted_inputs.size(1))
            # 将胜出神经元的激活转回Hebbian
            flat_softwta_activs[win_neurons, competing_idx] = -flat_softwta_activs[win_neurons, competing_idx]
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # 计算塑性更新Δw
            yx = F.conv2d(
                x.transpose(0, 1),
                softwta_activs.transpose(0, 1),
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)
            # 对批次和输出像素求和
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)  # 按比例缩放
            self.weight.grad = delta_weight  # 存储梯度，供优化器使用

        return weighted_input


class DeepSoftHebb(nn.Module):
    def __init__(self):
        super(DeepSoftHebb, self).__init__()  # 调用父类的初始化函数
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)  # 对输入的3通道图像进行批量归一化，不学习仿射参数
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1,)  # 使用SoftHebb算法的第一个卷积层
        self.activ1 = Triangle(power=0.7)  # 第一个卷积层后的激活函数，使用Triangle激活函数，幂次为0.7
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)  # 第一个池化层，使用最大池化
        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)  # 第二个卷积块的批量归一化
        self.conv2 = SoftHebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65,)  # 第二个使用SoftHebb算法的卷积层
        self.activ2 = Triangle(power=1.4)  # 第二个卷积层后的激活函数，幂次为1.4
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)  # 第二个池化层
        # block 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)  # 第三个卷积块的批量归一化
        self.conv3 = SoftHebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25,)  # 第三个使用SoftHebb算法的卷积层
        self.activ3 = Triangle(power=1.0)  # 第三个卷积层后的激活函数，幂次为1.0
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # 第三个池化层，使用平均池化
        # block 4
        self.flatten = nn.Flatten()  # 展平层，用于将多维的输出展平成一维
        self.classifier = nn.Linear(24576, 10)  # 全连接层，用于分类，输入维度为24576，输出维度为10
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, 24576)  # 初始化全连接层的权重
        self.dropout = nn.Dropout(0.5)  # Dropout层，防止过拟合，丢弃率为0.5

    def forward(self, x):
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))  # 通过第一卷积块
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))  # 通过第二卷积块
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))  # 通过第三卷积块
        # block 4
        return self.classifier(self.dropout(self.flatten(out)))  # 通过全连接层进行分类


class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()  # 调用父类的初始化函数
        self.inplace = inplace  # 是否原地修改输入的标志
        self.power = power  # 幂次参数，用于控制非线性程度

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)  # 输入特征去中心化
        return F.relu(input, inplace=self.inplace) ** self.power  # 应用ReLU激活函数后，再进行幂次运算


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    """
    自定义学习率调度器，用于SoftHebb卷积块的无监督训练。
    根据当前神经元的范数与理论收敛范数（=1）之间的差异来调整初始学习率。
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        # 初始化函数
        self.optimizer = optimizer  # 优化器
        # 从优化器的参数组中提取并存储初始学习率
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]
        self.power_lr = power_lr  # 幂次，用于调整学习率调整的敏感度
        super().__init__(optimizer, last_epoch, verbose)  # 调用父类的初始化方法

    def get_lr(self):
        # 获取新的学习率值
        if not self._get_lr_called_within_step:
            # 如果在step()调用之外调用get_lr()，则发出警告
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []  # 新学习率的列表
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # 计算当前神经元的范数与理论收敛范数之间的差异
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                # 根据差异调整学习率，使用幂次运算
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr  # 返回新计算的学习率列表


class TensorLRSGD(optim.SGD):
    @torch.no_grad()  # 不计算梯度，以提高运算效率
    def step(self, closure=None):
        """
        执行单个优化步骤，使用非标量（张量）学习率。

        参数:
            closure (callable, optional): 一个闭包，重新评估模型并返回损失。
        """
        loss = None  # 初始化损失为None
        if closure is not None:  # 如果提供了闭包函数
            with torch.enable_grad():  # 启用梯度计算
                loss = closure()  # 调用闭包函数获取损失值

        for group in self.param_groups:  # 遍历所有参数组
            weight_decay = group['weight_decay']  # 权重衰减（正则化项）
            momentum = group['momentum']  # 动量
            dampening = group['dampening']  # 抑制动量的因子
            nesterov = group['nesterov']  # 是否使用Nesterov动量

            for p in group['params']:  # 遍历参数组中的每个参数
                if p.grad is None:  # 如果参数没有梯度，则跳过
                    continue
                d_p = p.grad  # 获取参数的梯度
                if weight_decay != 0:  # 如果设置了权重衰减
                    d_p = d_p.add(p, alpha=weight_decay)  # 对梯度应用权重衰减
                if momentum != 0:  # 如果设置了动量
                    param_state = self.state[p]  # 获取参数的状态
                    if 'momentum_buffer' not in param_state:  # 如果动量缓冲区不存在
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()  # 创建动量缓冲区
                    else:
                        buf = param_state['momentum_buffer']  # 获取动量缓冲区
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)  # 更新动量缓冲区
                    if nesterov:  # 如果使用Nesterov动量
                        d_p = d_p.add(buf, alpha=momentum)  # 应用Nesterov动量
                    else:
                        d_p = buf  # 使用标准动量

                p.add_(-group['lr'] * d_p)  # 更新参数，这里`group['lr']`可以是一个张量
        return loss  # 返回损失值


class CustomStepLR(StepLR):
    """
    自定义的学习率调度器，通过阶梯函数调整学习率，适用于线性读出（分类器）的监督训练。
    """

    def __init__(self, optimizer, nb_epochs):
        # 根据总训练轮数定义学习率调整的阈值比例
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        # 计算实际的学习率调整阈值轮数
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)  # 调用父类构造函数，不使用内置的步长和伽马值

    def get_lr(self):
        # 获取新的学习率值
        if self.last_epoch in self.step_thresold:
            # 如果当前轮数达到某个阈值，则学习率减半
            return [group['lr'] * 0.5 for group in self.optimizer.param_groups]
        # 否则，保持当前的学习率不变
        return [group['lr'] for group in self.optimizer.param_groups]


class FastCIFAR10(torchvision.datasets.CIFAR10):
    """
    通过移除PIL接口和在GPU上预加载数据来提高在CIFAR10上的训练性能（达到2-3倍的加速）。
    """

    def __init__(self, *args, **kwargs):
        # 从关键字参数中获取设备信息，默认为CPU
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)  # 调用父类构造函数，加载CIFAR10数据

        # 将图片数据转为张量，并将数据类型转换为浮点数，然后除以255进行归一化，最后移动到指定的设备上
        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        # 调整数据维度的顺序以符合PyTorch的输入要求：从(N,H,W,C)变为(N,C,H,W)
        self.data = torch.movedim(self.data, -1, 1)
        # 将目标（标签）也转换为张量，并移动到指定的设备上
        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        根据索引获取数据和标签。

        参数:
            index : int
                要返回的元素的索引。

        返回:
            tuple: (image, target) 其中target是目标类别的索引。
        """
        img = self.data[index]  # 获取对应索引的图像数据
        target = self.targets[index]  # 获取对应索引的目标（标签）

        return img, target  # 返回图像数据和目标（标签）


# Main training loop CIFAR10
if __name__ == "__main__":
    device = torch.device('cuda:0')  # 设置设备为第一块GPU
    model = DeepSoftHebb()  # 实例化模型
    model.to(device)  # 将模型移动到指定的设备

    # 定义无监督训练的优化器，注意这里学习率设置为负值，因为SGD进行的是下降
    unsup_optimizer = TensorLRSGD([{"params": model.conv1.parameters(), "lr": -0.08}, {"params": model.conv2.parameters(), "lr": -0.005},
        {"params": model.conv3.parameters(), "lr": -0.01}, ], lr=0)
    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)  # 无监督训练的学习率调度器

    # 定义监督训练的优化器
    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=50)  # 监督训练的学习率调度器
    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    # 加载CIFAR10训练集和测试集
    trainset = FastCIFAR10('/data/hwj/dataset', train=True, download=True)
    unsup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    sup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = FastCIFAR10('/data/hwj/dataset', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    # 无监督训练阶段
    running_loss = 0.0
    for i, data in enumerate(unsup_trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        unsup_optimizer.zero_grad()  # 清零梯度

        with torch.no_grad():  # 不计算梯度，因为是无监督训练
            outputs = model(inputs)

        unsup_optimizer.step()  # 优化步骤
        unsup_lr_scheduler.step()  # 调整学习率

    # 监督训练阶段，训练分类器
    unsup_optimizer.zero_grad()  # 清零梯度
    # 将卷积层设置为不更新梯度和评估模式
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv3.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.conv3.eval()
    model.bn1.eval()
    model.bn2.eval()
    model.bn3.eval()

    # 开始监督训练
    for epoch in range(50):
        model.classifier.train()  # 将分类器设置为训练模式
        model.dropout.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(sup_trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            sup_optimizer.zero_grad()  # 清零梯度

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            sup_optimizer.step()  # 参数更新

            # 计算训练统计数据
            running_loss += loss.item()
            if epoch % 10 == 0 or epoch == 49:  # 每10个epoch打印一次训练统计数据
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        sup_lr_scheduler.step()  # 调整学习率

        # 测试集上的评估
        if epoch % 10 == 0 or epoch == 49:
            print(f'Accuracy of the network on the train images: {100 * correct // total} %')
            print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

            model.eval()  # 设置模型为评估模式
            running_loss = 0.
            correct = 0
            total = 0
            with torch.no_grad():  # 测试阶段不计算梯度
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
            print(f'test loss: {running_loss / total:.3f}')
