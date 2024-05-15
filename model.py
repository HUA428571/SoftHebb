import torch
import torch.nn as nn

try:
    from utils import RESULT, activation
except:
    from hebb.utils import RESULT, activation
from layer import generate_block
import os
import os.path as op


def load_layers(params, model_name, resume=None, verbose=True, model_path_override=None):
    """
    Create Model and load state if resume
    """

    if resume is not None:
        if model_path_override is None:
            model_path = op.join(RESULT, 'network', model_name, 'models', 'checkpoint.pth.tar')
        else:
            model_path = model_path_override

        if op.isfile(model_path):
            checkpoint = torch.load(model_path)  # , map_location=device)
            state_dict = checkpoint['state_dict']
            params2 = checkpoint['config']
            if resume == 'without_classifier':
                classifier_key = list(params.keys())[-1]
                params2[classifier_key] = params[classifier_key]

            model = MultiLayer(params2)

            state_dict2 = model.state_dict()

            if resume == 'without_classifier':
                for key, value in state_dict.items():
                    if resume == 'without_classifier' and str(params[classifier_key]['num']) in key:
                        continue
                    if key in state_dict2:
                        state_dict2[key] = value
                model.load_state_dict(state_dict2)
            else:
                model.load_state_dict(state_dict)
            # log.from_dict(checkpoint['measures'])
            starting_epoch = 0  # checkpoint['epoch']
            print('\n', 'Model %s loaded successfuly with best perf' % (model_name))
            # shutil.rmtree(op.join(RESULT, params.folder_name, 'figures'))
            # os.mkdir(op.join(RESULT, params.folder_name, 'figures'))
        else:
            print('\n', 'Model %s not found' % model_name)
            model = MultiLayer(params)
        print('\n')
    else:
        model = MultiLayer(params)

    if verbose:
        model.__str__()

    return model


def save_layers(model, model_name, epoch, blocks, filename='checkpoint.pth.tar', storing_path=None):
    """
    Save model and each of its training blocks
    """
    if storing_path is None:
        if not op.isdir(RESULT):
            os.makedirs(RESULT)
        if not op.isdir(op.join(RESULT, 'network')):
            os.mkdir(op.join(RESULT, 'network'))
            os.mkdir(op.join(RESULT, 'layer'))

        folder_path = op.join(RESULT, 'network', model_name)
        if not op.isdir(folder_path):
            os.makedirs(op.join(folder_path, 'models'))
        storing_path = op.join(folder_path, 'models')


    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config,
        'epoch': epoch
    }, op.join(storing_path, filename))

    for block_id in blocks:
        block = model.get_block(block_id)
        block_path = op.join(RESULT, 'layer', 'block%s' % block.num)
        if not op.isdir(block_path):
            os.makedirs(block_path)
        folder_path = op.join(block_path, block.get_name())
        if not op.isdir(folder_path):
            os.mkdir(folder_path)
        torch.save({
            'state_dict': block.state_dict(),
            'epoch': epoch
        }, op.join(folder_path, filename))


class MultiLayer(nn.Module):
    """
    MultiLayer类是一个自定义的神经网络模型，它由预设的块列表创建。

    参数:
    blocks_params: dict, 块的参数字典
    blocks: nn.Module, 块的模块，如果没有提供，则会根据blocks_params生成
    """

    def __init__(self, blocks_params: dict, blocks: nn.Module = None) -> None:
        """
        初始化函数，创建MultiLayer对象。

        参数:
        blocks_params: dict, 块的参数字典
        blocks: nn.Module, 块的模块，如果没有提供，则会根据blocks_params生成
        """
        super().__init__()
        self.train_mode = None  # 训练模式
        self.train_blocks = []  # 训练块列表

        self.config = blocks_params  # 块的配置参数
        if blocks_params is not None:  # 如果提供了块的参数
            blocks = []  # 初始化块列表
            for _, params in blocks_params.items():  # 遍历每个块的参数
                blocks.append(generate_block(params))  # 生成块并添加到列表
            self.blocks = nn.Sequential(*blocks)  # 将块列表转换为序列
        else:  # 如果没有提供块的参数
            self.blocks = nn.Sequential(*blocks)  # 直接将提供的块转换为序列

    def foward_x_wta(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入x进行前向传播，并返回最后一个块的输出。

        参数:
        x: torch.Tensor, 输入张量

        返回:
        torch.Tensor, 最后一个块的输出
        """
        for id, block in self.generator_block():  # 遍历每个块
            if id != len(self.blocks) - 1:  # 如果不是最后一个块
                x = block(x)  # 对x进行处理
            else:  # 如果是最后一个块
                return block.foward_x_wta(x)  # 返回最后一个块的输出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入x进行前向传播，并返回最后一个块的输出。

        参数:
        x: torch.Tensor, 输入张量

        返回:
        torch.Tensor, 最后一个块的输出
        """
        x = self.blocks(x)  # 对x进行处理
        return x  # 返回处理后的结果

    def get_block(self, id):
        """
        获取指定id的块。

        参数:
        id: int, 块的id

        返回:
        nn.Module, 指定id的块
        """
        return self.blocks[id]

    def sub_model(self, block_ids):
        """
        获取一个子模型，该子模型包含了从第一个块到指定id的块。

        参数:
        block_ids: list, 块的id列表

        返回:
        MultiLayer, 子模型
        """
        sub_blocks = []  # 初始化子块列表
        max_id = max(block_ids)  # 获取最大的id
        for id, block in self.generator_block():  # 遍历每个块
            sub_blocks.append(self.get_block(id))  # 将块添加到子块列表
            if id == max_id:  # 如果达到了最大的id
                break  # 结束循环

        return MultiLayer(None, sub_blocks)  # 返回子模型

    def is_hebbian(self) -> bool:
        """
        判断最后一个块是否是Hebbian块。

        返回:
        bool, 如果最后一个块是Hebbian块，则返回True，否则返回False
        """
        return self.blocks[-1].is_hebbian()

    def get_lr(self) -> float:
        """
        获取最后一个Hebbian块的学习率。

        返回:
        float, 最后一个Hebbian块的学习率
        """
        if self.train_blocks:  # 如果有训练块
            for i in reversed(self.train_blocks):  # 从后往前遍历训练块
                if self.blocks[-i].is_hebbian():  # 如果是Hebbian块
                    return self.blocks[-i].get_lr()  # 返回学习率
        if self.blocks[0].is_hebbian():  # 如果第一个块是Hebbian块
            return self.blocks[0].get_lr()  # 返回学习率
        return 0  # 如果没有Hebbian块，返回0

    def radius(self, layer=None) -> str:
        """
        获取第一个Hebbian块的半径。

        参数:
        layer: int, 层的id，如果提供，则返回该层的半径，否则返回第一个Hebbian块的半径

        返回:
        str, 第一个Hebbian块的半径
        """
        if layer is not None:  # 如果提供了层的id
            return self.blocks[layer].radius()  # 返回该层的半径
        if self.train_blocks:  # 如果有训练块
            r = []  # 初始化半径列表
            for i in reversed(self.train_blocks):  # 从后往前遍历训练块
                if self.blocks[i].is_hebbian():  # 如果是Hebbian块
                    r.append(self.blocks[i].radius())  # 将半径添加到列表
            return '\n ************************************************************** \n'.join(r)  # 返回半径列表
        if self.blocks[0].is_hebbian():  # 如果第一个块是Hebbian块
            return self.blocks[0].radius()  # 返回半径
        return ''  # 如果没有Hebbian块，返回空字符串

    def convergence(self) -> str:
        """
        获取最后一个Hebbian块的收敛情况。

        返回:
        str, 最后一个Hebbian块的收敛情况
        """
        for i in range(1, len(self.blocks) + 1):  # 从后往前遍历每个块
            if self.blocks[-i].is_hebbian():  # 如果是Hebbian块
                return self.blocks[-i].layer.convergence()  # 返回收敛情况
        return 0, 0  # 如果没有Hebbian块，返回(0, 0)

    def reset(self):
        """
        重置第一个Hebbian块。
        """
        if self.blocks[0].is_hebbian():  # 如果第一个块是Hebbian块
            self.blocks[0].layer.reset()  # 重置第一个块

    def generator_block(self):
        """
        生成一个块的生成器，可以用于遍历所有的块。

        返回:
        generator, 块的生成器
        """
        for id, block in enumerate(self.blocks):  # 遍历每个块
            yield id, block  # 返回块的id和块

    def update(self):
        """
        更新所有的训练块。
        """
        for block in self.train_blocks:  # 遍历每个训练块
            self.get_block(block).update()  # 更新块

    def __str__(self):
        """
        返回模型的字符串表示。

        返回:
        str, 模型的字符串表示
        """
        for _, block in self.generator_block():  # 遍历每个块
            block.__str__()  # 返回块的字符串表示

    def train(self, mode=True, blocks=[]):
        """
        设置模型的训练模式和训练块。

        参数:
        mode: bool, 训练模式，如果为True，则为训练模式，如果为False，则为预测模式
        blocks: list, 训练块的列表
        """
        self.training = mode  # 设置训练模式
        self.train_blocks = blocks  # 设置训练块

        for param in self.parameters():  # 遍历每个参数
            param.requires_grad = False  # 设置参数不需要梯度
        for _, block in self.generator_block():  # 遍历每个块
            block.eval()  # 设置块为评估模式

        for block in blocks:  # 遍历每个训练块
            module = self.get_block(block)  # 获取块

            module.train(mode)  # 设置块的训练模式
            for param in module.parameters():  # 遍历块的每个参数
                param.requires_grad = True  # 设置参数需要梯度


class HebbianOptimizer:
    def __init__(self, model):
        """
        HebbianOptimizer是一个自定义的优化器，特别将无监督层的权重更新委托给这些层本身。

        参数:
        model (torch.nn.Module): Pytorch模型
        """
        self.model = model
        self.param_groups = []

    @torch.no_grad()
    def step(self, *args):
        """
        执行一步优化。这个方法首先遍历模型中的所有块，如果块是Hebbian块（即无监督层），则调用该块的`update`方法进行更新。
        这个方法使用了`torch.no_grad()`装饰器，表示在执行这个方法时不需要计算梯度，这是因为Hebbian学习规则通常不需要梯度下降。
        """
        loss = None

        for block in self.model.blocks:
            if block.is_hebbian():
                block.update(*args)

    def zero_grad(self):
        """
        这个方法什么也没做，因为在Hebbian学习中，我们不需要重置梯度。
        """
        pass


class AggregateOptim:
    def __init__(self, optimizers):
        """
        自定义优化器，将多个优化器聚合在一起以同时运行。

        参数:
        optimizers (List[torch.autograd.optim.Optimizer]): 需要同时调用的优化器列表
        """
        self.optimizers = optimizers
        self.param_groups = []
        for optim in self.optimizers:
            self.param_groups.extend(optim.param_groups)

    def __repr__(self):
        """
        返回优化器的字符串表示。

        返回:
        str, 优化器的字符串表示
        """
        representations = []
        for optim in self.optimizers:
            representations.append(repr(optim))
        return '\n'.join(representations)

    def step(self):
        """
        执行一步优化，调用所有优化器的step方法。
        """
        for optim in self.optimizers:
            optim.step()

    def zero_grad(self):
        """
        重置所有优化器的梯度。
        """
        for optim in self.optimizers:
            optim.zero_grad()
