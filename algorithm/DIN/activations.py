import torch
import torch.nn as nn
import torch.nn.functional as F

class PReLU(nn.Module):
    """
    PyTorch实现的PReLU激活函数
    """
    def __init__(self, num_parameters=1, init=1.0, name=""):
        """
        Args:
            num_parameters (int): 需要学习的alpha参数数量，通常为输入张量的最后一维大小
            init (float): alpha的初始值
            name (str): 用于命名参数的后缀
        """
        super(PReLU, self).__init__()
        self.name = name
        self.num_parameters = num_parameters
        self.alpha = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        
    def forward(self, x):
        """
        前向传播计算
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            torch.Tensor: 经过PReLU激活后的张量
        """
        return torch.max(torch.zeros_like(x), x) + self.alpha * torch.min(torch.zeros_like(x), x)

class Dice(nn.Module):
    """
    PyTorch实现的Dice激活函数
    """
    def __init__(self, num_features, name="", eps=1e-9):
        """
        Args:
            num_features (int): 输入特征的维度，用于BatchNorm操作
            name (str): 用于命名参数的后缀
            eps (float): 防止数值不稳定的小常数
        """
        super(Dice, self).__init__()
        self.name = name
        self.eps = eps
        self.alpha = nn.Parameter(torch.Tensor(num_features).fill_(1.0))
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=0.01, affine=False)
        
    def forward(self, x):
        """
        前向传播计算
        Args:
            x (torch.Tensor): 输入张量
        Returns:
            torch.Tensor: 经过Dice激活后的张量
        """
        # 保存原始形状用于后续reshape
        input_shape = x.shape
        # 如果输入张量维度大于2，展平为二维进行BatchNorm操作
        if x.dim() > 2:
            x = x.view(-1, input_shape[-1])
        # 计算标准化和sigmoid
        x_normed = self.bn(x)
        p = torch.sigmoid(x_normed)
        # 恢复原始形状
        if x.dim() > 2:
            p = p.view(input_shape)
        # 计算Dice输出
        return x * p + self.alpha * x * (1 - p)