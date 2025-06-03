import torch

def leakyrelu(x, leak=0.01):
    """
    手动实现 LeakyReLU 激活函数
    Args:
        x (Tensor): 输入张量
        leak (float): 负斜率值 (默认: 0.01)
    Returns:
        Tensor: 经过 LeakyReLU 处理后的张量
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

