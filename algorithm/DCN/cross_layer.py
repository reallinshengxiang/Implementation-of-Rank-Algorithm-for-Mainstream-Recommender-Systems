import torch
import torch.nn as nn

def cross_layer(x0, xl, index):
    """
        DCN模型的Cross Layer实现
    Args:
        x0 (tensor): Cross Layer的原始输入
        xl (tensor): Cross Layer上一层的输出
        index (int): Cross Layer的序号
    Returns:
        tensor, 维度与x0一致
    """
    dimension = x0.shape[-1]
    # 创建可训练参数
    wl = nn.Parameter(torch.zeros(dimension, 1), requires_grad=True)
    bl = nn.Parameter(torch.zeros(dimension, 1), requires_grad=True)
    # 初始化参数
    nn.init.xavier_normal_(wl)
    nn.init.zeros_(bl)
    # 计算输出
    xl_wl = torch.matmul(xl, wl)  # (batch, d) * (d, 1) = (batch, 1)
    x0_xl_wl = torch.mul(x0, xl_wl)  # (batch, d) * (batch, 1) = (batch, d)
    output = x0_xl_wl + bl.t() + xl  # 加上偏置并与xl残差连接
    return output
