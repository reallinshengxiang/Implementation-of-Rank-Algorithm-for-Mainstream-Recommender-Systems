import torch
import torch.nn as nn

def residual_unit(input_tensor, internal_dim=None, index=None):
    """
    DeepCrossing模型的残差网络单元
    Args:
        input_tensor (torch.Tensor): 输入张量
        internal_dim (int): 网络内部维度
        index (int): 层索引
    Returns:
        torch.Tensor: 输出张量，维度与输入相同
    """
    output_dim = input_tensor.size(-1)
    # 残差路径
    x = nn.Linear(output_dim, internal_dim)(input_tensor)
    x = nn.ReLU()(x)
    x = nn.Linear(internal_dim, output_dim)(x)
    # 残差连接
    output = nn.ReLU()(input_tensor + x)
    return output