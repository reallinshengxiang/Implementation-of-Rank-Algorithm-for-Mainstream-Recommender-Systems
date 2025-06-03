import torch
import torch.nn as nn
import torch.nn.functional as F

def din_attention(query, keys, keys_length, is_softmax=False):
    """
    实现DIN模型中的attention模块
    
    Args:
        query (torch.Tensor): 目标, shape=(B, H)
        keys (torch.Tensor): 历史行为序列, shape=(B, T, H)
        keys_length (torch.Tensor): 历史行为队列长度, 目的为生成mask, shape=(B,)
        is_softmax (bool): attention权重是否使用softmax激活
        
    Returns:
        torch.Tensor: weighted sum pooling结果
    """
    batch_size, seq_len, embedding_dim = keys.size()
    # 扩展query维度与keys匹配
    query = query.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, H)
    # 特征交叉
    cross = torch.cat([query, keys, query - keys, query * keys], dim=-1)  # (B, T, 4*H)
    # 注意力打分网络
    att_net = nn.Sequential(
        nn.Linear(4 * embedding_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    # 计算注意力权重
    output_weight = att_net(cross)  # (B, T, 1)
    # 生成mask
    keys_mask = torch.arange(seq_len, device=keys.device).expand(batch_size, seq_len) < keys_length.unsqueeze(1)
    keys_mask = keys_mask.unsqueeze(-1)  # (B, T, 1)
    if is_softmax:
        # 对mask区域填充极小值
        paddings = torch.ones_like(output_weight) * (-2 ** 32 + 1)
        output_weight = torch.where(keys_mask, output_weight, paddings)
        # scale以稳定梯度
        output_weight = output_weight / (embedding_dim ** 0.5)
        # softmax归一化
        output_weight = F.softmax(output_weight, dim=1)  # (B, T, 1)
    else:
        # 不使用softmax时直接应用mask
        keys_mask = keys_mask.float()
        output_weight = output_weight * keys_mask  # (B, T, 1)
    # 加权求和
    outputs = torch.bmm(output_weight.transpose(1, 2), keys)  # (B, 1, T) * (B, T, H) = (B, 1, H)
    outputs = outputs.squeeze(1)  # (B, H)
    return outputs

# 测试代码
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    # B=2, T=3, H=4
    fake_keys = torch.randn(2, 3, 4)
    fake_query = torch.randn(2, 4)
    fake_keys_length = torch.tensor([0, 1]) 
    # 测试不使用softmax的情况
    attention_out1 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=False)
    print("不使用softmax激活:")
    print(attention_out1)
    # 测试使用softmax的情况
    attention_out2 = din_attention(fake_query, fake_keys, fake_keys_length, is_softmax=True)
    print("使用softmax激活:")
    print(attention_out2)