import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def leakyrelu(x, leak=0.01):
    """PyTorch版本的LeakyReLU激活函数"""
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

def bst_transformer(queries, keys, values, keys_length, heads, index, max_length, use_position_embedding=True):
    """
    实现BST Transformer层
    
    参数:
        queries (torch.Tensor): 目标feed+历史行为序列, shape=(B, T, dk)
        keys (torch.Tensor): 在BST中, 同queries
        values (torch.Tensor): 在BST中, 同queries
        keys_length (torch.Tensor): 历史行为队列长度+1, 目的为生成mask, shape=(B,)
        heads (int): head数量
        index (int): transformer层的序号
        max_length: 序列长度T的最大值
        use_position_embedding (bool): 是否使用位置编码
    """
    # 获取维度信息
    d_k = queries.size(-1)
    d_model = d_k
    batch_size, seq_len = queries.size(0), queries.size(1)
    # 位置编码
    if use_position_embedding:
        # 创建位置编码表
        position_embedding_table = nn.Parameter(
            torch.zeros(max_length, d_k), requires_grad=True
        )
        nn.init.xavier_uniform_(position_embedding_table)
        # 获取位置索引并查找嵌入
        position_ids = torch.arange(seq_len, device=queries.device).unsqueeze(0)  # (1, T)
        position_emb = position_embedding_table[position_ids]  # (1, T, d_k)
        # 添加位置编码
        queries = queries + position_emb
        keys = keys + position_emb
    # 投影矩阵
    w_q = nn.Parameter(torch.zeros(heads, d_k, d_model))
    w_k = nn.Parameter(torch.zeros(heads, d_k, d_model))
    w_v = nn.Parameter(torch.zeros(heads, d_k, d_model))
    w_o = nn.Parameter(torch.zeros(heads * d_model, d_model))
    # 初始化权重
    nn.init.xavier_uniform_(w_q)
    nn.init.xavier_uniform_(w_k)
    nn.init.xavier_uniform_(w_v)
    nn.init.xavier_uniform_(w_o)
    # 多头注意力计算
    # (batch, heads, T, d_model)
    Q = torch.einsum("bik,hkj->bhij", queries, w_q)
    K = torch.einsum("bik,hkj->bhij", keys, w_k)
    V = torch.einsum("bik,hkj->bhij", values, w_v)
    # 转置K用于矩阵乘法
    K_T = K.transpose(2, 3)  # (batch, heads, d_model, T)
    # 创建掩码
    keys_mask = torch.arange(queries.size(1), device=queries.device).expand(batch_size, -1)
    keys_mask = keys_mask >= keys_length.unsqueeze(1)  # (batch, T)
    keys_mask = keys_mask.to(queries.dtype) * (-2**32 + 1)
    keys_mask = keys_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, T, 1)
    # 计算注意力得分
    attn_scores = torch.matmul(Q, K_T) / math.sqrt(d_k)  # (batch, heads, T, T)
    attn_scores = attn_scores + keys_mask  # 应用掩码
    attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, T, T)
    context = torch.matmul(attn_weights, V)  # (batch, heads, T, d_model)
    # 合并多头结果
    context = context.transpose(1, 2).contiguous()  # (batch, T, heads, d_model)
    context = context.view(batch_size, seq_len, -1)  # (batch, T, heads*d_model)
    output = torch.matmul(context, w_o)  # (batch, T, d_model)
    # 第一个残差连接和层归一化
    residual = output + queries
    output = F.layer_norm(residual, [d_model])
    # 前馈网络
    ffn = nn.Linear(d_model, d_model).to(queries.device)
    output = leakyrelu(ffn(output))
    # 第二个残差连接和层归一化
    residual = output + residual
    output = F.layer_norm(residual, [d_model])
    
    return output

if __name__ == "__main__":
    # 测试代码
    torch.manual_seed(42)
    # B=2, T=3, d_k=4
    fake_queries = torch.randn(2, 3, 4)
    fake_keys_length = torch.tensor([0, 1])
    out = bst_transformer(
        queries=fake_queries,
        keys=fake_queries,
        values=fake_queries,
        keys_length=fake_keys_length,
        heads=3,
        index=0,
        max_length=5,
        use_position_embedding=True
    )
    print(out)