"""
    [1] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, Wenwu Ou. 2019.
    Behavior Sequence Transformer for E-commerce Recommendation in Alibaba.

    [2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In NIPS. 5998–6008.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import argparse

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 加载词汇表
def load_vocabulary(vocab_file):
    if not os.path.exists(vocab_file):
        return []
    with open(vocab_file, 'r') as f:
        return [line.strip() for line in f]

# LeakyReLU激活函数
def leakyrelu(x, leak=0.01):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)

# BST Transformer层
class BSTTransformer(nn.Module):
    def __init__(self, d_model, nhead, max_len, dropout=0.1):
        super(BSTTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        # 位置编码
        self.position_embedding = nn.Embedding(max_len, d_model)
        # 多头注意力投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # 层归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, queries, keys, values, key_padding_mask=None):
        # 添加位置编码
        batch_size, seq_len, _ = queries.size()
        pos_indices = torch.arange(seq_len, device=queries.device).expand(batch_size, -1)
        queries = queries + self.position_embedding(pos_indices)
        keys = keys + self.position_embedding(pos_indices)
        # 计算注意力
        q = self.w_q(queries).view(batch_size, seq_len, self.nhead, -1).transpose(1, 2)
        k = self.w_k(keys).view(batch_size, seq_len, self.nhead, -1).transpose(1, 2)
        v = self.w_v(values).view(batch_size, seq_len, self.nhead, -1).transpose(1, 2)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        # 应用掩码
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # 第一个残差连接和层归一化
        output1 = self.norm1(queries + self.dropout(self.w_o(context)))
        # 前馈网络
        output2 = self.ffn(output1)
        # 第二个残差连接和层归一化
        output = self.norm2(output1 + self.dropout(output2))
        return output

# 微信数据集处理
class WechatDataset(Dataset):
    def __init__(self, data_path, vocab_dir, max_seq_length=50):
        self.data = pd.read_parquet(data_path)
        self.vocab_dir = vocab_dir
        self.max_seq_length = max_seq_length
        # 加载词汇表
        self.vocabs = {
            'userid': load_vocabulary(os.path.join(vocab_dir, 'userid.txt')),
            'feedid': load_vocabulary(os.path.join(vocab_dir, 'feedid.txt')),
            'device': load_vocabulary(os.path.join(vocab_dir, 'device.txt')),
            'authorid': load_vocabulary(os.path.join(vocab_dir, 'authorid.txt')),
            'bgm_song_id': load_vocabulary(os.path.join(vocab_dir, 'bgm_song_id.txt')),
            'bgm_singer_id': load_vocabulary(os.path.join(vocab_dir, 'bgm_singer_id.txt')),
            'manual_tag_list': load_vocabulary(os.path.join(vocab_dir, 'manual_tag_id.txt')),
        }
        # 创建词汇表索引映射
        self.vocab_indices = {k: {v: i for i, v in enumerate(vocab)} for k, vocab in self.vocabs.items()}
        # 连续特征列
        self.dense_features = [
            "videoplayseconds", "u_read_comment_7d_sum", "u_like_7d_sum", 
            "u_click_avatar_7d_sum", "u_forward_7d_sum", "u_comment_7d_sum", 
            "u_follow_7d_sum", "u_favorite_7d_sum", "i_read_comment_7d_sum", 
            "i_like_7d_sum", "i_click_avatar_7d_sum", "i_forward_7d_sum", 
            "i_comment_7d_sum", "i_follow_7d_sum", "i_favorite_7d_sum", 
            "c_user_author_read_comment_7d_sum"
        ]
        # 类别特征列
        self.category_features = [
            "userid", "device", "authorid", "bgm_song_id", "bgm_singer_id", "manual_tag_list"
        ]
        # 序列特征列
        self.sequence_features = ["feedid"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 处理连续特征
        dense = torch.tensor([row.get(f, 0.0) for f in self.dense_features], dtype=torch.float32)
        # 处理类别特征
        category = {}
        for col in self.category_features:
            if col in self.vocab_indices and row.get(col) in self.vocab_indices[col]:
                category[col] = torch.tensor(self.vocab_indices[col][row[col]], dtype=torch.long)
            else:
                category[col] = torch.tensor(0, dtype=torch.long)  # 未知类别
        # 处理序列特征
        seq_feedid = row.get("feedid", [])
        if not isinstance(seq_feedid, list):
            seq_feedid = [seq_feedid]
        # 截断或填充序列
        seq_length = min(len(seq_feedid), self.max_seq_length)
        seq_indices = torch.zeros(self.max_seq_length, dtype=torch.long)
        for i in range(seq_length):
            if seq_feedid[i] in self.vocab_indices["feedid"]:
                seq_indices[i] = self.vocab_indices["feedid"][seq_feedid[i]]
        # 目标标签
        label = torch.tensor(row.get("read_comment", 0.0), dtype=torch.float32)
        return {
            'dense': dense,
            'category': category,
            'seq_feedid': seq_indices,
            'seq_length': seq_length,
            'label': label
        }

# BST模型
class BSTModel(nn.Module):
    def __init__(self, vocab_dir, hidden_units=[512, 256, 128], 
                 dropout_rate=0.1, batch_norm=True,
                 d_model=16, nhead=4, num_transformer_blocks=1,
                 max_seq_length=50, pooling_method='sum'):
        super(BSTModel, self).__init__()
        # 加载词汇表大小
        self.vocab_sizes = {
            'userid': len(load_vocabulary(os.path.join(vocab_dir, 'userid.txt'))) + 1,
            'feedid': len(load_vocabulary(os.path.join(vocab_dir, 'feedid.txt'))) + 1,
            'device': len(load_vocabulary(os.path.join(vocab_dir, 'device.txt'))) + 1,
            'authorid': len(load_vocabulary(os.path.join(vocab_dir, 'authorid.txt'))) + 1,
            'bgm_song_id': len(load_vocabulary(os.path.join(vocab_dir, 'bgm_song_id.txt'))) + 1,
            'bgm_singer_id': len(load_vocabulary(os.path.join(vocab_dir, 'bgm_singer_id.txt'))) + 1,
            'manual_tag_list': len(load_vocabulary(os.path.join(vocab_dir, 'manual_tag_id.txt'))) + 1,
        }
        # 连续特征数量
        self.num_dense_features = 16  # 硬编码，根据实际情况调整
        # 嵌入层
        self.embeddings = nn.ModuleDict({
            'userid': nn.Embedding(self.vocab_sizes['userid'], 16),
            'device': nn.Embedding(self.vocab_sizes['device'], 2),
            'authorid': nn.Embedding(self.vocab_sizes['authorid'], 4),
            'bgm_song_id': nn.Embedding(self.vocab_sizes['bgm_song_id'], 4),
            'bgm_singer_id': nn.Embedding(self.vocab_sizes['bgm_singer_id'], 4),
            'manual_tag_list': nn.Embedding(self.vocab_sizes['manual_tag_list'], 4),
            'feedid': nn.Embedding(self.vocab_sizes['feedid'], 16),
        })
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            BSTTransformer(d_model=16, nhead=nhead, max_len=max_seq_length+1, dropout=dropout_rate)
            for _ in range(num_transformer_blocks)
        ])
        # DNN层
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.pooling_method = pooling_method
        # 计算DNN输入维度
        category_emb_dim = 16 + 2 + 4 + 4 + 4 + 4  # 用户ID + 设备 + 作者ID + 背景音乐ID + 歌手ID + 标签
        transformer_dim = 16  # feedid嵌入维度
        # DNN层
        dnn_layers = []
        input_dim = self.num_dense_features + category_emb_dim + transformer_dim
        for hidden_dim in hidden_units:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            dnn_layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropout_rate > 0:
                dnn_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, dense, category, seq_feedid, seq_length):
        # 处理类别特征
        category_emb = []
        for col, embedding in self.embeddings.items():
            if col in category:
                category_emb.append(embedding(category[col]))
        category_emb = torch.cat(category_emb, dim=1)  # (batch_size, category_emb_dim)
        # 处理序列特征
        seq_emb = self.embeddings['feedid'](seq_feedid)  # (batch_size, seq_length, embedding_dim)
        # 创建掩码
        max_len = seq_feedid.size(1)
        mask = torch.arange(max_len, device=seq_feedid.device).expand(len(seq_length), max_len) >= seq_length.unsqueeze(1)
        # Transformer处理
        transformer_output = seq_emb
        for transformer in self.transformer_blocks:
            transformer_output = transformer(
                queries=transformer_output,
                keys=transformer_output,
                values=transformer_output,
                key_padding_mask=mask
            )
        # 池化操作
        if self.pooling_method == 'sum':
            transformer_output = torch.sum(transformer_output, dim=1)
        else:  # mean
            transformer_output = torch.sum(transformer_output, dim=1) / seq_length.unsqueeze(1).float()
        # 连接所有特征
        all_features = torch.cat([dense, category_emb, transformer_output], dim=1)
        # 通过DNN获取预测
        logits = self.dnn(all_features)
        probabilities = torch.sigmoid(logits)
        return probabilities, logits

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        # 移至设备
        dense = batch['dense'].to(device)
        seq_feedid = batch['seq_feedid'].to(device)
        seq_length = batch['seq_length'].to(device)
        label = batch['label'].to(device)
        # 处理类别特征
        category = {k: v.to(device) for k, v in batch['category'].items()}
        # 前向传播
        optimizer.zero_grad()
        probabilities, logits = model(dense, category, seq_feedid, seq_length)
        # 计算损失
        loss = criterion(logits.squeeze(), label)
        # 反向传播
        loss.backward()
        optimizer.step()
        # 收集统计信息
        total_loss += loss.item()
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(probabilities.squeeze().cpu().detach().numpy())
    # 计算指标
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}')
    return avg_loss, accuracy, auc

# 评估函数
def evaluate(model, eval_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Epoch {epoch} Evaluation'):
            # 移至设备
            dense = batch['dense'].to(device)
            seq_feedid = batch['seq_feedid'].to(device)
            seq_length = batch['seq_length'].to(device)
            label = batch['label'].to(device)
            # 处理类别特征
            category = {k: v.to(device) for k, v in batch['category'].items()}
            # 前向传播
            probabilities, logits = model(dense, category, seq_feedid, seq_length)
            # 计算损失
            loss = criterion(logits.squeeze(), label)  
            # 收集统计信息
            total_loss += loss.item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(probabilities.squeeze().cpu().numpy())
    # 计算指标
    avg_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch}, Eval Loss: {avg_loss:.4f}, Eval Accuracy: {accuracy:.4f}, Eval AUC: {auc:.4f}')
    return avg_loss, accuracy, auc, all_preds

# 主函数
def main(args):
    # 设置随机种子
    set_seed(42)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 创建数据集
    print('Loading data...')
    train_dataset = WechatDataset(args.train_data, args.vocabulary_dir, args.sequence_max_length)
    eval_dataset = WechatDataset(args.eval_data, args.vocabulary_dir, args.sequence_max_length)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    # 创建模型
    print('Creating model...')
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
    model = BSTModel(
        vocab_dir=args.vocabulary_dir,
        hidden_units=hidden_units,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
        d_model=16,
        nhead=args.num_transformer_heads,
        num_transformer_blocks=args.num_transformer_block,
        max_seq_length=args.sequence_max_length,
        pooling_method=args.pooling_method
    ).to(device)
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # 训练和评估
    best_auc = 0.0
    print('Start training...')
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_loss, train_acc, train_auc = train(model, train_loader, criterion, optimizer, device, epoch)     
        # 评估
        eval_loss, eval_acc, eval_auc, predictions = evaluate(model, eval_loader, criterion, device, epoch)
        # 保存最佳模型
        if eval_auc > best_auc:
            best_auc = eval_auc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f'Model saved at epoch {epoch} with best AUC: {best_auc:.4f}')
        # 保存检查点
        if epoch % args.save_checkpoints_steps == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_auc': eval_auc
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pth'))
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    model.eval()
    # 生成预测结果
    print('Generating predictions...')
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Generating predictions'):
            # 移至设备
            dense = batch['dense'].to(device)
            seq_feedid = batch['seq_feedid'].to(device)
            seq_length = batch['seq_length'].to(device)
            # 处理类别特征
            category = {k: v.to(device) for k, v in batch['category'].items()}
            # 前向传播
            probabilities, _ = model(dense, category, seq_feedid, seq_length)
            all_preds.extend(probabilities.squeeze().cpu().numpy())
    # 保存预测结果
    test_df = pd.read_parquet(args.eval_data)
    result_df = pd.DataFrame({
        'read_comment': test_df['read_comment'].values,
        'probability': all_preds
    })
    result_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f'Predictions saved to {os.path.join(args.output_dir, "predictions.csv")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BST Model Training')
    # 训练参数
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where output files are saved")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, required=True, help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_units", type=str, default="512,256,128", help="Hidden units for DNN")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_norm", action='store_true', help="Use batch normalization")
    parser.add_argument("--sequence_max_length", type=int, default=50, help="Maximal length of user behavior sequence")
    parser.add_argument("--num_transformer_block", type=int, default=1, help="Number of transformer blocks")
    parser.add_argument("--num_transformer_heads", type=int, default=4, help="Number of heads in transformer")
    parser.add_argument("--pooling_method", type=str, choices=["sum", "mean"], default="sum", help="Pooling method")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    main(args)