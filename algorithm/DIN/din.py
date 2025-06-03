"""
    [1] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui
    Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through
    rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference
    on Knowledge Discovery & Data Mining. ACM, 1059–1068.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import argparse

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# DIN模型中的激活函数
class Dice(nn.Module):
    def __init__(self, num_features, eps=1e-9):
        super(Dice, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        
    def forward(self, x):
        x_normed = self.bn(x)
        x_p = torch.sigmoid(x_normed)
        return self.alpha * (1.0 - x_p) * x + x_p * x

def prelu(x, alpha=1.0):
    return torch.max(torch.zeros_like(x), x) + alpha * torch.min(torch.zeros_like(x), x)

# DIN模型中的注意力机制
def din_attention(query, keys, keys_length, is_softmax=False):
    """
    DIN注意力机制实现
    
    Args:
        query: 目标向量, shape=(batch_size, embedding_dim)
        keys: 历史行为序列, shape=(batch_size, max_length, embedding_dim)
        keys_length: 历史行为序列长度, shape=(batch_size,)
        is_softmax: 是否对注意力得分使用softmax
    
    Returns:
        加权后的历史行为表示, shape=(batch_size, embedding_dim)
    """
    batch_size, max_length, embedding_dim = keys.size()
    # 扩展query维度以便与keys进行交互
    query = query.unsqueeze(1).expand_as(keys)  # (batch_size, max_length, embedding_dim)
    # 特征交叉
    cross = torch.cat([query, keys, query - keys, query * keys], dim=2)  # (batch_size, max_length, 4*embedding_dim)
    # 注意力网络
    att_net = nn.Sequential(
        nn.Linear(4 * embedding_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(keys.device)
    # 计算注意力得分
    att_score = att_net(cross).squeeze(2)  # (batch_size, max_length)
    # 创建掩码
    mask = torch.arange(max_length, device=keys.device).expand(batch_size, max_length) < keys_length.unsqueeze(1)
    if is_softmax:
        # 对未掩码的位置应用softmax
        paddings = torch.ones_like(att_score) * (-2 ** 32 + 1)
        att_score = torch.where(mask, att_score, paddings)
        att_score = att_score / (embedding_dim ** 0.5)  # 缩放以稳定梯度
        att_weight = torch.softmax(att_score, dim=1)  # (batch_size, max_length)
    else:
        # 不使用softmax时，直接应用掩码
        att_weight = att_score.masked_fill(~mask, 0.0)
    # 加权求和
    att_weight = att_weight.unsqueeze(2)  # (batch_size, max_length, 1)
    output = torch.sum(att_weight * keys, dim=1)  # (batch_size, embedding_dim)
    return output

# 微信数据集处理
class WechatDataset(Dataset):
    def __init__(self, data_path, vocab_dir):
        self.data = pd.read_parquet(data_path)
        self.vocab_dir = vocab_dir
        # 加载词汇表
        self.vocabs = {
            'userid': self._load_vocabulary('userid.txt'),
            'feedid': self._load_vocabulary('feedid.txt'),
            'device': self._load_vocabulary('device.txt'),
            'authorid': self._load_vocabulary('authorid.txt'),
            'bgm_song_id': self._load_vocabulary('bgm_song_id.txt'),
            'bgm_singer_id': self._load_vocabulary('bgm_singer_id.txt'),
            'manual_tag_list': self._load_vocabulary('manual_tag_id.txt'),
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
        # 历史行为序列特征
        self.sequence_features = ["his_read_comment_7d_seq"]
        # 目标特征
        self.target_features = ["feedid"]
        
    def _load_vocabulary(self, filename):
        vocab_path = os.path.join(self.vocab_dir, filename)
        if not os.path.exists(vocab_path):
            return []
        with open(vocab_path, 'r') as f:
            return [line.strip() for line in f]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 处理连续特征
        dense = {}
        for col in self.dense_features:
            dense[col] = torch.tensor(row.get(col, 0.0), dtype=torch.float32)
        # 处理类别特征
        category = {}
        for col in self.category_features:
            if col in self.vocab_indices and row.get(col) in self.vocab_indices[col]:
                category[col] = torch.tensor(self.vocab_indices[col][row[col]], dtype=torch.long)
            else:
                category[col] = torch.tensor(0, dtype=torch.long)  # 未知类别
        # 处理历史行为序列
        sequence = {}
        for col in self.sequence_features:
            seq = row.get(col, [])
            if isinstance(seq, str):
                seq = seq.split(',')
            indices = []
            for item in seq:
                if item in self.vocab_indices.get('feedid', {}):
                    indices.append(self.vocab_indices['feedid'][item])
                else:
                    indices.append(0)  # 未知feedid
            sequence[col] = torch.tensor(indices, dtype=torch.long)
            sequence[f"{col}_length"] = torch.tensor(len(indices), dtype=torch.long)
        # 处理目标特征
        target = {}
        for col in self.target_features:
            if col in self.vocab_indices and row.get(col) in self.vocab_indices[col]:
                target[col] = torch.tensor(self.vocab_indices[col][row[col]], dtype=torch.long)
            else:
                target[col] = torch.tensor(0, dtype=torch.long)
        # 目标标签
        label = torch.tensor(row.get("read_comment", 0.0), dtype=torch.float32)
        return {
            'dense': dense,
            'category': category,
            'sequence': sequence,
            'target': target,
            'label': label
        }

# 自定义的collate_fn处理变长序列
def din_collate_fn(batch):
    """
    自定义collate_fn处理变长序列
    """
    dense_batch = {}
    category_batch = {}
    sequence_batch = {}
    target_batch = {}
    label_batch = []
    # 找出最大序列长度
    max_seq_length = 0
    for item in batch:
        for col in item['sequence']:
            if col.endswith('_length'):
                continue
            max_seq_length = max(max_seq_length, len(item['sequence'][col]))
    # 处理连续特征
    for col in batch[0]['dense']:
        dense_batch[col] = torch.stack([item['dense'][col] for item in batch])
    # 处理类别特征
    for col in batch[0]['category']:
        category_batch[col] = torch.stack([item['category'][col] for item in batch])
    # 处理目标特征
    for col in batch[0]['target']:
        target_batch[col] = torch.stack([item['target'][col] for item in batch])
    # 处理序列特征
    for col in batch[0]['sequence']:
        if col.endswith('_length'):
            sequence_batch[col] = torch.stack([item['sequence'][col] for item in batch])
        else:
            # 创建填充后的序列
            padded_seqs = []
            for item in batch:
                seq = item['sequence'][col]
                padded = torch.zeros(max_seq_length, dtype=torch.long)
                padded[:len(seq)] = seq
                padded_seqs.append(padded)
            sequence_batch[col] = torch.stack(padded_seqs)
    # 处理标签
    label_batch = torch.stack([item['label'] for item in batch])
    return {
        'dense': dense_batch,
        'category': category_batch,
        'sequence': sequence_batch,
        'target': target_batch,
        'label': label_batch
    }

# DIN模型
class DIN(nn.Module):
    def __init__(self, vocab_dir, hidden_units=None, activation='dice', 
                 dropout_rate=0.1, batch_norm=True, use_softmax=False, 
                 l2_lambda=0.2, mini_batch_aware_regularization=True):
        super(DIN, self).__init__()
        if hidden_units is None:
            hidden_units = [512, 256, 128]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_softmax = use_softmax
        self.l2_lambda = l2_lambda
        self.mini_batch_aware_regularization = mini_batch_aware_regularization
        # 加载词汇表大小
        self.vocab_sizes = {
            'userid': len(self._load_vocabulary(vocab_dir, 'userid.txt')) + 1,
            'feedid': len(self._load_vocabulary(vocab_dir, 'feedid.txt')) + 1,
            'device': len(self._load_vocabulary(vocab_dir, 'device.txt')) + 1,
            'authorid': len(self._load_vocabulary(vocab_dir, 'authorid.txt')) + 1,
            'bgm_song_id': len(self._load_vocabulary(vocab_dir, 'bgm_song_id.txt')) + 1,
            'bgm_singer_id': len(self._load_vocabulary(vocab_dir, 'bgm_singer_id.txt')) + 1,
            'manual_tag_list': len(self._load_vocabulary(vocab_dir, 'manual_tag_id.txt')) + 1,
        }
        # 连续特征数量
        self.num_dense_features = 16
        # 嵌入层
        self.embeddings = nn.ModuleDict({
            'userid': nn.Embedding(self.vocab_sizes['userid'], 16),
            'device': nn.Embedding(self.vocab_sizes['device'], 2),
            'authorid': nn.Embedding(self.vocab_sizes['authorid'], 4),
            'bgm_song_id': nn.Embedding(self.vocab_sizes['bgm_song_id'], 4),
            'bgm_singer_id': nn.Embedding(self.vocab_sizes['bgm_singer_id'], 4),
            'manual_tag_list': nn.Embedding(self.vocab_sizes['manual_tag_list'], 4),
            'feedid': nn.Embedding(self.vocab_sizes['feedid'], 16),
            'his_read_comment_7d_seq': nn.Embedding(self.vocab_sizes['feedid'], 16),
        })
        input_dim = self.num_dense_features
        # 计算类别特征嵌入的总维度
        category_dim = 0
        category_keys = ['userid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'manual_tag_list']
        for key in category_keys:
            if key in self.embeddings:
                category_dim += self.embeddings[key].embedding_dim
        input_dim += category_dim
        input_dim += self.embeddings['feedid'].embedding_dim  # 目标feedid
        input_dim += self.embeddings['his_read_comment_7d_seq'].embedding_dim  # 注意力输出
        # 全连接层
        self.fcn = nn.ModuleList()
        for i, unit in enumerate(hidden_units):
            self.fcn.append(nn.Linear(input_dim, unit))
            if activation == 'dice':
                self.fcn.append(Dice(unit))
            else:
                # 使用PReLU作为备选激活函数
                self.fcn.append(nn.PReLU())
            if batch_norm:
                self.fcn.append(nn.BatchNorm1d(unit))
            if dropout_rate > 0:
                self.fcn.append(nn.Dropout(dropout_rate))
            input_dim = unit
        self.output_layer = nn.Linear(input_dim, 1)
        
    def _load_vocabulary(self, vocab_dir, filename):
        vocab_path = os.path.join(vocab_dir, filename)
        if not os.path.exists(vocab_path):
            return []
        with open(vocab_path, 'r') as f:
            return [line.strip() for line in f]
    
    def forward(self, dense, category, sequence, target):
        # 处理连续特征
        dense_input = torch.cat([dense[col].unsqueeze(1) for col in dense], dim=1)
        # 处理类别特征
        category_emb = []
        for col, embedding in self.embeddings.items():
            if col in category:
                category_emb.append(embedding(category[col]))
        # 处理目标feed
        target_feed_emb = self.embeddings['feedid'](target['feedid'])
        # 处理历史行为序列
        seq_emb = self.embeddings['his_read_comment_7d_seq'](sequence['his_read_comment_7d_seq'])
        seq_length = sequence['his_read_comment_7d_seq_length']
        # 注意力机制
        attention_output = din_attention(target_feed_emb, seq_emb, seq_length, self.use_softmax)
        # 拼接所有特征
        concat_all = torch.cat([dense_input] + category_emb + [target_feed_emb, attention_output], dim=1)
        # 通过全连接层
        net = concat_all
        for layer in self.fcn:
            net = layer(net)
        logit = self.output_layer(net)
        probability = torch.sigmoid(logit)
        # 计算嵌入层的L2正则化项（如果启用了mini_batch_aware_regularization）
        l2_reg = 0.0
        if self.mini_batch_aware_regularization and self.l2_lambda > 0:
            # 对类别嵌入、目标feed和注意力输出计算L2范数
            embedding_vars = torch.cat([torch.cat(category_emb, dim=1), target_feed_emb, attention_output], dim=1)
            l2_reg = self.l2_lambda * torch.norm(embedding_vars, p=2, dim=1).mean()
        return probability, logit, l2_reg

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch, l2_lambda):
    model.train()
    total_loss = 0
    total_l2_loss = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        # 移至设备
        dense = {k: v.to(device) for k, v in batch['dense'].items()}
        category = {k: v.to(device) for k, v in batch['category'].items()}
        sequence = {k: v.to(device) for k, v in batch['sequence'].items()}
        target = {k: v.to(device) for k, v in batch['target'].items()}
        label = batch['label'].to(device)
        # 前向传播
        optimizer.zero_grad()
        probability, logit, l2_reg = model(dense, category, sequence, target)
        # 计算损失
        ce_loss = criterion(probability.squeeze(), label)
        loss = ce_loss + l2_reg
        # 反向传播
        loss.backward()
        optimizer.step()
        # 收集统计信息
        total_loss += loss.item()
        total_l2_loss += l2_reg.item() if l2_reg != 0 else 0
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(probability.squeeze().cpu().detach().numpy())
    # 计算指标
    avg_loss = total_loss / len(train_loader)
    avg_l2_loss = total_l2_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train L2 Loss: {avg_l2_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train AUC: {auc:.4f}')
    return avg_loss, accuracy, auc

# 评估函数
def evaluate(model, eval_loader, criterion, device, epoch, l2_lambda):
    model.eval()
    total_loss = 0
    total_l2_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Epoch {epoch} Evaluation'):
            # 移至设备
            dense = {k: v.to(device) for k, v in batch['dense'].items()}
            category = {k: v.to(device) for k, v in batch['category'].items()}
            sequence = {k: v.to(device) for k, v in batch['sequence'].items()}
            target = {k: v.to(device) for k, v in batch['target'].items()}
            label = batch['label'].to(device)
            # 前向传播
            probability, logit, l2_reg = model(dense, category, sequence, target)
            # 计算损失
            ce_loss = criterion(probability.squeeze(), label)
            loss = ce_loss + l2_reg
            # 收集统计信息
            total_loss += loss.item()
            total_l2_loss += l2_reg.item() if l2_reg != 0 else 0
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(probability.squeeze().cpu().numpy())
    # 计算指标
    avg_loss = total_loss / len(eval_loader)
    avg_l2_loss = total_l2_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    print(f'Epoch {epoch}, Eval Loss: {avg_loss:.4f}, Eval L2 Loss: {avg_l2_loss:.4f}, Eval Accuracy: {accuracy:.4f}, Eval AUC: {auc:.4f}')
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
    train_dataset = WechatDataset(args.train_data, args.vocabulary_dir)
    eval_dataset = WechatDataset(args.eval_data, args.vocabulary_dir)
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=din_collate_fn
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=din_collate_fn
    )
    # 创建模型
    print('Creating model...')
    hidden_units = [int(unit) for unit in args.hidden_units.split(',')]
    model = DIN(
        vocab_dir=args.vocabulary_dir,
        hidden_units=hidden_units,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
        use_softmax=args.use_softmax,
        l2_lambda=args.l2_lambda,
        mini_batch_aware_regularization=args.mini_batch_aware_regularization
    ).to(device)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # 训练和评估
    best_auc = 0.0
    print('Start training...')
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_loss, train_acc, train_auc = train(model, train_loader, criterion, optimizer, device, epoch, args.l2_lambda)
        # 评估
        eval_loss, eval_acc, eval_auc, predictions = evaluate(model, eval_loader, criterion, device, epoch, args.l2_lambda)
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
            dense = {k: v.to(device) for k, v in batch['dense'].items()}
            category = {k: v.to(device) for k, v in batch['category'].items()}
            sequence = {k: v.to(device) for k, v in batch['sequence'].items()}
            target = {k: v.to(device) for k, v in batch['target'].items()}
            # 前向传播
            probability, _, _ = model(dense, category, sequence, target)
            all_preds.extend(probability.squeeze().cpu().numpy())
    # 保存预测结果
    test_df = pd.read_parquet(args.eval_data)
    result_df = pd.DataFrame({
        'read_comment': test_df['read_comment'].values,
        'probability': all_preds
    })
    result_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f'Predictions saved to {os.path.join(args.output_dir, "predictions.csv")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Interest Network (DIN) Training')
    # 训练参数
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where output files are saved")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, required=True, help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_units", type=str, default="512,256,128", help="Comma-separated list of hidden units")
    parser.add_argument("--activation", type=str, default="dice", help="Activation function: dice or prelu")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Whether to use batch normalization")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--mini_batch_aware_regularization", type=bool, default=True, help="Whether to use mini-batch aware regularization")
    parser.add_argument("--l2_lambda", type=float, default=0.2, help="L2 regularization coefficient")
    parser.add_argument("--use_softmax", type=bool, default=False, help="Whether to use softmax in attention mechanism")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    main(args)