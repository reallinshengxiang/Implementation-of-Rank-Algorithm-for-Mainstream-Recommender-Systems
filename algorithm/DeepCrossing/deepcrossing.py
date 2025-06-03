"""
    [1] Shan, Ying, et al. "Deep crossing: Web-scale modeling without manually crafted combinatorial features."
    Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM, 2016.
"""

import os
import sys
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

# DeepCrossing模型的残差单元
def residual_unit(input_tensor, internal_dim, index):
    """
    Args:
        input_tensor (torch.Tensor): 输入张量
        internal_dim (int): 网络内部维度
        index (int): 层索引
    Returns:
        torch.Tensor: 输出张量，维度与输入相同
    """
    output_dim = input_tensor.size(-1)
    device = input_tensor.device  # 获取输入张量的设备
    # 残差路径 - 使用在正确设备上创建的层
    x = nn.Linear(output_dim, internal_dim).to(device)(input_tensor)
    x = nn.ReLU()(x)
    x = nn.Linear(internal_dim, output_dim).to(device)(x)
    # 残差连接
    output = nn.ReLU()(input_tensor + x)
    return output

# 微信数据集处理
class WechatDataset(Dataset):
    def __init__(self, data_path, vocab_dir, max_seq_length=50):
        self.data = pd.read_parquet(data_path)
        self.vocab_dir = vocab_dir
        self.max_seq_length = max_seq_length
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
        dense = torch.tensor([row.get(f, 0.0) for f in self.dense_features], dtype=torch.float32)
        # 处理类别特征
        category = {}
        for col in self.category_features:
            if col in self.vocab_indices and row.get(col) in self.vocab_indices[col]:
                category[col] = torch.tensor(self.vocab_indices[col][row[col]], dtype=torch.long)
            else:
                category[col] = torch.tensor(0, dtype=torch.long)  # 未知类别
        # 目标标签
        label = torch.tensor(row.get("read_comment", 0.0), dtype=torch.float32)
        return {
            'dense': dense,
            'category': category,
            'label': label
        }

# DeepCrossing模型
class DeepCrossingModel(nn.Module):
    def __init__(self, vocab_dir, residual_internal_dim=128, residual_network_num=1):
        super(DeepCrossingModel, self).__init__()
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
        })
        # 计算输入维度
        category_emb_dim = 16 + 2 + 4 + 4 + 4 + 4  # 用户ID + 设备 + 作者ID + 背景音乐ID + 歌手ID + 标签
        self.input_dim = self.num_dense_features + category_emb_dim
        # 残差网络部分
        self.residual_internal_dim = residual_internal_dim
        self.residual_network_num = residual_network_num
        # 输出层
        self.output_layer = nn.Linear(self.input_dim, 1)
        
    def _load_vocabulary(self, vocab_dir, filename):
        vocab_path = os.path.join(vocab_dir, filename)
        if not os.path.exists(vocab_path):
            return []
        with open(vocab_path, 'r') as f:
            return [line.strip() for line in f]
    
    def forward(self, dense, category):
        # 处理类别特征
        category_emb = []
        for col, embedding in self.embeddings.items():
            if col in category:
                category_emb.append(embedding(category[col]))
        
        category_emb = torch.cat(category_emb, dim=1)  # (batch_size, category_emb_dim)
        # 连接所有特征
        concat_all = torch.cat([dense, category_emb], dim=1)
        # 残差网络部分
        net = concat_all
        for i in range(self.residual_network_num):
            net = residual_unit(net, self.residual_internal_dim, index=i)
        # 输出层
        logit = self.output_layer(net)
        probability = torch.sigmoid(logit)
        return probability, logit

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        # 移至设备
        dense = batch['dense'].to(device)
        label = batch['label'].to(device)
        # 处理类别特征
        category = {k: v.to(device) for k, v in batch['category'].items()}
        # 前向传播
        optimizer.zero_grad()
        probabilities, logits = model(dense, category)
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
            label = batch['label'].to(device)
            # 处理类别特征
            category = {k: v.to(device) for k, v in batch['category'].items()}
            # 前向传播
            probabilities, logits = model(dense, category)
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
    train_dataset = WechatDataset(args.train_data, args.vocabulary_dir)
    eval_dataset = WechatDataset(args.eval_data, args.vocabulary_dir)
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
    model = DeepCrossingModel(
        vocab_dir=args.vocabulary_dir,
        residual_internal_dim=args.residual_internal_dim,
        residual_network_num=args.residual_network_num
    ).to(device)
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
            # 处理类别特征
            category = {k: v.to(device) for k, v in batch['category'].items()}
            # 前向传播
            probabilities, _ = model(dense, category)
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
    parser = argparse.ArgumentParser(description='DeepCrossing Model Training')
    # 训练参数
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where output files are saved")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, required=True, help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--residual_internal_dim", type=int, default=128, help="Residual module internal dimension")
    parser.add_argument("--residual_network_num", type=int, default=1, help="Numbers of residual networks")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    main(args)