"""
    [1] Wang, R., Fu, B., Fu, G., & Wang, M. (2017, August).
    Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17 (p. 12). ACM
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

# DCN模型的Cross Layer实现
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
    # 将参数移至与输入相同的设备
    device = x0.device
    wl = wl.to(device)
    bl = bl.to(device)
    # 计算输出
    xl_wl = torch.matmul(xl, wl)  # (batch, d) * (d, 1) = (batch, 1)
    x0_xl_wl = torch.mul(x0, xl_wl)  # (batch, d) * (batch, 1) = (batch, d)
    output = x0_xl_wl + bl.t() + xl  # 加上偏置并与xl残差连接
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

# DCN模型
class DCNModel(nn.Module):
    def __init__(self, vocab_dir, hidden_units=[512, 256, 128], num_cross_layer=1):
        super(DCNModel, self).__init__()
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
        # Cross部分
        self.num_cross_layer = num_cross_layer
        # DNN部分
        dnn_layers = []
        input_dim = self.input_dim
        for hidden_dim in hidden_units:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            dnn_layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.dnn = nn.Sequential(*dnn_layers)
        # 输出层
        self.output_layer = nn.Linear(self.input_dim + hidden_units[-1], 1)
        
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
        # Cross部分
        cross_vec = concat_all
        for i in range(self.num_cross_layer):
            cross_vec = cross_layer(x0=concat_all, xl=cross_vec, index=i)
        # DNN部分
        dnn_vec = self.dnn(concat_all)
        # 组合Cross和DNN的输出
        output = torch.cat([cross_vec, dnn_vec], dim=1)
        logit = self.output_layer(output)
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
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
    model = DCNModel(
        vocab_dir=args.vocabulary_dir,
        hidden_units=hidden_units,
        num_cross_layer=args.num_cross_layer
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
    parser = argparse.ArgumentParser(description='DCN Model Training')
    # 训练参数
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where output files are saved")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, required=True, help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--hidden_units", type=str, default="512,256,128", help="Comma-separated list of number of units in each hidden layer of the dnn part")
    parser.add_argument("--num_cross_layer", type=int, default=1, help="Numbers of cross layer")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    main(args)