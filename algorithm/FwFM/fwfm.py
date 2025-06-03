"""
    [1] J. Pan et al. "Field-weighted factorization machines for click-through rate prediction in display advertising."
    Proc. World Wide Web Conf., pp. 1349-1357, 2018.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import argparse

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class WechatDataset(Dataset):
    """微信算法数据集加载器"""
    def __init__(self, data_path, vocab_dir, is_train=True):
        self.is_train = is_train
        # 读取Parquet文件
        self.data = pd.read_parquet(data_path)
        self.vocab_dir = vocab_dir
        # 清洗数据：替换'None'为NaN
        for feature in ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']:
            self.data[feature] = self.data[feature].astype(str).replace('None', np.nan)  # 确保转为字符串后替换
        self._load_vocab()
        self._encode_features()
        if self.is_train:
            self.labels = self.data['read_comment'].values.astype(float)  # 确保标签为浮点型

    def _load_vocab(self):
        """加载词汇表"""
        self.vocabs = {}
        for feature in ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']:
            vocab_path = os.path.join(self.vocab_dir, f'{feature}.txt')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    self.vocabs[feature] = [line.strip() for line in f.readlines()]
            else:
                self.vocabs[feature] = []  # 处理无词汇表的情况（如有）

    def _encode_features(self):
        """对特征进行编码"""
        self.encoded_features = {}
        # 预先将词汇表转换为集合（O(1)查找速度）
        vocab_sets = {feature: set(vocab) for feature, vocab in self.vocabs.items()}
        for feature in ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id', 'bgm_singer_id']:
            if feature in self.vocabs and self.vocabs[feature]:
                encoder = LabelEncoder()
                encoder.classes_ = np.array(self.vocabs[feature])  # 使用预定义词汇表
                # 填充缺失值
                series = self.data[feature]
                mode_value = series.mode(dropna=True).values[0] if not series.mode(dropna=True).empty else 'unknown'
                filled_series = series.fillna(mode_value)
                # 高效替换不在词汇表中的值（向量化操作）
                valid_mask = filled_series.isin(vocab_sets[feature])  # 生成布尔掩码
                filled_series = np.where(valid_mask, filled_series, mode_value)  # 条件替换
                # 编码特征
                self.encoded_features[feature] = encoder.transform(filled_series)
            else:
                self.encoded_features[feature] = self.data[feature].fillna(0).astype(int)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        features = {
            'userid': self.encoded_features['userid'][idx],
            'feedid': self.encoded_features['feedid'][idx],
            'device': self.encoded_features['device'][idx],
            'authorid': self.encoded_features['authorid'][idx],
            'bgm_song_id': self.encoded_features['bgm_song_id'][idx],
            'bgm_singer_id': self.encoded_features['bgm_singer_id'][idx],
        }
        if self.is_train:
            return features, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return features, torch.tensor(0, dtype=torch.float32)  # 测试时返回占位标签

class FwFM(nn.Module):
    """Field-weighted Factorization Machines模型"""
    def __init__(self, field_dims, embed_dim):
        super(FwFM, self).__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        # 一阶项
        self.linear = nn.ModuleList([
            nn.Embedding(field_dim, 1) for field_dim in field_dims
        ])
        # 嵌入层
        self.embedding = nn.ModuleList([
            nn.Embedding(field_dim, embed_dim) for field_dim in field_dims
        ])
        # 初始化嵌入层权重
        for emb in self.embedding:
            nn.init.xavier_uniform_(emb.weight)
        # 计算field pair数量
        self.num_pairs = self.num_fields * (self.num_fields - 1) // 2
        # Field-weighted参数
        self.field_weight = nn.Parameter(
            torch.randn(self.num_pairs), requires_grad=True
        )
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        x: {'feature_name': feature_tensor}
        """
        # 提取特征值
        feature_values = [x[f'userid'], x[f'feedid'], x[f'device'], 
                         x[f'authorid'], x[f'bgm_song_id'], x[f'bgm_singer_id']]
        # 一阶项计算
        linear_terms = [self.linear[i](feature_values[i]) for i in range(self.num_fields)]
        linear_sum = sum(linear_terms)
        # 二阶项计算
        embeddings = [self.embedding[i](feature_values[i]) for i in range(self.num_fields)]
        # 计算field pair交互
        quadratic_sum = 0
        pair_idx = 0
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                # 计算field i和j的交互
                interaction = torch.sum(embeddings[i] * embeddings[j], dim=1, keepdim=True)
                # 应用field pair权重
                quadratic_sum += self.field_weight[pair_idx] * interaction
                pair_idx += 1
        # 计算最终输出
        y = linear_sum + quadratic_sum + self.bias
        y = torch.sigmoid(y)
        return y.squeeze(1)

def train(model, train_loader, criterion, optimizer, device, epoch):
    """训练模型"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (features, labels) in progress_bar:
        # 将数据移至设备
        for key in features:
            features[key] = features[key].to(device)
        labels = labels.to(device)
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_description(f'Epoch {epoch}, Loss: {total_loss/(i+1):.4f}')
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    labels_list = []
    with torch.no_grad():
        for features, labels in data_loader:
            # 将数据移至设备
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    # 计算AUC和准确率
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(labels_list, predictions)
    acc = accuracy_score(labels_list, [1 if p > 0.5 else 0 for p in predictions])
    return total_loss / len(data_loader), auc, acc

def predict(model, data_loader, device):
    """模型预测"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in data_loader:
            # 将数据移至设备
            for key in features:
                features[key] = features[key].to(device)
            # 前向传播
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
    return predictions

def load_data(data_path):
    if data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")

def main():
    parser = argparse.ArgumentParser(description='FwFM for Click-Through Rate Prediction')
    # 训练参数
    parser.add_argument('--model_dir', type=str, default='./model_dir', help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='./output_dir', help='输出目录')
    parser.add_argument('--train_data', type=str, default='../../dataset/wechat_algo_data1/dataframe/train.csv', help='训练数据路径')
    parser.add_argument('--eval_data', type=str, default='../../dataset/wechat_algo_data1/dataframe/test.csv', help='评估数据路径')
    parser.add_argument('--vocabulary_dir', type=str, default='../../dataset/wechat_algo_data1/vocabulary/', help='词汇表目录')
    parser.add_argument('--num_epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='学习率')
    parser.add_argument('--embedding_dim', type=int, default=8, help='嵌入维度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用的设备')
    args = parser.parse_args()
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # 加载数据
    print("加载数据...")
    train_dataset = WechatDataset(args.train_data, args.vocabulary_dir, is_train=True)
    eval_dataset = WechatDataset(args.eval_data, args.vocabulary_dir, is_train=True)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    # 定义模型
    print("初始化模型...")
    # 获取各特征的维度
    field_dims = [
        len(train_dataset.vocabs['userid']),
        len(train_dataset.vocabs['feedid']),
        len(train_dataset.vocabs['device']),
        len(train_dataset.vocabs['authorid']),
        len(train_dataset.vocabs['bgm_song_id']),
        len(train_dataset.vocabs['bgm_singer_id']),
    ]
    model = FwFM(field_dims, args.embedding_dim).to(args.device)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 训练模型
    print("开始训练...")
    best_auc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, args.device, epoch)
        eval_loss, eval_auc, eval_acc = evaluate(model, eval_loader, criterion, args.device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval AUC: {eval_auc:.4f}, Eval Acc: {eval_acc:.4f}')
        # 保存最佳模型
        if eval_auc > best_auc:
            best_auc = eval_auc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f'模型已保存，AUC: {best_auc:.4f}')
    # 加载最佳模型进行预测
    print("加载最佳模型进行预测...")
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    # 预测并保存结果
    print("生成预测结果...")
    predictions = predict(model, eval_loader, args.device)
    # 保存预测结果
    test_df = load_data(args.eval_data)
    result_df = pd.DataFrame({
        'userid': test_df['userid'],
        'feedid': test_df['feedid'],
        'read_comment_probability': predictions,
        'read_comment_label': test_df['read_comment']
    })
    result_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f"预测结果已保存至 {os.path.join(args.output_dir, 'predictions.csv')}")

if __name__ == "__main__":
    main()