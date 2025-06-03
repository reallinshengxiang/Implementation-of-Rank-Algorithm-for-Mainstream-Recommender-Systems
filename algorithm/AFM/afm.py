"""
    [1] Xiao, Jun, et al. "Attentional factorization machines: Learning the weight of feature interactions via attention networks."
    arXiv preprint arXiv:1708.04617 (2017).
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

class WechatDataset(Dataset):
    """微信视频号数据集"""
    def __init__(self, data_path, feature_columns, label_columns, vocabulary_dir):
        self.data = pd.read_parquet(data_path)
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.vocab_dir = vocabulary_dir
        # 加载词汇表
        self.vocabs = {}
        for col in self.feature_columns['category']:
            vocab_file = os.path.join(vocabulary_dir, f'{col}.txt')
            if os.path.exists(vocab_file):
                with open(vocab_file, 'r') as f:
                    vocab = [line.strip() for line in f if line.strip()]
                    self.vocabs[col] = {word: idx for idx, word in enumerate(vocab)}
        # 处理序列特征
        for seq_col in self.feature_columns['sequence']:
            self.data[seq_col] = self.data[seq_col].apply(
                lambda x: x.split(',') if isinstance(x, str) and x != '' else []
            )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 处理连续特征
        dense_features = torch.tensor(
            [row[col] for col in self.feature_columns['dense']], 
            dtype=torch.float32
        )
        # 处理类别特征
        category_features = {}
        for col in self.feature_columns['category']:
            if col in self.vocabs and row[col] in self.vocabs[col]:
                category_features[col] = torch.tensor(self.vocabs[col][row[col]], dtype=torch.long)
            else:
                category_features[col] = torch.tensor(0, dtype=torch.long)  # 未知类别
        # 处理标签
        label = torch.tensor(row[self.label_columns[0]], dtype=torch.float32)
        return dense_features, category_features, label

class AFM(nn.Module):
    """注意力因子分解机模型"""
    def __init__(self, feature_columns, embedding_dim, attention_factor):
        super(AFM, self).__init__()
        self.feature_columns = feature_columns
        self.embedding_dim = embedding_dim
        self.attention_factor = attention_factor
        # 连续特征处理
        self.dense_features = feature_columns['dense']
        self.num_dense = len(self.dense_features)
        self.dense_layer = nn.Linear(self.num_dense, 1)
        # 类别特征处理
        self.category_features = feature_columns['category']
        self.embeddings = nn.ModuleDict()
        for col in self.category_features:
            # 词汇表大小+1（用于未知类别）
            vocab_size = len(feature_columns['vocab'][col]) + 1
            self.embeddings[col] = nn.Embedding(vocab_size, embedding_dim)
        # 注意力网络
        self.num_fields = len(self.category_features)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_factor),
            nn.ReLU(),
            nn.Linear(attention_factor, 1)
        )
        # 预测层
        self.p = nn.Linear(embedding_dim, 1)
        
    def forward(self, dense_input, category_input):
        # 处理连续特征
        dense_logit = self.dense_layer(dense_input)  # [batch, 1]
        # 处理类别特征
        category_embeddings = []
        for col in self.category_features:
            embed = self.embeddings[col](category_input[col])  # [batch, embedding_dim]
            category_embeddings.append(embed)
        # 两两特征交互
        pair_interactions = []
        for i in range(self.num_fields):
            for j in range(i+1, self.num_fields):
                # 哈达玛积
                interaction = torch.mul(category_embeddings[i], category_embeddings[j])  # [batch, embedding_dim]
                pair_interactions.append(interaction) 
        # 堆叠所有交互结果
        pair_interactions = torch.stack(pair_interactions, dim=1)  # [batch, num_pairs, embedding_dim]
        # 注意力机制
        attention_scores = self.attention(pair_interactions)  # [batch, num_pairs, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, num_pairs, 1]
        # 加权求和
        weighted_sum = torch.sum(pair_interactions * attention_weights, dim=1)  # [batch, embedding_dim]
        # 预测分数
        afm_logit = self.p(weighted_sum)  # [batch, 1]       
        # 最终预测
        total_logit = dense_logit + afm_logit
        prediction = torch.sigmoid(total_logit)
        return prediction, total_logit

def create_feature_columns(vocabulary_dir):
    """创建特征列配置"""
    feature_columns = {
        'dense': [
            "videoplayseconds", "u_read_comment_7d_sum", "u_like_7d_sum", 
            "u_click_avatar_7d_sum", "u_forward_7d_sum", "u_comment_7d_sum", 
            "u_follow_7d_sum", "u_favorite_7d_sum", "i_read_comment_7d_sum", 
            "i_like_7d_sum", "i_click_avatar_7d_sum", "i_forward_7d_sum", 
            "i_comment_7d_sum", "i_follow_7d_sum", "i_favorite_7d_sum", 
            "c_user_author_read_comment_7d_sum"
        ],
        'category': [
            "userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id", "manual_tag_list"  # 保持与数据列名一致
        ],
        'sequence': [],
        'vocab': {}
    }
    label_columns = ["read_comment"]
    # 为manual_tag_list添加特殊映射
    column_to_vocab_mapping = {
        "manual_tag_list": "manual_tag_id",  # 指定列名与词汇表文件名的映射关系
    }
    # 加载词汇表大小
    for col in feature_columns['category']:
        # 优先使用特殊映射，否则使用列名
        vocab_name = column_to_vocab_mapping.get(col, col)
        vocab_file = os.path.join(vocabulary_dir, f'{vocab_name}.txt')   
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                vocab = [line.strip() for line in f if line.strip()]
            feature_columns['vocab'][col] = vocab
            print(f"Loaded vocabulary for {col} from {vocab_file}")
        else:
            print(f"Warning: Vocabulary file not found for {col} (searched {vocab_file})")
            feature_columns['vocab'][col] = []  # 使用空词汇表，后续处理
    return feature_columns, label_columns

def train(model, train_loader, criterion, optimizer, device, epoch):
    """训练模型一个epoch"""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (dense, category, labels) in progress_bar:
        # 移至设备
        dense = dense.to(device)
        category = {k: v.to(device) for k, v in category.items()}
        labels = labels.to(device)
        # 前向传播
        optimizer.zero_grad()
        predictions, _ = model(dense, category)
        loss = criterion(predictions.squeeze(), labels)
        # 反向传播
        loss.backward()
        optimizer.step()
        # 记录指标
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.squeeze().cpu().detach().numpy())
        # 更新进度条
        progress_bar.set_description(f'Epoch {epoch}, Loss: {total_loss/(batch_idx+1):.4f}')
    # 计算整体指标
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.4f}, Train AUC: {auc:.4f}, Train Accuracy: {accuracy:.4f}')
    return total_loss/len(train_loader), auc, accuracy

def evaluate(model, eval_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for dense, category, labels in eval_loader:
            # 移至设备
            dense = dense.to(device)
            category = {k: v.to(device) for k, v in category.items()}
            labels = labels.to(device)
            # 前向传播
            predictions, _ = model(dense, category)
            loss = criterion(predictions.squeeze(), labels)
            # 记录指标
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.squeeze().cpu().numpy())
    # 计算整体指标
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    print(f'Eval Loss: {total_loss/len(eval_loader):.4f}, Eval AUC: {auc:.4f}, Eval Accuracy: {accuracy:.4f}')
    return total_loss/len(eval_loader), auc, accuracy

def predict(model, test_loader, device):
    """模型预测"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for dense, category, _ in test_loader:
            # 移至设备
            dense = dense.to(device)
            category = {k: v.to(device) for k, v in category.items()}
            # 前向传播
            predictions, _ = model(dense, category)
            all_preds.extend(predictions.squeeze().cpu().numpy())
    return np.array(all_preds)

def main():
    """训练入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AFM Model for WeChat Video')
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where prediction results are saved")
    parser.add_argument("--train_data", type=str, default="../../dataset/wechat_algo_data1/dataframe/train.parquet", help="Path to the train data")
    parser.add_argument("--eval_data", type=str, default="../../dataset/wechat_algo_data1/dataframe/test.parquet", help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, default="../../dataset/wechat_algo_data1/vocabulary/", help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Epoch of training phase")
    parser.add_argument("--train_steps", type=int, default=10000, help="Number of training steps to perform")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--attention_factor", type=int, default=128, help="Hidden layer size of the attention network")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()
    print(args)
    # 创建特征列
    feature_columns, label_columns = create_feature_columns(args.vocabulary_dir)
    # 创建数据集和数据加载器
    train_dataset = WechatDataset(args.train_data, feature_columns, label_columns, args.vocabulary_dir)
    eval_dataset = WechatDataset(args.eval_data, feature_columns, label_columns, args.vocabulary_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # 创建模型
    device = torch.device(args.device)
    model = AFM(feature_columns, args.embedding_dim, args.attention_factor).to(device)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # 训练模型
    best_auc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_auc, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        eval_loss, eval_auc, eval_acc = evaluate(model, eval_loader, criterion, device)
        # 保存最佳模型
        if eval_auc > best_auc:
            best_auc = eval_auc
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print(f'Model saved at epoch {epoch} with AUC: {best_auc:.4f}')
    # 加载最佳模型进行预测
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pth')))
    predictions = predict(model, eval_loader, device)
    # 保存预测结果
    test_df = pd.read_parquet(args.eval_data)
    result_df = pd.DataFrame({
        'probabilities': predictions,
        'read_comment': test_df['read_comment'].values
    })
    result_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.csv')}")

if __name__ == "__main__":
    main()