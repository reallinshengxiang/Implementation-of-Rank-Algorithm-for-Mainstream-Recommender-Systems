"""
    [1] Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017).

    [2] Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE
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
        }
        # 创建词汇表索引映射
        self.vocab_indices = {k: {v: i for i, v in enumerate(vocab)} for k, vocab in self.vocabs.items()}
        # 类别特征列
        self.category_features = [
            "userid", "feedid", "device", "authorid", "bgm_song_id", "bgm_singer_id"
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
            'category': category,
            'label': label
        }

# DeepFM模型
class DeepFM(nn.Module):
    def __init__(self, vocab_dir, embedding_dim=8, hidden_units=None, dropout_rate=0.1, batch_norm=True):
        super(DeepFM, self).__init__()
        if hidden_units is None:
            hidden_units = [512, 256, 128] 
        # 加载词汇表大小
        self.vocab_sizes = {
            'userid': len(self._load_vocabulary(vocab_dir, 'userid.txt')) + 1,
            'feedid': len(self._load_vocabulary(vocab_dir, 'feedid.txt')) + 1,
            'device': len(self._load_vocabulary(vocab_dir, 'device.txt')) + 1,
            'authorid': len(self._load_vocabulary(vocab_dir, 'authorid.txt')) + 1,
            'bgm_song_id': len(self._load_vocabulary(vocab_dir, 'bgm_song_id.txt')) + 1,
            'bgm_singer_id': len(self._load_vocabulary(vocab_dir, 'bgm_singer_id.txt')) + 1,
        }
        # 类别特征数量
        self.num_categories = len(self.vocab_sizes)
        # FM一阶部分
        self.first_order_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, 1)
            for col, vocab_size in self.vocab_sizes.items()
        })
        # FM二阶部分
        self.second_order_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embedding_dim)
            for col, vocab_size in self.vocab_sizes.items()
        })
        # Deep部分
        self.deep_layers = nn.ModuleList()
        input_dim = self.num_categories * embedding_dim
        for unit in hidden_units:
            self.deep_layers.append(nn.Linear(input_dim, unit))
            if batch_norm:
                self.deep_layers.append(nn.BatchNorm1d(unit))
            self.deep_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.deep_layers.append(nn.Dropout(dropout_rate))
            input_dim = unit
        self.deep_output_layer = nn.Linear(input_dim, 1)
        # 输出层
        self.final_layer = nn.Linear(3, 1)  # 融合一阶、二阶和深度部分的输出
        
    def _load_vocabulary(self, vocab_dir, filename):
        vocab_path = os.path.join(vocab_dir, filename)
        if not os.path.exists(vocab_path):
            return []
        with open(vocab_path, 'r') as f:
            return [line.strip() for line in f]
    
    def forward(self, category):
        # FM一阶部分
        first_order_outputs = []
        for col, embedding in self.first_order_embeddings.items():
            if col in category:
                first_order_outputs.append(embedding(category[col]))
        fm_first_order_logit = torch.sum(torch.cat(first_order_outputs, dim=1), dim=1, keepdim=True)
        # FM二阶部分
        second_order_embeddings = []
        for col, embedding in self.second_order_embeddings.items():
            if col in category:
                second_order_embeddings.append(embedding(category[col]))
        # 先加再平方
        sum_embedding = torch.sum(torch.stack(second_order_embeddings, dim=1), dim=1)
        sum_embedding_square = torch.square(sum_embedding)
        # 先平方再加
        square_embedding = [torch.square(emb) for emb in second_order_embeddings]
        square_sum_embedding = torch.sum(torch.stack(square_embedding, dim=1), dim=1)
        # FM二阶交叉项
        fm_second_order_logit = 0.5 * torch.sum(sum_embedding_square - square_sum_embedding, dim=1, keepdim=True)
        # Deep部分
        deep_input = torch.cat(second_order_embeddings, dim=1)
        deep_output = deep_input
        for layer in self.deep_layers:
            deep_output = layer(deep_output)    
        deep_logit = self.deep_output_layer(deep_output)
        # 融合三个部分的输出
        total_logit = torch.cat([fm_first_order_logit, fm_second_order_logit, deep_logit], dim=1)
        total_logit = self.final_layer(total_logit)
        probability = torch.sigmoid(total_logit)
        return probability, total_logit, fm_first_order_logit, fm_second_order_logit, deep_logit

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        # 移至设备
        label = batch['label'].to(device)
        # 处理类别特征
        category = {k: v.to(device) for k, v in batch['category'].items()}
        # 前向传播
        optimizer.zero_grad()
        probabilities, _, _, _, _ = model(category)
        # 计算损失
        loss = criterion(probabilities.squeeze(), label)
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
            label = batch['label'].to(device)
            # 处理类别特征
            category = {k: v.to(device) for k, v in batch['category'].items()}
            # 前向传播
            probabilities, _, _, _, _ = model(category)
            # 计算损失
            loss = criterion(probabilities.squeeze(), label)
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
    hidden_units = [int(unit) for unit in args.hidden_units.split(',')]
    model = DeepFM(
        vocab_dir=args.vocabulary_dir,
        embedding_dim=args.embedding_dim,
        hidden_units=hidden_units,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm
    ).to(device)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
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
            category = {k: v.to(device) for k, v in batch['category'].items()} 
            # 前向传播
            probabilities, _, _, _, _ = model(category)
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
    parser = argparse.ArgumentParser(description='DeepFM Model Training')
    # 训练参数
    parser.add_argument("--model_dir", type=str, default="./model_dir", help="Directory where model parameters are saved")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Directory where output files are saved")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--vocabulary_dir", type=str, required=True, help="Folder where the vocabulary file is stored")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--hidden_units", type=str, default="512,256,128", help="Comma-separated list of number of units in each hidden layer of the deep part")
    parser.add_argument("--batch_norm", type=bool, default=True, help="Perform batch normalization (True or False)")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000, help="Save checkpoints every this many steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()
    main(args)