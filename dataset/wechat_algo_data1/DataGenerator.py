"""
    生成微信视频号比赛训练集/测试集的pytorch数据文件

    训练集: date_ = 8-13(生成特征需要开7天的窗口)
    测试集: date_ = 14

    特征：
        user侧：
            userid: 用户id
            u_read_comment_7d_sum: 近7天查看评论次数
            u_like_7d_sum: 近7天点赞次数
            u_click_avatar_7d_sum: 近7天点击头像次数
            u_favorite_7d_sum: 近7天收藏次数
            u_forward_7d_sum: 近7天转发次数
            u_comment_7d_sum: 近7天评论次数
            u_follow_7d_sum: 近7天关注次数
            his_read_comment_7d_seq: 近7天查看评论序列, 最长50个
            device: 设备类型

        item侧:
            feedid: feedid
            i_read_comment_7d_sum: 近7天被查看评论次数
            i_like_7d_sum: 近7天被点赞次数
            i_click_avatar_7d_sum: 近7天被点击头像次数
            i_favorite_7d_sum: 近7天被收藏次数
            i_forward_7d_sum: 近7天被转发次数
            i_comment_7d_sum: 近7天被评论次数
            i_follow_7d_sum: 近7天经由此feedid, 作者被关注次数
            videoplayseconds: feed时长
            authorid: 作者id
            bgm_song_id: 背景音乐id
            bgm_singer_id: 背景音乐歌手id
            manual_tag_list: 人工标注的分类标签

        交叉侧:(过于稀疏且耗费资源, 暂时只考虑第一个)
            c_user_author_read_comment_7d_sum:  user对当前item作者的查看评论次数
            c_user_author_like_7d_sum:  user对当前item作者的点赞次数
            c_user_author_click_avatar_7d_sum:  user对当前item作者的点击头像次数
            c_user_author_favorite_7d_sum:  user对当前item作者的收藏次数
            c_user_author_forward_7d_sum:  user对当前item作者的转发次数
            c_user_author_comment_7d_sum:  user对当前item作者的评论次数
            c_user_author_follow_7d_sum:  user对当前item作者的关注次数
"""

import os
import warnings
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from collections import Counter

warnings.filterwarnings('ignore')
tqdm.pandas(desc='pandas bar')

ACTION_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
END_DAY = 14

class DataGenerator:
    """生成微信视频号训练集/测试集的PyTorch格式数据"""

    def __init__(self, dataset_dir: str = './', out_path: str = './'):
        """
        Args:
            dataset_dir: 数据文件所在文件夹路径
            out_path: 输出文件路径
        """

        self.dataset_dir = dataset_dir
        self.out_path = out_path       
        self.dense_features = [
            "videoplayseconds",
            "u_read_comment_7d_sum",
            "u_like_7d_sum",
            "u_click_avatar_7d_sum",
            "u_forward_7d_sum",
            "u_comment_7d_sum",
            "u_follow_7d_sum",
            "u_favorite_7d_sum",
            "i_read_comment_7d_sum",
            "i_like_7d_sum",
            "i_click_avatar_7d_sum",
            "i_forward_7d_sum",
            "i_comment_7d_sum",
            "i_follow_7d_sum",
            "i_favorite_7d_sum",
            "c_user_author_read_comment_7d_sum",
        ]
        self.category_features = [
            "userid",
            "feedid",
            "device",
            "authorid",
            "bgm_song_id",
            "bgm_singer_id",
        ]
        self.seq_features = ["his_read_comment_7d_seq", "manual_tag_list"]
        self.labels = [
            "read_comment",
            "comment",
            "like",
            "click_avatar",
            "forward",
            "follow",
            "favorite",
        ]
        # 创建存放vocabulary文件的文件夹
        self.vocab_dir = os.path.join(self.out_path, 'vocabulary')
        # 创建存放features分片的文件夹
        self.features_dir = os.path.join(self.out_path, 'features')
        # 创建存放dataframe的文件夹
        self.dataframe_dir = os.path.join(self.out_path, 'dataframe')
        # 创建存放PyTorch数据的文件夹
        self.pytorch_dir = os.path.join(self.out_path, 'pytorch_data')
        # 确保所有目录存在
        for path in [self.vocab_dir, self.features_dir, self.dataframe_dir, self.pytorch_dir]:
            os.makedirs(path, exist_ok=True)
        print("开始数据处理流程...")     
        if not os.path.exists(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY')):
            print("步骤1: 加载数据...")
            self._load_data()
            print("步骤2: 数据预处理...")
            self._preprocess()
        print("步骤3: 生成词汇表...")
        self._generate_vocabulary_file()
        print("步骤4: 生成特征...")
        self._generate_features()
        print("步骤5: 生成数据框...")
        self._generate_dataframe()
        print("步骤6: 生成PyTorch数据...")
        self._generate_pytorch_data()
        print("数据处理完成!")

    def _load_data(self):
        """读入数据"""
        print("正在加载user_action.csv...")
        self.user_action = pd.read_csv(os.path.join(self.dataset_dir, 'user_action.csv'))
        print(f"user_action数据形状: {self.user_action.shape}")
        
        print("正在加载feed_info.csv...")
        self.feed_info = pd.read_csv(os.path.join(self.dataset_dir, 'feed_info.csv'),
                                     usecols=["feedid", "authorid", "videoplayseconds", "bgm_song_id", "bgm_singer_id",
                                              "manual_tag_list"])
        print(f"feed_info数据形状: {self.feed_info.shape}")

    def _preprocess(self):
        """数据预处理，把所有类别变量取值前面都加上前缀"""
        print("开始数据预处理...")
        self.feed_info['feedid'] = self.feed_info['feedid'].astype(str)
        self.feed_info['authorid'] = self.feed_info['authorid'].astype(str)
        # int型column中有空值存在的情况下, pd.read_csv后会被cast成float, 需要用扩展类型代替int
        self.feed_info['bgm_song_id'] = self.feed_info['bgm_song_id'].astype(pd.Int16Dtype()).astype(str)
        self.feed_info['bgm_singer_id'] = self.feed_info['bgm_singer_id'].astype(pd.Int16Dtype()).astype(str)
        print("处理feed_info前缀...")
        for index, row in tqdm(self.feed_info.iterrows(), total=self.feed_info.shape[0], desc="添加前缀"):
            self.feed_info.at[index, 'feedid'] = 'feedid_' + row['feedid']
            self.feed_info.at[index, 'authorid'] = 'authorid_' + row['authorid']
            self.feed_info.at[index, 'bgm_song_id'] = 'bgm_song_id_' + row['bgm_song_id'] if row['bgm_song_id'] != '<NA>' else np.nan
            self.feed_info.at[index, 'bgm_singer_id'] = 'bgm_singer_id_' + row['bgm_singer_id'] if row['bgm_singer_id'] != '<NA>' else np.nan
            self.feed_info.at[index, 'manual_tag_list'] = ['manual_tag_id_' + tag for tag in
                                                           row['manual_tag_list'].split(';')] if pd.notna(row['manual_tag_list']) else np.nan
        print("处理user_action前缀...")
        self.user_action['userid'] = 'userid_' + self.user_action['userid'].astype(str)
        self.user_action['feedid'] = 'feedid_' + self.user_action['feedid'].astype(str)
        self.user_action['device'] = 'device_' + self.user_action['device'].astype(str)

    def _generate_vocabulary_file(self):
        """
            生成所有类别特征的vocabulary文件(txt格式)
            userid, feedid, device, authorid, bgm_song_id, bgm_singer_id, manual_tag_id
        """
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.vocab_dir, 'VOCAB_FILE_ALREADY')):
            print("Vocabulary files ready!")
            return
        # 每个分类变量对应一个Counter对象
        vocabulary_dict = {}
        # user_id, device
        action_scope = self.user_action[self.user_action['date_'].between(8, 14)]
        user_vocab = Counter(action_scope['userid'])
        device_vocab = Counter(action_scope['device'])
        vocabulary_dict['userid'] = user_vocab
        vocabulary_dict['device'] = device_vocab

        # feedid, authorid, bgm_song_id, bgm_singer_id
        feedid_vocab = Counter(self.feed_info['feedid'])
        authorid_vocab = Counter(self.feed_info['authorid'])
        # bgm_song_id_vocab和bgm_singer_id_vocab有空值, 需要处理Counter
        bgm_song_id_vocab = Counter(self.feed_info['bgm_song_id'])
        if np.nan in bgm_song_id_vocab:
            bgm_song_id_vocab.pop(np.nan)
        bgm_singer_id_vocab = Counter(self.feed_info['bgm_singer_id'])
        if np.nan in bgm_singer_id_vocab:
            bgm_singer_id_vocab.pop(np.nan)
        vocabulary_dict['feedid'] = feedid_vocab
        vocabulary_dict['authorid'] = authorid_vocab
        vocabulary_dict['bgm_song_id'] = bgm_song_id_vocab
        vocabulary_dict['bgm_singer_id'] = bgm_singer_id_vocab

        # manual_tag_list
        manual_tag_id_vocab = Counter()
        for index, row in tqdm(self.feed_info.iterrows(), total=self.feed_info.shape[0], desc="构建标签词汇表"):
            # 修复逻辑：正确检查manual_tag_list是否为有效列表
            tag_list = row['manual_tag_list']
            if isinstance(tag_list, list) and len(tag_list) > 0:
                manual_tag_id_vocab.update(tag_list)
            elif isinstance(tag_list, np.ndarray) and tag_list.size > 0:
                # 处理numpy数组的情况
                manual_tag_id_vocab.update(tag_list.tolist())
                
        vocabulary_dict['manual_tag_id'] = manual_tag_id_vocab
        print("保存词汇表文件...")
        for variable_name, vocab in vocabulary_dict.items():
            vocabulary_file_path = os.path.join(self.vocab_dir, variable_name + '.txt')
            with open(vocabulary_file_path, 'w') as f:
                for key, _ in vocab.items():
                    f.write(str(key) + '\n')
            print(f"  {variable_name}: {len(vocab)} 个词汇")
        # 生成完成标识
        with open(os.path.join(self.vocab_dir, 'VOCAB_FILE_ALREADY'), 'w'):
            pass


    def _generate_features(self, start_day: int = 1, window_size: int = 7):
        """
        生成user侧, item侧, 交叉侧特征
        Args:
            start_day:  从第几天开始构建特征, 默认从第1天开始
            window_size:  特征窗口大小, 默认为7
        """
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.features_dir, 'FEATURES_PKL_ALREADY')):
            print("Features pkl ready!")
            return
        print("生成用户侧特征...")
        # user侧
        user_features_data = self.user_action[["userid", "date_"] + ACTION_COLUMN_LIST]
        # 1. 统计特征
        user_arr = []
        for start in range(start_day, END_DAY - window_size + 1):
            # 需要聚合的数据范围
            temp = user_features_data[
                (user_features_data['date_'] >= start) & (user_features_data['date_'] < (start + window_size))]
            # date_列要重新生成
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(['userid']).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'userid':
                    new_column_names.append('userid')
                else:
                    new_column_names.append('u_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 重新生成date_列
            temp['date_'] = start + window_size
            user_arr.append(temp)
        user_agg_features = pd.concat(user_arr, ignore_index=True)
        user_agg_features.to_pickle(os.path.join(self.features_dir, 'user_agg_features.pkl'))
        # 2. 历史序列特征
        user_arr = []
        user_features_data = self.user_action[["userid", "feedid", "date_"] + ACTION_COLUMN_LIST]
        # 基本逻辑同统计特征, 独立拆分出来是为了逻辑清晰
        for start in range(start_day, END_DAY - window_size + 1):
            temp = user_features_data[(user_features_data['date_'] >= start) &
                                      (user_features_data['date_'] < (start + window_size)) &
                                      (user_features_data['read_comment'] == 1)]
            temp.drop(columns=['date_'], inplace=True)
            temp = temp.groupby(['userid']).agg(
                his_read_comment_7d_seq=pd.NamedAgg(column="feedid", aggfunc=list)).reset_index()
            # 只取后50个元素
            temp['his_read_comment_7d_seq'] = temp.apply(
                lambda row: row.his_read_comment_7d_seq[-50:] if isinstance(row.his_read_comment_7d_seq, list) and len(row.his_read_comment_7d_seq) > 50 else row.his_read_comment_7d_seq, axis=1)
            temp['date_'] = start + window_size
            user_arr.append(temp)
        user_seq_features = pd.concat(user_arr, ignore_index=True)
        user_seq_features.to_pickle(os.path.join(self.features_dir, 'user_seq_features.pkl'))
        print("生成内容侧特征...")
        # item侧
        feed_features_data = self.user_action[["feedid", "date_"] + ACTION_COLUMN_LIST]
        # 1. 统计特征
        feed_arr = []
        for start in range(start_day, END_DAY - window_size + 1):
            # 需要聚合的数据范围
            temp = feed_features_data[
                (feed_features_data['date_'] >= start) & (feed_features_data['date_'] < (start + window_size))]
            # date_列要重新生成
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(['feedid']).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'feedid':
                    new_column_names.append('feedid')
                else:
                    new_column_names.append('i_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 重新生成date_列
            temp['date_'] = start + window_size
            feed_arr.append(temp)
        feed_agg_features = pd.concat(feed_arr, ignore_index=True)
        feed_agg_features.to_pickle(os.path.join(self.features_dir, 'feed_agg_features.pkl'))
        print("生成交叉特征...")
        # 交叉侧
        cross_features_data = pd.merge(
            self.user_action[["userid", "feedid", "date_"] + ['read_comment']], 
            self.feed_info, 
            on="feedid",
            how="left"
        )[["userid", "authorid", "date_"] + ['read_comment']]
        # 1. 统计特征
        cross_arr = []
        for start in range(start_day, END_DAY - window_size + 1):
            temp = cross_features_data[
                (cross_features_data['date_'] >= start) & (cross_features_data['date_'] < (start + window_size))]
            temp.drop(columns=['date_'], inplace=True)
            # 聚合
            temp = temp.groupby(["userid", "authorid"]).agg(['sum']).reset_index()
            # 结果数据的列名
            new_column_names = []
            for col, agg_name in temp.columns.values:
                if col == 'userid' or col == "authorid":
                    new_column_names.append(col)
                else:
                    new_column_names.append('c_user_author_' + col + '_7d_' + agg_name)
            temp.columns = new_column_names
            # 只保留大于0的行, 节省空间资源
            temp = temp[temp['c_user_author_read_comment_7d_sum'] > 0]
            # 重新生成date_列
            temp['date_'] = start + window_size
            cross_arr.append(temp)

        cross_agg_features = pd.concat(cross_arr, ignore_index=True)
        cross_agg_features.to_pickle(os.path.join(self.features_dir, 'cross_agg_features.pkl'))
        # 生成完成标识
        with open(os.path.join(self.features_dir, 'FEATURES_PKL_ALREADY'), 'w'):
            pass

    def _generate_dataframe(self):
        """生成样本表"""
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY')):
            print("DataFrame ready!")
            return
        print("合并所有特征...")
        self.user_action = self.user_action[self.user_action['date_'].between(8, 14)]
        user_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'user_agg_features.pkl'))
        user_seq_features = pd.read_pickle(os.path.join(self.features_dir, 'user_seq_features.pkl'))
        feed_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'feed_agg_features.pkl'))
        cross_agg_features = pd.read_pickle(os.path.join(self.features_dir, 'cross_agg_features.pkl'))
        self.user_action = pd.merge(self.user_action, self.feed_info, on=['feedid'], how='left')
        self.user_action = pd.merge(self.user_action, user_agg_features, on=['userid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, user_seq_features, on=['userid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, feed_agg_features, on=['feedid', 'date_'], how='left')
        self.user_action = pd.merge(self.user_action, cross_agg_features, on=['userid', 'authorid', 'date_'], how='left')
        print("处理缺失值和特征变换...")
        # 填补空值
        for col in self.dense_features:
            # 使用向量化操作代替循环，大幅提高效率
            self.user_action[col] = np.log1p(self.user_action[col].fillna(0))
        # 处理序列特征
        for col in self.seq_features:
            self.user_action[col] = self.user_action[col].apply(
                lambda x: ','.join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ''
            )
        print("保存训练集和测试集...")
        # 保存训练集和测试集
        train_data = self.user_action[self.user_action['date_'].between(8, 13)]
        test_data = self.user_action[self.user_action['date_'] == 14]
        train_data.to_parquet(os.path.join(self.dataframe_dir, 'train.parquet'))
        test_data.to_parquet(os.path.join(self.dataframe_dir, 'test.parquet'))
        print(f"训练集样本数: {len(train_data)}")
        print(f"测试集样本数: {len(test_data)}")
        # 生成完成标识
        with open(os.path.join(self.dataframe_dir, 'DATAFRAME_ALREADY'), 'w'):
            pass

    def _generate_pytorch_data(self):
        """生成PyTorch格式的数据集"""
        # 如果任务已经完成, 退出
        if os.path.exists(os.path.join(self.pytorch_dir, 'PYTORCH_DATA_ALREADY')):
            print("PyTorch data ready!")
            return
        print("加载处理后的数据...")
        # 加载训练集和测试集
        train_df = pd.read_parquet(os.path.join(self.dataframe_dir, 'train.parquet'))
        test_df = pd.read_parquet(os.path.join(self.dataframe_dir, 'test.parquet'))
        # 准备数据字典
        def prepare_data(df):
            data = {
                'dense_features': df[self.dense_features].values.astype(np.float32),
                'category_features': df[self.category_features].values.astype(str),
                'seq_features': {
                    'his_read_comment_7d_seq': df['his_read_comment_7d_seq'].apply(
                        lambda x: x.split(',') if pd.notna(x) and x != '' else []
                    ).tolist(),
                    'manual_tag_list': df['manual_tag_list'].apply(
                        lambda x: x.split(',') if pd.notna(x) and x != '' else []
                    ).tolist(),
                },
                'labels': df[self.labels].values.astype(np.float32)
            }
            return data
        print("转换训练数据...")
        # 处理训练数据
        train_data = prepare_data(train_df)
        print("转换测试数据...")
        # 处理测试数据
        test_data = prepare_data(test_df)
        print("保存PyTorch张量...")
        # 保存为PyTorch张量格式
        torch.save({
            'train': {
                'dense_features': torch.tensor(train_data['dense_features'], dtype=torch.float32),
                'labels': torch.tensor(train_data['labels'], dtype=torch.float32)
            },
            'test': {
                'dense_features': torch.tensor(test_data['dense_features'], dtype=torch.float32),
                'labels': torch.tensor(test_data['labels'], dtype=torch.float32)
            },
            'category_features': {
                'train': train_data['category_features'],
                'test': test_data['category_features']
            },
            'seq_features': {
                'train': train_data['seq_features'],
                'test': test_data['seq_features']
            },
            'metadata': {
                'dense_features': self.dense_features,
                'category_features': self.category_features,
                'seq_features': self.seq_features,
                'labels': self.labels
            }
        }, os.path.join(self.pytorch_dir, 'dataset.pt'))
        print("构建词汇表映射...")
        # 保存分类特征的词汇表
        vocab_dict = {}
        for feature in self.category_features:
            vocab_file = os.path.join(self.vocab_dir, f'{feature}.txt')
            if os.path.exists(vocab_file):
                with open(vocab_file, 'r') as f:
                    vocab = [line.strip() for line in f if line.strip()]
                    vocab_dict[feature] = {word: idx for idx, word in enumerate(vocab)}
        # 保存序列特征的词汇表
        manual_tag_file = os.path.join(self.vocab_dir, 'manual_tag_id.txt')
        if os.path.exists(manual_tag_file):
            with open(manual_tag_file, 'r') as f:
                manual_tag_vocab = [line.strip() for line in f if line.strip()]
                vocab_dict['manual_tag_id'] = {tag: idx for idx, tag in enumerate(manual_tag_vocab)}
        # 保存feedid的词汇表用于序列特征
        feedid_file = os.path.join(self.vocab_dir, 'feedid.txt')
        if os.path.exists(feedid_file):
            with open(feedid_file, 'r') as f:
                feedid_vocab = [line.strip() for line in f if line.strip()]
                vocab_dict['feedid_seq'] = {feed: idx for idx, feed in enumerate(feedid_vocab)}
        with open(os.path.join(self.pytorch_dir, 'vocab_dict.pkl'), 'wb') as f:
            pickle.dump(vocab_dict, f)
        print("词汇表统计:")
        for key, vocab in vocab_dict.items():
            print(f"  {key}: {len(vocab)} 个词汇")
        # 生成完成标识
        with open(os.path.join(self.pytorch_dir, 'PYTORCH_DATA_ALREADY'), 'w'):
            pass


if __name__ == '__main__':
    # 使用示例
    data_generator = DataGenerator(
        dataset_dir='./',  # 输入数据目录
        out_path='./'      # 输出数据目录
    )