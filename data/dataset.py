import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RealSkeletonDataset(Dataset):
    """读取合成（或清洗后）连续动作数据的 PyTorch 数据集加载器"""

    def __init__(self, excel_path, seq_len=156):
        # 论文配置表中指定，平均序列长度设定为 156 帧[cite: 2]
        print(f"正在加载数据集: {excel_path} ...")
        self.df = pd.read_excel(excel_path)
        self.seq_len = seq_len
        self.samples = self._parse_data()
        print(f"数据集加载完毕！共提取出 {len(self.samples)} 个有效动作序列。")

    def _parse_data(self):
        """
        数据解析核心逻辑：将二维表格转换为高维张量
        """
        samples = []
        total_frames = len(self.df)

        # 按 seq_len 长度滑动切分连续的数据帧
        for i in range(0, total_frames - self.seq_len + 1, self.seq_len):
            chunk = self.df.iloc[i: i + self.seq_len]

            # 前 150 列为双人三维骨架坐标 (2人 * 25关节 * 3坐标 = 150)
            # 网络输入规范要求将其重塑为 (T_seq, Person=2, Joints=25, Coords=3) 的维度[cite: 2]
            features = chunk.iloc[:, :150].values
            features = features.reshape(self.seq_len, 2, 25, 3)

            # 提取 'score' 列作为回归标签，并归一化至 [0, 1] 区间以便网络寻优
            score = chunk['score'].iloc[0] / 100.0

            samples.append({
                "skeleton": torch.tensor(features, dtype=torch.float32),
                "score": torch.tensor([score], dtype=torch.float32)
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["skeleton"], item["score"]