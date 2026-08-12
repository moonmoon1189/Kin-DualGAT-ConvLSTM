# split_data_by_subject.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

INPUT_EXCEL = "data/synthetic_basketball_data.xlsx"
OUTPUT_DIR = "data"
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_by_subject():
    df = pd.read_excel(INPUT_EXCEL)

    # 确保有 subject_id 列（如果没有，则按序列分配虚拟ID，但此处已有）
    if 'subject_id' not in df.columns:
        raise ValueError("Excel 文件中没有 'subject_id' 列，请先生成带 subject_id 的数据！")

    subjects = df['subject_id'].unique()
    print(f"共有 {len(subjects)} 名运动员")

    # 按 subject 划分
    train_subjs, temp_subjs = train_test_split(subjects, test_size=(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)
    val_subjs, test_subjs = train_test_split(temp_subjs, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)

    train_df = df[df['subject_id'].isin(train_subjs)]
    val_df = df[df['subject_id'].isin(val_subjs)]
    test_df = df[df['subject_id'].isin(test_subjs)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_excel(os.path.join(OUTPUT_DIR, "train.xlsx"), index=False)
    val_df.to_excel(os.path.join(OUTPUT_DIR, "val.xlsx"), index=False)
    test_df.to_excel(os.path.join(OUTPUT_DIR, "test.xlsx"), index=False)

    print(f"训练集: {len(train_df)} 行 (运动员: {sorted(train_subjs)})")
    print(f"验证集: {len(val_df)} 行 (运动员: {sorted(val_subjs)})")
    print(f"测试集: {len(test_df)} 行 (运动员: {sorted(test_subjs)})")
    print("✅ 拆分完成！")

if __name__ == "__main__":
    split_by_subject()