# generate_synthetic_data.py
import numpy as np
import pandas as pd
import os

# ================================
# 参数配置
# ================================
SEQ_LEN = 156                # 每个序列帧数（与数据集定义一致）
NUM_SAMPLES = 500            # 生成多少个独立序列
NUM_JOINTS = 25              # 关节数量
NUM_PERSONS = 2              # 人数（防守+进攻）
NUM_SUBJECTS = 30            # 模拟30名运动员（论文中运动员总数）

# 输出路径（确保 data 文件夹存在）
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_basketball_data.xlsx")

# ================================
# 辅助函数：三维随机游走
# ================================
def random_walk_3d(length, step=0.1):
    pos = np.zeros((length, 3))
    for t in range(1, length):
        delta = np.random.normal(0, step, 3)
        pos[t] = pos[t-1] + delta
    return pos

# ================================
# 生成单个序列的骨架和评分（增加 subject_id 参数）
# ================================
def generate_one_sequence(seq_len=SEQ_LEN, joints=NUM_JOINTS, persons=NUM_PERSONS, subject_id=None):
    """
    生成一个动作片段（seq_len 帧）的双人骨架坐标（x, y, z）及对应的防守评分。
    新增参数 subject_id：该序列所属的运动员编号（1~30）
    """
    # 防守者质心轨迹
    def_pos = random_walk_3d(seq_len, step=0.12)
    # 进攻者质心轨迹
    off_pos = random_walk_3d(seq_len, step=0.15)
    off_pos += np.array([2.5, 0.5, 0.0])

    # 随机骨架形状
    base_joint_offsets = np.random.randn(joints, 3) * 0.3
    base_joint_offsets = base_joint_offsets / np.linalg.norm(base_joint_offsets, axis=1, keepdims=True) * 0.2

    skeletons = np.zeros((seq_len, persons, joints, 3))
    for t in range(seq_len):
        def_joints = def_pos[t] + base_joint_offsets + np.random.normal(0, 0.02, (joints, 3))
        off_joints = off_pos[t] + base_joint_offsets * 1.1 + np.random.normal(0, 0.02, (joints, 3))
        skeletons[t, 0] = def_joints
        skeletons[t, 1] = off_joints

    # 计算评分
    def_center = skeletons[:, 0, 0, :]
    off_center = skeletons[:, 1, 0, :]
    distances = np.linalg.norm(def_center - off_center, axis=1)
    avg_dist = np.mean(distances)
    def_vel = np.diff(def_pos, axis=0)
    vel_std = np.std(np.linalg.norm(def_vel, axis=1))

    score = 80 - 15 * (avg_dist / 3.0) - 5 * (vel_std / 0.3) + np.random.normal(0, 3)
    score = np.clip(score, 0, 100)

    # 展平骨架
    flat_seq = skeletons.reshape(seq_len, -1)  # (seq_len, persons*joints*3)

    return flat_seq, score, subject_id

# ================================
# 生成所有序列并写入 Excel（包含 subject_id 列）
# ================================
def generate_dataset():
    all_data = []

    # 为每个序列循环分配 subject_id（1~30）
    subject_ids = [i % NUM_SUBJECTS + 1 for i in range(NUM_SAMPLES)]

    for i in range(NUM_SAMPLES):
        subject_id = subject_ids[i]
        flat_seq, score, _ = generate_one_sequence(subject_id=subject_id)
        for t in range(SEQ_LEN):
            # 前150列坐标 + score + subject_id
            row = list(flat_seq[t]) + [score, subject_id]
            all_data.append(row)

    # 构建列名
    cols = []
    for p in range(NUM_PERSONS):
        for j in range(NUM_JOINTS):
            for c in ['x', 'y', 'z']:
                cols.append(f'P{p+1}_J{j+1}_{c}')
    cols.append('score')
    cols.append('subject_id')   # 新增列

    df = pd.DataFrame(all_data, columns=cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ 合成数据已生成并保存至：{OUTPUT_FILE}")
    print(f"总共生成 {NUM_SAMPLES} 个动作序列，合计 {len(df)} 帧数据。")
    print(f"新增 'subject_id' 列，范围 1~{NUM_SUBJECTS}，用于跨受试者划分。")

if __name__ == "__main__":
    generate_dataset()