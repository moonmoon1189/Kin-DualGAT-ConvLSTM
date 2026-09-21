import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class CBFQADataset(Dataset):
    def __init__(self, data_root, split="Train", max_len=300):
        self.data_root = data_root
        self.split = split
        self.max_len = max_len

        splits_path = os.path.join(data_root, "splits.json")
        with open(splits_path, 'r') as f:
            self.splits = json.load(f)

        self.sequence_ids = self.splits[split]

    def load_npy(self, path):
        return np.load(path).astype(np.float32)

    def load_score_json(self, seq_id):
        path = os.path.join(self.data_root, "Quality_Scores", f"{seq_id}_score.json")
        with open(path, 'r') as f:
            data = json.load(f)
        return data["mean_score"]

    def load_meta_json(self, seq_id):
        path = os.path.join(self.data_root, "Motion_Data", f"{seq_id}_meta.json")
        with open(path, 'r') as f:
            return json.load(f)

    def normalize_coordinates(self, coords):
        # Center coordinates relative to the first frame's spine base if needed
        return coords

    def pad_or_truncate(self, tensor):
        T, J, C = tensor.shape
        if T >= self.max_len:
            return tensor[:self.max_len], self.max_len
        pad_len = self.max_len - T
        padded = np.pad(tensor, ((0, pad_len), (0, 0), (0, 0)), mode='constant')
        return padded, T

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        meta = self.load_meta_json(seq_id)

        def_coords = self.load_npy(os.path.join(self.data_root, "Motion_Data", meta["files"]["defender"]))
        off_coords = self.load_npy(os.path.join(self.data_root, "Motion_Data", meta["files"]["attacker"]))

        def_coords = self.normalize_coordinates(def_coords)
        off_coords = self.normalize_coordinates(off_coords)

        score = self.load_score_json(seq_id) / 100.0  # Normalize score to [0,1]

        return {
            "defender": torch.tensor(def_coords),
            "attacker": torch.tensor(off_coords),
            "score": torch.tensor([score], dtype=torch.float32)
        }


def collate_fn(batch):
    lengths = [item["defender"].shape[0] for item in batch]
    max_batch_len = max(lengths)

    padded_def = []
    padded_off = []
    scores = []

    for item in batch:
        T, J, C = item["defender"].shape
        pad_len = max_batch_len - T

        p_def = torch.nn.functional.pad(item["defender"], (0, 0, 0, 0, 0, pad_len))
        p_off = torch.nn.functional.pad(item["attacker"], (0, 0, 0, 0, 0, pad_len))

        padded_def.append(p_def)
        padded_off.append(p_off)
        scores.append(item["score"])

    return {
        "defender": torch.stack(padded_def),
        "attacker": torch.stack(padded_off),
        "score": torch.stack(scores),
        "length": torch.tensor(lengths, dtype=torch.long)
    }