import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class EEGImageNetDatasetS2(Dataset):
    def __init__(self, dataset_dir, subject, granularity, stage, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        loaded = torch.load(os.path.join(dataset_dir, "EEG-ImageNet_stage2.pth"))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if subject >= 0:
            chosen_data = [loaded["dataset"][i] for i in range(len(loaded["dataset"])) if
                           loaded["dataset"][i]["subject"] == subject]
            if subject == 17:
                chosen_data = [i for i in chosen_data if self.labels.index(i["label"]) not in [51, 58]]
        else:
            chosen_data = loaded["dataset"]
        if granularity == "coarse":
            chosen_data = [i for i in chosen_data if i["granularity"] == "coarse"]
        elif granularity == "all":
            chosen_data = chosen_data
        else:
            fine_num = int(granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            chosen_data = [i for i in chosen_data if
                           i["granularity"] == "fine" and self.labels.index(i["label"]) in fine_category_range]
        if stage in [20, 30]:
            chosen_data = [i for i in chosen_data if i["stage"] == stage]
        else:
            dataset_30 = [i for i in chosen_data if i["stage"] == 30]
            dataset_20 = [i for i in chosen_data if i["stage"] == 20]
            chosen_data = dataset_30 + dataset_20
        self.data = chosen_data
        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = False

    def __getitem__(self, index):
        if self.use_image_label:
            path = self.data[index]["image"]
            label = Image.open(os.path.join(self.dataset_dir, "imageNet_images", path.split('_')[0], path))
            if label.mode == 'L':
                label = label.convert('RGB')
            if self.transform:
                label = self.transform(label)
            else:
                label = path
        else:
            label = self.labels.index(self.data[index]["label"])
        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]
        return feat, label

    def __len__(self):
        return len(self.data)

    def add_frequency_feat(self, feat):
        if len(feat) == len(self.data):
            self.frequency_feat = torch.from_numpy(feat).float()
        else:
            raise ValueError(f"Frequency features must have the same length. feat: {len(feat)}, data: {len(self.data)}")
