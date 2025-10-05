import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class EEGImageNetDatasetS1(Dataset):
    def __init__(self, dataset_dir, subject, granularity, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        loaded = torch.load(os.path.join(dataset_dir, "EEG-ImageNet_stage2.pth"))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if subject >= 0:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == subject]
        else:
            chosen_data = loaded['dataset']
            if subject == -2:
                exclude_sub = [0, 4, 6, 12, 13]
                chosen_data = [i for i in chosen_data if i['subject'] not in exclude_sub]
        if granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]
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
            # print(f"{index} {path} {label.size()}")
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
            raise ValueError("Frequency features must have same length")
