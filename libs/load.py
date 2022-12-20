import glob
import json
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from libs.preprocess import MonkeyPreprocess

class MonkeyDataset(Dataset):
    def __init__(self, data_path, preprocess, model_type, img_size, num_joints, sigma):
        super().__init__()
        json_file_path = glob.glob(os.path.join(data_path, "*.json"))[0]
        metadata = self.read_data(json_file_path)
        self.img_paths = metadata['img_paths']
        self.bboxes = metadata['bboxes']
        self.landmarks = metadata['landmarks']
        self.visibilities = metadata['visibilities']

        self.img_size = img_size
        self.map_size = self.img_size // 8 if model_type == "CPM" else self.img_size // 4
        self.sigma = sigma

        self.num_joints = num_joints
        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1.5, 1.2, 1.2, 1.5,
                1.2, 1.2, 1.5, 1.5, 1.2, 1.5, 1.2, 1.2
            ],
            dtype=np.float32
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.preprocess = MonkeyPreprocess(self.img_size, self.num_joints, preprocess)
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        bbox = np.array(self.bboxes[idx]).astype(np.float32)
        landmark = np.array(self.landmarks[idx]).astype(np.float32).reshape(-1, 2)
        visibility = np.array(self.visibilities[idx]).astype(np.float32)
        joints_weight = self.joints_weight.copy()

        # do data preprocessing-> crop objects based on bbox and do augmentations
        img, landmark, joints_weight, visibility = self.preprocess.apply(img, bbox, landmark, joints_weight, visibility)

        # generate groundtruth heatmap
        heatmaps = self.gen_heatmap(landmark)

        # generate weights for different joints while training
        joints_weight = self.gen_weights(visibility, joints_weight)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, heatmaps, joints_weight, landmark
            
    def __len__(self):
        return len(self.img_paths)

    def gen_heatmap(self, landmark):
        target = np.zeros((self.num_joints, self.map_size, self.map_size), dtype=np.float32)
        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            mu_x = int(landmark[joint_id][0] * self.map_size)
            mu_y = int(landmark[joint_id][1] * self.map_size)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= self.map_size or ul[1] >= self.map_size or br[0] < 0 or br[1] < 0:
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2

            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.map_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.map_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.map_size)
            img_y = max(0, ul[1]), min(br[1], self.map_size)

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target

    def gen_weights(self, visibility, joints_weight):
        joints_weight = np.multiply(visibility, joints_weight)
        return joints_weight

    def read_data(self, json_path):
        data_path = os.path.split(json_path)[0]

        f = open(json_path)
        data = json.load(f)

        img_paths, bboxes, landmarks, visibilities = [], [], [], []
        for item in data['data']:
            file = item['file']
            bbox = item['bbox']
            landmark = item['landmarks']
            visibility = item['visibility']

            img_file = os.path.join(data_path, file)

            img_paths.append(img_file)
            bboxes.append(bbox)
            landmarks.append(landmark)
            visibilities.append(visibility)

        metadata = {
            'img_paths' : img_paths,
            'bboxes' : bboxes,
            'landmarks' : landmarks,
            'visibilities' : visibilities
        }
        return metadata


def load_data(data_path, preprocess, model_type, batch_size, img_size, num_joints, sigma, action):
    if action == "train":
        assert type(data_path) == list, "Your data_path for training must be a list with length of 2. The list must contain image paths for training and validation"
        train_set = MonkeyDataset(data_path[0], preprocess['train'], model_type, img_size, num_joints, sigma)
        valid_set = MonkeyDataset(data_path[1], preprocess['valid'], model_type, img_size, num_joints, sigma)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_set, valid_set, train_dataloader, val_dataloader

    elif action == "test":
        test_set = MonkeyDataset(data_path, preprocess, model_type, img_size, num_joints, sigma)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_set, test_dataloader
