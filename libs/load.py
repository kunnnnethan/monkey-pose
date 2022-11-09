import glob
import json
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


class MonkeyDataset(Dataset):
    def __init__(self, data_path, img_size, stride, sigma):
        super().__init__()
        json_file_path = glob.glob(os.path.join(data_path, "*.json"))[0]
        img_paths, bboxes, landmarks, visibilities = self.read_data(json_file_path)
        self.img_paths = img_paths
        self.bboxes = bboxes
        self.landmarks = landmarks
        self.visibilities = visibilities

        self.img_size = img_size
        self.stride = stride
        self.map_size = self.img_size // self.stride
        self.sigma = sigma
        
    def __getitem__(self, idx):
        # load img from PIL
        img = cv2.imread(self.img_paths[idx])
        landmark = np.array(self.landmarks[idx]).astype(np.float32).reshape(-1, 2)
        bbox = np.array(self.bboxes[idx]).astype(np.float32)

        img, bbox, landmark = self.crop_image(img, bbox, landmark)
        height, width, _ = img.shape

        # normalize landmarks with respect to image size

        landmark[:, 0] = landmark[:, 0] / width
        landmark[:, 1] = landmark[:, 1] / height
        heatmaps = self.gen_heatmap(landmark)

        bbox[0] = bbox[0] / width
        bbox[1] = bbox[1] / height
        bbox[2] = bbox[2] / width
        bbox[3] = bbox[3] / height
        centermap = self.gen_centermap(bbox)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)

        return img, landmark, heatmaps, centermap
            
    def __len__(self):
        return len(self.img_paths)

    def crop_image(self, img, bbox, landmark):
        height, width, _ = img.shape

        x1, y1, w, h = bbox
        x, y = x1 + w / 2, y1 + h / 2
        size = max(w, h)
        if size < min(height, width):
            left_bound = max(x - size / 2, 0)
            right_bound = min(x + size / 2, width)
            upper_bound = max(y - size / 2, 0)
            lower_bound = min(y + size / 2, height)

            img = img[int(upper_bound):int(lower_bound), int(left_bound):int(right_bound)]

            bbox[0] = bbox[0] - left_bound
            bbox[1] = bbox[1] - upper_bound

            landmark[:, 0] = landmark[:, 0] - left_bound
            landmark[:, 1] = landmark[:, 1] - upper_bound

        return img, bbox, landmark

    def gen_heatmap(self, landmark):
        target = np.zeros((17, self.map_size, self.map_size), dtype=np.float32)
        tmp_size = self.sigma * 3

        for joint_id in range(17):
            mu_x = int(landmark[joint_id][0] * self.map_size + 0.5)
            mu_y = int(landmark[joint_id][1] * self.map_size + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= self.map_size or ul[1] >= self.map_size or br[0] < 0 or br[1] < 0:
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.map_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.map_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.map_size)
            img_y = max(0, ul[1]), min(br[1], self.map_size)

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        #background = np.expand_dims(1 - np.amax(target, axis=0), axis=0)
        #return np.concatenate([background, target])
        return target

    def gen_centermap(self, bbox):
        x1, y1, w, h = bbox * self.img_size
        center = torch.tensor([x1 + w / 2, y1 + h / 2], dtype=torch.float32).view(1, 2)
        
        grid_x = torch.arange(self.img_size).repeat(self.img_size, 1)
        grid_y = torch.arange(self.img_size).repeat(self.img_size, 1).t()
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0) # size:(1, self.map_size, self.map_size, 2)

        center = center.unsqueeze(-2).unsqueeze(-2)
        exponent = torch.sum((grid - center)**2, dim=-1)
        centermap = torch.exp(-exponent / 2.0 / self.sigma / self.sigma / 9)
        return centermap

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

        return img_paths, bboxes, landmarks, visibilities


def load_data(data_path, batch_size, img_size, stride, sigma, shuffle):
    dataset = MonkeyDataset(data_path, img_size, stride, sigma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataset, dataloader
