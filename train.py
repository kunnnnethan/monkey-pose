import torch, os, random
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from libs.load import load_data
from libs.loss import JointsMSELoss
from model.HRNet import HRNet


def init():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Train:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HRNet(32, self.configs['num_joints'])


    def train(self):
        init()
        print("Using device:", self.device)

        train_set, train_dataloader = load_data(
            self.configs['data_path'], 
            self.configs['batch_size'], 
            self.configs['img_size'], 
            self.configs['num_joints'], 
            self.configs['sigma'], 
            True
        )
        print("The number of data in train set: ", train_set.__len__())

        self.model = self.model.to(self.device)

        criterion = JointsMSELoss(self.configs['use_joints_weight'])

        # define loss function and optimizer
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configs['learning_rate'], 
        #                            momentum=self.configs['momentum'], weight_decay=self.configs['weight_decay'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])

        for epoch in range(self.configs['epochs']):
            train_loss, val_loss = 0, 0
            train_acc, val_acc = 0, 0

            self.model.train()
            for i, (images, heatmaps, joints_weight, data) in enumerate(tqdm(train_dataloader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                joints_weight = joints_weight.to(self.device)

                optimizer.zero_grad()
                pred = self.model(images)
                
                loss = criterion(pred, heatmaps, joints_weight)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                
            print("Epoch: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}"
                    .format(epoch + 1, train_loss, train_acc, val_loss, val_acc))
            torch.save(self.model.state_dict(), os.path.join("weights", self.configs['model_name']))


if __name__ == "__main__":
    configs = None
    with open("configs/train.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(configs)
    t = Train(configs)
    t.train()
