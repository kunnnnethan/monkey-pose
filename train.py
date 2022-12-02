import torch, os, random
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from libs.load import load_data
from libs.loss import JointsMSELoss
from model.HRNet import HRNet
from model.CPM import CPM
from model.CPM2 import CPM2


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
        self.model = None
        if configs['model_type'] == "CPM":
            self.model = CPM(self.configs['num_joints'])
        elif configs['model_type'] == "HRNet":
            self.model = HRNet(48, self.configs['num_joints'])
        elif configs['model_type'] == "custom":
            self.model = CPM2(self.configs['num_joints'], 48, 32)
        else:
            raise NotImplementedError("Please specify model type in CPM or HRNet")
        if not os.path.exists("weights/"):
            os.mkdir("weights/")


    def train(self):
        init()
        print("Using device:", self.device)

        train_set, train_dataloader = load_data(
            self.configs['data_path'], 
            self.configs['model_type'],
            self.configs['batch_size'], 
            self.configs['img_size'], 
            self.configs['num_joints'], 
            self.configs['sigma'],
            True
        )
        print("The number of data in train set: ", train_set.__len__())

        self.model = self.model.to(self.device)

        criterion = None
        if self.configs['model_type'] == "CPM":
            criterion = nn.MSELoss()
        else:
            criterion = JointsMSELoss(self.configs['use_joints_weight'])

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

                loss = None
                if self.configs['model_type'] == "CPM":
                    pred1, pred2, pred3, pred4, pred5, pred6 = self.model(images)

                    loss1 = criterion(pred1, heatmaps)
                    loss2 = criterion(pred2, heatmaps)
                    loss3 = criterion(pred3, heatmaps)
                    loss4 = criterion(pred4, heatmaps)
                    loss5 = criterion(pred5, heatmaps)
                    loss6 = criterion(pred6, heatmaps)
                    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                elif self.configs['model_type'] == "custom":
                    #pred1, pred2, pred3, pred4, pred5, pred6 = self.model(images)
                    pred1, pred2, pred3, pred4 = self.model(images)
                    #heatmaps1 = heatmaps[:, :self.configs['num_joints']]
                    #heatmaps2 = heatmaps[:, self.configs['num_joints']:self.configs['num_joints'] * 2]
                    #heatmaps3 = heatmaps[:, self.configs['num_joints'] * 2:]

                    loss1 = criterion(pred1, heatmaps, joints_weight)
                    loss2 = criterion(pred2, heatmaps, joints_weight)
                    loss3 = criterion(pred3, heatmaps, joints_weight)
                    loss4 = criterion(pred4, heatmaps, joints_weight)
                    #loss4 = criterion(pred4, heatmaps, joints_weight)
                    #loss5 = criterion(pred5, heatmaps, joints_weight)
                    #loss6 = criterion(pred6, heatmaps, joints_weight)
                    #loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                    loss = loss1 + loss2 + loss3 + loss4
                else:
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
