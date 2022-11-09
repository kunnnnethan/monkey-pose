import torch, os, random
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from libs.load import load_data
from model.CPM import CPM
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
        #self.model = CPM(self.configs['num_joints'])
        self.model = HRNet(32, self.configs['num_joints'])


    def train(self):
        init()
        print("Using device:", self.device)

        train_set, train_dataloader = load_data(self.configs['data_path'], self.configs['batch_size'], self.configs['img_size'], 
                                                self.configs['stride'], self.configs['sigma'], True)
        print("The number of data in train set: ", train_set.__len__())

        self.model = self.model.to(self.device)

        criterion = nn.MSELoss()

        # define loss function and optimizer
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configs['learning_rate'], 
        #                            momentum=self.configs['momentum'], weight_decay=self.configs['weight_decay'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['learning_rate'])

        for epoch in range(self.configs['epochs']):
            train_loss, val_loss = 0, 0
            train_acc, val_acc = 0, 0

            self.model.train()
            for i, (images, landmarks, heatmaps, centermaps) in enumerate(tqdm(train_dataloader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                centermaps = centermaps.to(self.device)

                optimizer.zero_grad()
                pred = self.model(images)
                
                loss = criterion(pred, heatmaps)
                #loss2 = criterion(heat2, heatmaps)
                #loss3 = criterion(heat3, heatmaps)
                #loss4 = criterion(heat4, heatmaps)
                #loss5 = criterion(heat5, heatmaps)
                #loss6 = criterion(heat6, heatmaps)
                #loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

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
