import torch
import numpy as np
import cv2
import os
import time
import yaml
import torch.nn.functional as F
from tqdm import tqdm

from libs.load import load_data
from model.CPM import CPM
from model.HRNet import HRNet


class Test:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model = CPM(self.configs['num_joints'])
        self.model = HRNet(32, self.configs['num_joints'])

    def load_model(self):
        weight_path = os.path.join("weights", self.configs['model_name'])
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            assert False, "Model is not exist in {}".format(weight_path)

        weight = torch.load(weight_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(weight)

    def detect(self):
        print("Using device:", self.device)

        test_set, test_dataloader = load_data(self.configs['data_path'], self.configs['batch_size'], self.configs['img_size'], 
                                                self.configs['stride'], self.configs['sigma'], False)
        print("The number of data in test set: ", test_set.__len__())

        self.load_model()
        self.model.eval()

        start_time = time.time()

        # --------------------------
        # Testing Stage
        # --------------------------
        with torch.no_grad():
            for i, (images, landmarks, heatmaps, centermaps) in enumerate(tqdm(test_dataloader)):    
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                centermaps = centermaps.to(self.device)
                #heat1, heat2, heat3, heat4, heat5, heat6 = self.model(images, centermaps)
                heat6 = self.model(images)
                

                images = images * 255.0
                pred_maps = heat6
                targ_maps = heatmaps
                pred_maps = F.interpolate(pred_maps, size=(self.configs['img_size'], self.configs['img_size']), 
                                            mode='bilinear', align_corners=True)
                targ_maps = F.interpolate(targ_maps, size=(self.configs['img_size'], self.configs['img_size']), 
                                            mode='bilinear', align_corners=True)

                for i in range(len(pred_maps)):
                    img = images[i]
                    img = img.cpu().numpy().transpose(1, 2, 0)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    preds = pred_maps[i]
                    preds = preds.cpu().numpy().transpose(1, 2, 0)
                    targs = targ_maps[i]
                    targs = targs.cpu().numpy().transpose(1, 2, 0)

                    kpts = []
                    for i in range(17):
                        m = preds[:, :, i]
                        h, w = np.unravel_index(m.argmax(), m.shape)
                        x = int(w * self.configs['img_size'] / m.shape[1])
                        y = int(h * self.configs['img_size'] / m.shape[0])
                        kpts.append([x,y])

                    for i in range(17):
                        pred = preds[:, :, i]
                        print(np.unique(pred))
                        pred = cv2.normalize(pred, pred, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                    dtype=cv2.CV_8U)
                        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
                        print(np.unique(pred))

                        targ = targs[:, :, i]
                        targ = cv2.normalize(targ, targ, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                    dtype=cv2.CV_8U)
                        targ = cv2.applyColorMap(targ, cv2.COLORMAP_JET)
                    
                        display1 = img * 0.8 + pred * 0.2
                        display2 = img * 0.8 + targ * 0.2

                        display1 = cv2.circle(display1, kpts[i], radius=1, thickness=5, color=(0, 0, 0))

                        display = np.concatenate((display1, display2), axis=1).astype(np.uint8)
                        cv2.imshow("img", display)
                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            print("quit display")
                            exit(1)


        end_time = time.time()

        print("Testing cost {} sec(s)".format(end_time - start_time))


if __name__ == "__main__":
    configs = None
    with open("configs/test.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(configs)
    t = Test(configs)
    t.detect()
