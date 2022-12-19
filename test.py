import torch
import numpy as np
import cv2
import os
import time
import yaml
import torch.nn.functional as F
from tqdm import tqdm

from libs.load import load_data
from libs.draw import draw_limbs, draw_joints
from libs.utils import get_max_preds
from libs.metrics import PCK

from model.HRNet import HRNet
from model.CPM import CPM
from model.CPHRNet import CPHRNet
from model.CPHRNetv2 import CPHRNetv2


class Test:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if configs['model_type'] == "CPM":
            self.model = CPM(self.configs['num_joints'])
        elif configs['model_type'] == "HRNet":
            self.model = HRNet(self.configs['num_channels'], self.configs['num_joints'])
        elif configs['model_type'] == "CPHRNet":
            self.model = CPHRNet(self.configs['num_channels'], self.configs['num_joints'])
        elif configs['model_type'] == "CPHRNetv2":
            self.model = CPHRNetv2(self.configs['num_channels'], 32, self.configs['num_joints'])
        else:
            raise NotImplementedError("Please specify model type in CPM or HRNet")

    def load_model(self):
        weight_path = os.path.join("weights", self.configs['model_name'] + ".pth")
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            assert False, "Model is not exist in {}".format(weight_path)

    def detect(self):
        print("Using device:", self.device)

        test_set, test_dataloader = load_data(
            self.configs['data_path'], 
            self.configs['preprocess'],
            self.configs['model_type'],
            self.configs['batch_size'], 
            self.configs['img_size'], 
            self.configs['num_joints'], 
            self.configs['sigma'], 
            "test"
        )
        print("The number of data in test set: ", test_set.__len__())

        self.model = self.model.to(self.device)
        self.load_model()
        self.model.eval()

        PCK_acc = np.zeros(self.configs['num_joints'] + 1)
        start_time = time.time()
        # --------------------------
        # Testing Stage
        # --------------------------
        with torch.no_grad():
            for i, (images, heatmaps, joints_weight, landmarks) in enumerate(tqdm(test_dataloader)):
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)

                output = None
                if self.configs['model_type'] in ["CPM", "CPHRNet"]:
                    _, _, _, _, _, output = self.model(images)
                elif self.configs['model_type'] == "CPHRNetv2":
                    _, _, output = self.model(images)
                else:
                    output = self.model(images)
                
                images[:, 0] = images[:, 0] * 0.229 + 0.485
                images[:, 1] = images[:, 1] * 0.224 + 0.456
                images[:, 2] = images[:, 2] * 0.225 + 0.406
                images = images * 255.0

                pred_landmarks, maxvals = get_max_preds(output.cpu().numpy())
                pred_landmarks = pred_landmarks * configs['img_size']
                landmarks = landmarks.numpy() * configs['img_size']

                acc = PCK(pred_landmarks, landmarks, 
                                self.configs['img_size'], self.configs['img_size'], self.configs['num_joints'])
                PCK_acc += np.array(acc)

                if self.configs['display_results']:
                    pred_maps = F.interpolate(output, size=(self.configs['img_size'], self.configs['img_size']), 
                                                mode='bilinear', align_corners=True)
                    targ_maps = F.interpolate(heatmaps, size=(self.configs['img_size'], self.configs['img_size']), 
                                                mode='bilinear', align_corners=True)

                    for i in range(self.configs['batch_size']):
                        img = images[i]
                        img = img.cpu().numpy().transpose(1, 2, 0)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        pred_img = img.copy()
                        targ_img = img.copy()

                        pred_heatmap = pred_maps[i].cpu().numpy().transpose(1, 2, 0)
                        targ_heatmap = targ_maps[i].cpu().numpy().transpose(1, 2, 0)

                        pred_landmark = pred_landmarks[i].astype(np.int32)
                        targ_landmark = landmarks[i].astype(np.int32)

                        pred_img = draw_limbs(pred_img, pred_landmark)
                        targ_img = draw_limbs(targ_img, targ_landmark)

                        pred_img = draw_joints(pred_img, pred_landmark)
                        targ_img = draw_joints(targ_img, targ_landmark)

                        for j in range(self.configs['num_joints']):
                            print(maxvals[i][j])

                            pred = pred_heatmap[:, :, j]
                            pred = cv2.normalize(pred, pred, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                        dtype=cv2.CV_8U)
                            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

                            targ = targ_heatmap[:, :, j]
                            targ = cv2.normalize(targ, targ, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, 
                                                        dtype=cv2.CV_8U)
                            targ = cv2.applyColorMap(targ, cv2.COLORMAP_JET)
                        
                            display1 = pred_img * 0.8 + pred * 0.2
                            display2 = targ_img * 0.8 + targ * 0.2

                            display = np.concatenate((display1, display2), axis=1).astype(np.uint8)
                            cv2.imshow("img", display)
                            key = cv2.waitKey(0)
                            if key == ord('q'):
                                print("quit display")
                                exit(1)

        end_time = time.time()

        PCK_acc /= test_dataloader.__len__()
        summary = {}
        summary["overall"] = PCK_acc[0]
        summary["head"] = PCK_acc[4]
        summary["neck"] = PCK_acc[5]
        summary["shoulder"] = (PCK_acc[6] + PCK_acc[9]) / 2
        summary["elbow"] = (PCK_acc[7] + PCK_acc[10]) / 2
        summary["wrist"] = (PCK_acc[8] + PCK_acc[11]) / 2
        summary["hip"] = PCK_acc[12]
        summary["knee"] = (PCK_acc[13] + PCK_acc[15]) / 2
        summary["angle"] = (PCK_acc[14] + PCK_acc[16]) / 2
        summary["tail"] = PCK_acc[17]

        print("Accuracy of pose estimation: {}, Testing cost {} sec(s) per image"
                .format(summary, (end_time - start_time) / test_set.__len__()))


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
