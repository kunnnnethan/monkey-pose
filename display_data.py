import numpy as np
import cv2
import yaml
from tqdm import tqdm
import torch.nn.functional as F

from libs.load import load_data
from libs.draw import draw_limbs, draw_joints


if __name__ == "__main__":
    configs = None
    with open("configs/train.yaml", "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_set, train_dataloader = load_data(
        configs['data_path'], 
        configs['model_type'], 
        configs['batch_size'], 
        configs['img_size'], 
        configs['num_joints'], 
        configs['sigma'], 
        False
    )

    print("length of train set: ", train_set.__len__())
    for _, (images, heatmaps, joints_weight, data) in enumerate(tqdm(train_dataloader)):
        images[:, 0] = images[:, 0] * 0.229 + 0.485
        images[:, 1] = images[:, 1] * 0.224 + 0.456
        images[:, 2] = images[:, 2] * 0.225 + 0.406
        images = images * 255.0

        landmarks = data['landmark']
        landmarks = landmarks * configs['img_size']
        heatmaps = F.interpolate(heatmaps, size=(configs['img_size'], configs['img_size']), mode='bilinear', align_corners=True)

        for j in range(configs['batch_size']):
            img = images[j].numpy().transpose(1, 2, 0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            landmark = landmarks[j].numpy().astype(np.int32)

            img = draw_limbs(img, landmark)
            img = draw_joints(img, landmark)

            heatmap = heatmaps[j].numpy().transpose(1, 2, 0)

            for i in range(configs['num_joints']):
                joint = heatmap[:, :, i]
                
                joint = cv2.normalize(joint, joint, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                joint = cv2.applyColorMap(joint, cv2.COLORMAP_JET)
                
                display = img * 0.8 + joint * 0.2
                cv2.imshow("img", display.astype(np.uint8))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("quit display")
                    exit(1)
   