import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F

from libs.load import load_data


if __name__ == "__main__":
    batch_size = 1
    img_size = 368
    stride = 8 
    sigma = 1
    train_set, train_dataloader = load_data("data/train", batch_size, img_size, stride, sigma, False)

    print("length of train set: ", train_set.__len__())
    for i, (images, landmarks, heatmaps, centermap) in enumerate(tqdm(train_dataloader)):
        images = images * 255.0
        images = images.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        landmarks = landmarks.squeeze(0).numpy()

        heatmaps = F.interpolate(heatmaps, size=(img_size, img_size), mode='bilinear', align_corners=True)
        heatmaps = heatmaps.squeeze(0).numpy().transpose(1, 2, 0)
        centermap = F.interpolate(centermap, size=(img_size, img_size), mode='bilinear', align_corners=True)
        centermap = centermap.squeeze(0).numpy().transpose(1, 2, 0)
        heatmaps = np.concatenate([centermap, heatmaps], axis=2)

        annotations = []
        for i in range(len(landmarks)):
            x, y = int(landmarks[i, 0] * img_size), int(landmarks[i, 1] * img_size)
            annotations.append((x, y))

        # eyes and nose
        images = cv2.line(images, annotations[0], annotations[2], (250, 230, 230), 2)
        images = cv2.line(images, annotations[1], annotations[2], (250, 230, 230), 2)
        # neck
        images = cv2.line(images, annotations[3], annotations[4], (0, 0, 255), 2)
        # body
        images = cv2.line(images, annotations[4], annotations[11], (0, 165, 255), 2)
        # tail
        images = cv2.line(images, annotations[11], annotations[16], (205, 250, 255), 2)
        # right hand
        images = cv2.line(images, annotations[4], annotations[5], (255, 0, 0), 2)
        images = cv2.line(images, annotations[5], annotations[6], (255, 0, 0), 2)
        images = cv2.line(images, annotations[6], annotations[7], (255, 0, 0), 2)
        # left hand
        images = cv2.line(images, annotations[4], annotations[8], (230, 216, 173), 2)
        images = cv2.line(images, annotations[8], annotations[9], (230, 216, 173), 2)
        images = cv2.line(images, annotations[9], annotations[10], (230, 216, 173), 2)
        # right leg
        images = cv2.line(images, annotations[11], annotations[12], (0, 100, 0), 2)
        images = cv2.line(images, annotations[12], annotations[13], (0, 100, 0), 2)
        # left leg
        images = cv2.line(images, annotations[11], annotations[14], (144, 238, 144), 2)
        images = cv2.line(images, annotations[14], annotations[15], (144, 238, 144), 2)

        # landmarks
        for a in annotations:
          images = cv2.circle(images, a, 1, (0, 0, 0), 3)

        for i in range(19):
            heatmap = heatmaps[:, :, i]
            
            heatmap = cv2.normalize(heatmap, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            display = images * 0.8 + heatmap * 0.2
            cv2.imshow("img", display.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("quit display")
                exit(1)
   