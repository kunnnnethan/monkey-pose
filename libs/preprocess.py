import numpy as np
import cv2


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center, scale, rot, original_size, output_size, shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(original_size, np.ndarray) and not isinstance(output_size, np.ndarray):
        assert False, "original_size and output_size should be 2-d numpy arrays"

    original_size *= scale

    src_center = center
    dst_center = np.array([output_size[0] * 0.5, output_size[1] * 0.5])

    rot_rad = np.pi * rot / 180
    size = max(original_size[0], original_size[1])
    src_w = get_dir([size * -0.5, 0], rot_rad)
    src_h = get_dir([0, size * -0.5], rot_rad)

    dst_w = np.array([output_size[0] * -0.5, 0], np.float32)
    dst_h = np.array([0, output_size[1] * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = src_center + original_size * shift
    src[1, :] = src_center + src_w + original_size * shift
    src[2, :] = src_center + src_h + original_size * shift

    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_w
    dst[2, :] = dst_center + dst_h

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


class MonkeyPreprocess:
    def __init__(self, img_size, num_joints, preprocess):
        self.img_size = img_size
        self.num_joints = num_joints

        # augment images by rotating images randomly with angles between [-30, 30]
        self.rotate = preprocess['rotate']
        # augment images by scaling images randomly between [0.75, 1.25]
        self.scale = preprocess['scale']
        # augment images by flipping images horizontally
        self.horizontal_flip = preprocess['horizontal_flip']
        # augment images by modifying hsv channels of images
        self.hsv = preprocess['hsv']

        self.matched_joints = [[0, 1], [5, 8], [6, 9], [7, 10], [12, 14], [13, 15]]

    def apply(self, img, bbox, landmark, joints_weight, visibility):
        if self.horizontal_flip and np.random.rand() > 0.5:
            img, bbox, landmark, visibility = self.horizontal_flip_(img, bbox, landmark, visibility)
        if self.hsv and np.random.rand() > 0.5:
            img = self.hsv_(img)
        
        img, landmark, joints_weight = self.process_image(img, bbox, landmark, joints_weight)

        return img, landmark, joints_weight, visibility

    def process_image(self, img, bbox, landmark, joints_weight):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        original_size = np.array([w, h], dtype=np.float32)
        output_size = np.array([self.img_size, self.img_size], dtype=np.float32)

        scale = 1.1
        rotate = 0

        if self.rotate:
            rotate = rotate + (np.random.rand() - 0.5) * 30
        if self.scale:
            scale = scale * ((np.random.rand() - 0.5) * 0.5 + 1)
            
        trans = get_affine_transform(center, scale, rotate, original_size, output_size)
        img = cv2.warpAffine(img, trans, (int(self.img_size), int(self.img_size)), flags=cv2.INTER_LINEAR)

        landmark = cv2.transform(landmark.reshape(1, self.num_joints, 2), trans).reshape(self.num_joints, 2)
        landmark[:, 0] = landmark[:, 0] / self.img_size
        landmark[:, 1] = landmark[:, 1] / self.img_size

        mask = (landmark[:, 0] > 1) | (landmark[:, 0] < 0) | (landmark[:, 1] > 1) | (landmark[:, 1] < 0)
        joints_weight[mask] *= 0.0

        return img, landmark, joints_weight

    def horizontal_flip_(self, img, bbox, landmark, visibility):
        img = np.flip(img, axis=1)
        width = img.shape[1]
        bbox[0] = width - bbox[0] - bbox[2]
        landmark[:, 0] = width - landmark[:, 0]

        for matched in self.matched_joints:
            left, right = landmark[matched[0]].copy(), landmark[matched[1]].copy()
            landmark[matched[0]], landmark[matched[1]] = right, left

            left, right = visibility[matched[0]].copy(), visibility[matched[1]].copy()
            visibility[matched[0]], visibility[matched[1]] = right, left

        return img, bbox, landmark, visibility

    def hsv_(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(img)

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(np.uint8)
            lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
            lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img
