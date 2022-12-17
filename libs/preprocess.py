import numpy as np
import cv2


def pad_to_square(img, pad_value):
    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # Determine padding
    pad_size = dim_diff // 2
    pad = ((pad_size, pad_size), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad_size, pad_size), (0, 0))
    # Add padding
    img = np.pad(img, pad, 'constant', constant_values=pad_value)

    return img, pad

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class MonkeyPreprocess:
    def __init__(self, img_size, num_joints, preprocess):
        self.img_size = img_size
        self.num_joints = num_joints

        # augment images by rotate images with random angles between [-30, 30)
        self.rotate = preprocess['rotate']
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
        #img, landmark, joints_weight = self.process_image_(img, bbox, landmark, joints_weight)

        return img, landmark, joints_weight, visibility

    def process_image(self, img, bbox, landmark, joints_weight):
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        scale = np.array([w, h], dtype=np.float32) * 1.25
        rotate = 0

        if self.rotate:
            rotate = (np.random.rand() - 0.5) * 30
        if self.scale:
            scale = scale * ((np.random.rand() - 0.5) * 0.5 + 1)
        trans = get_affine_transform(center, scale, rotate, [self.img_size, self.img_size])
        img = cv2.warpAffine(
            img,
            trans,
            (int(self.img_size), int(self.img_size)),
            flags=cv2.INTER_LINEAR)

        landmark = cv2.transform(landmark.reshape(1, self.num_joints, 2), trans).reshape(self.num_joints, 2)
        landmark[:, 0] = landmark[:, 0] / self.img_size
        landmark[:, 1] = landmark[:, 1] / self.img_size

        mask = (landmark[:, 0] > 1) | (landmark[:, 0] < 0) | (landmark[:, 1] > 1) | (landmark[:, 1] < 0)
        joints_weight[mask] *= 0.0

        return img, landmark, joints_weight


    def process_image_(self, img, bbox, landmark, joints_weight):
        x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x3, y3 = x1 + w, y1 + h # bottom right corner
  
        img, pad = pad_to_square(img[y1:y3, x1:x3], 0)
        size = max(w, h)
        w, h = size, size

        # adjust coordinate of landmarks
        landmark[:, 0] = landmark[:, 0] - x1 + pad[1][0]
        landmark[:, 1] = landmark[:, 1] - y1 + pad[0][0]

        if self.rotate:
            height, width, _ = img.shape
            cx = (width - 1) / 2.
            cy = (height - 1) / 2.

            angle = (np.random.rand() - 0.5) * 30
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

            img = cv2.warpAffine(img, M, (width, height))
            landmark = cv2.transform(landmark.reshape(1, self.num_joints, 2), M).reshape(self.num_joints, 2)

        # resize images
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        # normalize landmark with respect to image size
        landmark[:, 0] = landmark[:, 0] / w
        landmark[:, 1] = landmark[:, 1] / h

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
