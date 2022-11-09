import cv2


def draw_limbs(img, annotations):
    limbs = [
        [0, 2], [1, 2], # face
        [3, 4], # neck
        [4, 11], # body
        [11, 16], # tail
        [4, 5], [5, 6], [6, 7], # right hand
        [4, 8], [8, 9], [9, 10], # left hand
        [11, 12], [12, 13], # right leg
        [11, 14], [14, 15] # left leg
    ]

    colors = [
        (250, 230, 230), (250, 230, 230),
        (0, 0, 255),
        (0, 165, 255),
        (205, 250, 255),
        (255, 0, 0), (255, 0, 0), (255, 0, 0),
        (230, 216, 173), (230, 216, 173), (230, 216, 173),
        (0, 100, 0), (0, 100, 0),
        (144, 238, 144), (144, 238, 144)
    ]

    for (l, c) in zip(limbs, colors):
        img = cv2.line(img, annotations[l[0]], annotations[l[1]], c, 2)
    
    return img


def draw_joints(img, annotations):
    for a in annotations:
        img = cv2.circle(img, a, 1, (0, 0, 0), 3)

    return img
