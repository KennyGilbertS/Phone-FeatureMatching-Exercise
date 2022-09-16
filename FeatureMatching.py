# Kenny Gilbert Setiawan - 2301898960
# Computer Vision QUIZ BB02

import cv2 as cv
import os
from matplotlib import pyplot as plt

base_path = 'Dataset/data'

inter_image = cv.imread('Dataset/target.jpg')
inter_image = cv.cvtColor(inter_image, cv.COLOR_BGR2RGB)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))

scene_images = []
imagelmao= []

for i in os.listdir(base_path):
    image2 = cv.imread(base_path + '/' + i)
    image_path = cv.imread(base_path + '/' + i)
    image_path = cv.cvtColor(image_path, cv.COLOR_BGR2GRAY)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
    image_path = clahe.apply(image_path)
    scene_images.append(image_path)
    imagelmao.append(image2)

SIFT = cv.xfeatures2d.SIFT_create()


inter_kp, inter_ds = SIFT.detectAndCompute(inter_image, None)

KDTREE_INDEX = 1 
TREE_CHECKS = 50

FLANN = cv.FlannBasedMatcher(dict(algorithm = KDTREE_INDEX), dict(checks = TREE_CHECKS))

all_mask = []
scene_index = -1
total_match = 0
scene_keypoints = None
final_match = None

for index, i in enumerate(scene_images):
    scene_kp, scene_ds = SIFT.detectAndCompute(i, None)
    matcher = FLANN.knnMatch(scene_ds, inter_ds, 2)
    match_count = 0
    scene_mask = [[0,0] for j in range(0, len(matcher))]

    for j, (m,n) in enumerate(matcher):
        if m.distance < 0.68 * n.distance:
            scene_mask[j] = [1,0]
            match_count +=1

    all_mask.append(scene_mask)
    if total_match < match_count:
        total_match = match_count
        scene_index = index
        scene_keypoints = scene_kp
        final_match = matcher
        
result = cv.drawMatchesKnn(
    imagelmao[scene_index], scene_keypoints,
    inter_image, inter_kp,
    final_match, None, matchColor=[255,0,0],
    matchesMask=all_mask[scene_index]
)

plt.title("Best Match Result")
plt.imshow(result, cmap="gray")
plt.show()
