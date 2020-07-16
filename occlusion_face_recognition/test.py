#!/usr/bin/python3
# *_* coding: utf-8 *_*
# @Author: samon
# @Email: samonsix@163.com
# @IDE: PyCharm
# @File: test.py
# @Date: 19-8-5 下午7:30
# @Descr:
import torch
import torch.nn as nn
import h5py
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import os
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import heapq

# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
model_path = './model/Backbone_IR_SE_50_Epoch_4_Batch_45488_Time_2019-06-18-21-42_checkpoint.pth'
img_path = "/media/ext/CASIA-WebFace_face/0000045"
img_dir = "/media/ext/CASIA-WebFace_face"
feature_path = './features/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone_name = "IR_SE_50"


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, tta=True):
    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a = int((128 - 112) / 2)  # x start
    b = int((128 - 112) / 2 + 112)  # x end
    c = int((128 - 112) / 2)  # y start
    d = int((128 - 112) / 2 + 112)  # y end
    ccropped = resized[a:b, c:d]  # center crop the image
    ccropped = ccropped[..., ::-1]  # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype=np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype=np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    return features


def distance(feature, feature_path):
    total = []
    name = []
    d = []
    s = []
    for file in os.listdir(feature_path):
        if file.endswith('.npy'):
            feature_root = os.path.join(feature_path, file)
            feature_name = os.path.splitext(file)[0]
            feature2 = np.load(feature_root)
            f1 = feature.numpy()
            f2 = feature2
            diff = np.subtract(f1, f2)
            dist = np.sum(np.square(diff))
            # sim = np.dot(f1,f2.T)
            name.append(feature_name)
            # total_1 = dist + (1-sim)#dist越小，sim越接近1，即total越小，两图片为同一人的可能性越大
            d.append(dist)
            # s.append(sim)
            # total.append(total_1)
            # print(total)

    threshold = 1.46
    min_num_index_list = map(d.index, heapq.nsmallest(1, d))
    for n in list(min_num_index_list):
        if d[n] < threshold:
            print("name:{}\t" "dist:{}\t" "相似度：{}%\t".format(name[n], d[n],
                                                             (2 * threshold - d[n]) / (2 * threshold) * 100))
        else:
            print("None!")
    print('=' * 60)


def write_hdf5(data: dict):
    # Create a new file
    file = h5py.File('data.h5', 'w')
    for k, v in data.items():
        file.create_dataset(k, data=v)
    file.close()

    # f = h5py.File('data.h5', 'r')
    # print(f["feature"])
    # print(f["label"].value)
    # exit()


def get_backbone():
    cfg = configurations[1]
    # support: 'ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50',
    # 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152'
    INPUT_SIZE = cfg['INPUT_SIZE']
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                     'ResNet_101': ResNet_101(INPUT_SIZE),
                     'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE),
                     'IR_101': IR_101(INPUT_SIZE),
                     'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE),
                     'IR_SE_101': IR_SE_101(INPUT_SIZE),
                     'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[backbone_name]

    assert (os.path.exists(model_path))
    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_path))
    BACKBONE.load_state_dict(torch.load(model_path))
    BACKBONE.to(device)

    # extract features
    BACKBONE.eval()  # set to evaluation mode

    return BACKBONE


def getfeatures(backbone):
    feature = []
    label = []
    count = 0
    for root_d, sub_d, file_list in os.walk(img_dir):
        for f in file_list:
            if f.endswith("jpg"):
                img_path = os.path.join(root_d, f)
                feature.append(extract_feature(img_path, backbone))
                label.append(int(os.path.basename(root_d)))
        count += 1
        if count % 100 == 0:
            print("label:{}, dir:{}/{}".format(len(label), count, len(os.listdir(root_d))))

    feature = torch.cat(feature, 0).numpy()
    label = np.array(label)
    print(feature.shape)
    print(label.shape)
    write_hdf5({"feature": feature, "label": label})


def test(img_path, backbone):
    for file in os.listdir(img_path):
        if file.endswith('.jpg'):
            img_root = os.path.join(img_path, file)
            img_name = os.path.splitext(file)[0]
            feature = extract_feature(img_root, backbone)
            print("imagename:", img_name)
            distance(feature, feature_path)


if __name__ == "__main__":
    backbone = get_backbone()
    test('./testdata/', backbone)
