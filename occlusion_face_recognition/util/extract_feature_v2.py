# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from backbone.model_irse import IR_SE_101
import torch.nn.functional as F

def l2_norm(input, axis = 1):
    """
    @param input:
    @param axis:
    @return:
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img_root, backbone, model_root, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    # pre-requisites
    assert(os.path.exists(img_root))
    # print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    # print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)


    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features


def extractFeature(img_root, backbone, model_root,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    """
    @param img_root:
    @param backbone:
    @param model_root:
    @param device:
    @param tta:
    @return:
    """
    # pre-requisites
    assert (os.path.exists(img_root))
    # print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    # print('Backbone Model Root:', model_root)

    # load image
    img = cv2.imread(img_root)
    ccropped = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
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

    # load backbone from a checkpoint
    # print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    #     np.save("features.npy", features)
    #     features = np.load("features.npy")

    return features

if __name__ == "__main__":
    imageDirGZ = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
    imageDirGL = "/home/zhex/Documents/project/profileFace/pic/AI/zhengxiangzhong"
    # imageDirGL = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
    modelRoot = "/home/zhex/pre_models/AILuoGang/2020-07-06-05-55_IR_SE_101_Epoch_55_LOSS_0.001.pth"
    net = IR_SE_101([112])

    imageNamesGZ = os.listdir(imageDirGZ)
    imageNamesGZ.sort(key=lambda x: int(x[:-4]))
    lengthGZ = len(imageNamesGZ)

    imageNamesGL = os.listdir(imageDirGL)
    imageNamesGL.sort(key=lambda x: int(x[:-4]))
    lengthGL = len(imageNamesGL)

    for i in range(lengthGZ):
        imagePath1 = os.path.join(imageDirGZ, imageNamesGZ[i])
        feature1 = extract_feature(imagePath1, net, modelRoot)
        for j in range(lengthGL):
            imagePath2 = os.path.join(imageDirGL, imageNamesGL[j])
            feature2 = extract_feature(imagePath2, net, modelRoot)
            # feature2 = extractFeature(imagePath2, net, modelRoot)
            # print("feature1.size()=",feature1.size())
            # print("feature2.size()=",feature2.size())
            distance = feature1.mm(feature2.t())
            distance = round(distance.item(), 4)
            print("{} and {}'s distance=".format(imageNamesGZ[i], imageNamesGL[j]), distance)
