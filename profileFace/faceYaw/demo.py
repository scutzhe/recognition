import os
import cv2
import csv
import numpy as np
from math import tan, cos, sin, asin, log, sqrt, pow
from fsaLib.FSANET_model import *
from face_detection import faceDetectionCenterFace,faceDetectionCenterMutilFace
from keras.layers import Average
from tqdm import tqdm


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):
    print("yaw,roll,pitch=",yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def sigmoid(x):
    return 1/(1+np.exp(-x))



def newAngle(yaw,pitch):
    """
    :param yaw:
    :param pitch:
    :return:
    """
    pitch_ = pitch * np.pi / 180
    yaw_ = yaw * np.pi / 180
    middleValue1 = 1 + pow(sin(yaw_)/tan(pitch_),2)
    middleValue2 = sin(pitch_) * sqrt(middleValue1)
    middleValue3 = asin(middleValue2)
    angle = abs(round(180 / np.pi * middleValue3,2))
    return angle


def yawCoefficient(yaw:float):
    """
    @param yaw:
    @return:
    """
    midValue = yaw / 45 - 1
    coefficient = sigmoid(midValue)
    return coefficient

def angleNoDetection(img,):
    """
    :param img:
    :return:
    """
    # angle(detected,img,faces,ad,img_size,img_w,img_h,model)
    imgNew = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    imgNew = np.expand_dims(imgNew,axis=0)
    p_result = model.predict(imgNew)
    yaw = p_result[0][0]
    pitch = p_result[0][1]
    roll = p_result[0][2]
    # return round(yaw,2), round(pitch,2), round(roll,2)
    return format(yaw,".6f")


def angle(detected,input_img,faces,ad,img_size,img_w,img_h,model):
    """
    @param detected:
    @param input_img:
    @param faces:
    @param ad: 0.6
    @param img_size:
    @param img_w:
    @param img_h:
    @param model:
    @return:
    """
    yaw, pitch, roll = 0, 0, 0
    if detected.shape[2] > 0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY

                x2 = x1 + w
                y2 = y1 + h

                # amplify face box
                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                if xw1 > xw2 or yw1 > yw2:
                    continue
                # print("xw1,yw1,xw2,yw2=",xw1,yw1,xw2,yw2)
                # cv2.rectangle(img,(xw1,yw1),(xw2,yw2),(255,0,0),2)
                # cv2.imshow("img",img)
                # cv2.waitKey(5000)

                faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                # print("face.shape=",face.shape)
                p_result = model.predict(face)
                yaw = p_result[0][0]
                pitch = p_result[0][1]
                roll = p_result[0][2]
        return yaw,pitch,roll

def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,model,time_detection,time_network,time_plot):
    
    # loop over the detections
    if detected.shape[2]>0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                # print((startX, startY, endX, endY))
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY
                
                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                
                faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
                
                face = np.expand_dims(faces[i,:,:,:], axis=0)
                p_result = model.predict(face)
                face = face.squeeze() # squeeze dims=1
                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])
                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
                
    return input_img #,time_network,time_plot

def createCapsuleModel(weightPath1,weightPath2,weightPath3):
    """
    @param weightPath1:
    @param weightPath2:
    @param weightPath3:
    @return:
    """
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')
    model1.load_weights(weightPath1)
    print('Finished loading model 1.')
    
    model2.load_weights(weightPath2)
    print('Finished loading model 2.')

    model3.load_weights(weightPath3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)
    return model


def createMetricModel(weightPath1, weightPath2, weightPath3):
    """
    @param weightPath1:
    @param weightPath2:
    @param weightPath3:
    @return:
    """
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7 * 3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    model3 = FSA_net_noS_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')
    model1.load_weights(weightPath1)
    print('Finished loading model 1.')

    model2.load_weights(weightPath2)
    print('Finished loading model 2.')

    model3.load_weights(weightPath3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)
    return model


def angleDetection(img:np.array):
    """
    @param img:
    @return:
    """
    try:
        os.mkdir('./yawResult')
    except OSError:
        pass
    img_size = 64
    ad = 0.6
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print('Starting detecting pose ...')
    detected_pre = np.empty((1,1,1))

    img_h, img_w = img.shape[:2]
    # detect faces using LBP detector
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detected = net.forward()
    # print("detected.shape=",detected.shape)

    if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
        detected = detected_pre
    faces = np.empty((detected.shape[2], img_size, img_size, 3))
    yaw, pitch, roll = angle(detected,img,faces,ad,img_size,img_w,img_h,model)
    # coefficient = yawCoefficient(yaw)
    # return coefficient
    yaw = round(yaw,2)
    return yaw

def getPreYaw(rootPath,preTxt):
    """
    @param rootPath:
    @param fileTxt:
    @return:
    """
    for root, dirs, names in os.walk(rootPath):
        for name in names:
            if name.endswith(".png") or name.endswith(".jpg"):
                imgPath = os.path.join(root,name)
                print("imgPath=",imgPath)
                writeName = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
                try:
                    img = cv2.imread(imgPath)
                    yaw = angleDetection(img)
                    preTxt.write(writeName + " " + str(yaw) + "\n")
                    preTxt.flush()
                except Exception as e:
                    print(e)
def recover(x):
    """
    :param x:
    :return:
    """
    radian = (log(x/(1-x)) + 1) * np.pi / 4
    angle = radian * 180 / np.pi
    return angle

# if __name__ == "__main__":
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     imgPath = "testImage/1.jpg"
#     imgBGR = cv2.imread(imgPath)
#     imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#     # yaw = angleDetection(imgRGB)
#     yaw = angleNoDetection(imgRGB)
#     print("yaw=",yaw)

# if __name__ == "__main__":
#     ### 300W_LP
#     ## capsule
#     # weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     # weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     # weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#
#     ## metric
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_metric_3_16_2_21_5/fsanet_metric_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_metric_3_16_2_21_5/fsanet_var_metric_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_metric_3_16_2_192_5/fsanet_noS_metric_3_16_2_192_5.h5'
#
#
#     # model = createCapsuleModel(weightPath1,weightPath2,weightPath3)
#     model = createMetricModel(weightPath1,weightPath2,weightPath3)
#     frontalFaceDir = "/home/zhex/test_result/faceYaw/luogangjiankongtest/frontal"
#     profileFaceDir = "/home/zhex/test_result/faceYaw/luogangjiankongtest/profile"
#     frontalNames = os.listdir(frontalFaceDir)
#     profileNames = os.listdir(profileFaceDir)
#     totalNumFrontal = len(frontalNames)
#     totalNumProfile = len(profileNames)
#     numFrontal = 0
#     numProfile = 0
#     numF =0
#     numP = 0
#     num = 0
#     total = totalNumFrontal + totalNumProfile
#
#     for imageName in tqdm(frontalNames):
#         imagePath = os.path.join(frontalFaceDir, imageName)
#         image = cv2.imread(imagePath)
#         h, w = image.shape[:2]
#         image = image[h // 4:3 * h // 4, w // 4:3 * w // 4, :]
#         flag = min(image.shape[0], image.shape[1])
#         if flag < 40:
#             numF += 1
#             continue
#         imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         yaw,_,_ = angleNoDetection(imgRGB)
#         if abs(yaw) <= 20:
#             numFrontal += 1
#
#     for imageName in tqdm(profileNames):
#         imagePath = os.path.join(profileFaceDir, imageName)
#         image = cv2.imread(imagePath)
#         h, w = image.shape[:2]
#         image = image[h // 4:3 * h // 4, w // 4:3 * w // 4, :]
#         flag = min(image.shape[0], image.shape[1])
#         if flag < 40:
#             numP += 1
#             continue
#         imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         yaw,_,_ = angleNoDetection(imgRGB)
#         if abs(yaw) > 20:
#             numProfile += 1
#
#     num = numFrontal + numProfile
#     print("faceYaw")
#     print("numFrontal/(totalNumFrontal - numF)=", numFrontal / (totalNumFrontal - numF))
#     print("numProfile/(totalNumProfile - numP)=", numProfile / (totalNumProfile - numP))
#     print("num/(total - numF - numP)=", round(num / (total - numF - numP), 4))


## face classification
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     # videoDir = "/home/zhex/Videos/profileFace/hall/test"
#     # videoNames = os.listdir(videoDir)
#     # videoNames.sort(key=lambda x:int(x[:-4]))
#     # for videoName in videoNames:
#     #     videoPath = os.path.join(videoDir,videoName)
#     #     print("videoPath=",videoPath)
#     videoPath = "/home/zhex/Videos/profileFace/hall/test/01.mp4"
#     videoName = "01.mp4"
#     vid = cv2.VideoCapture(videoPath)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # out = cv2.VideoWriter(videoName,fourcc,10,(2560,1440))
#     out = cv2.VideoWriter(videoName,fourcc,10,(1920,1080))
#     while True:
#         flag, frame = vid.read()
#         # print("frame.shape=",frame.shape)
#         if flag:
#             imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             imgDict = faceDetectionCenterMutilFace(imgRGB)
#             if len(imgDict)>0:
#                for key,value in imgDict.items():
#                     yaw, pitch, roll = angleNoDetection(value)
#                     # print("yaw=",yaw)
#                     # txt = str(yaw) + "_" + str(pitch) + "_" + str(roll)
#                     txt =  0 if abs(yaw) < 20 else 1
#                     cv2.rectangle(frame,(2*key[0],2*key[1]),(2*key[2],2*key[3]),(0,0,255),2)
#                     cv2.putText(frame,str(txt),(2*key[0],2*key[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#                     out.write(frame)
#                     cv2.imshow("frame",frame)
#                     cv2.waitKey(1)
#             else:
#                 continue
#         else:
#             break

## face classification
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     # videoDir = "/home/zhex/Videos/profileFace/hall/test"
#     # videoNames = os.listdir(videoDir)
#     # videoNames.sort(key=lambda x:int(x[:-4]))
#     # for videoName in videoNames:
#     #     videoPath = os.path.join(videoDir,videoName)
#     #     print("videoPath=",videoPath)
#     videoPath = "/home/zhex/Videos/profileFace/hall/test/06.mp4"
#     videoName = videoPath.split("/")[-1]
#     vid = cv2.VideoCapture(videoPath)
#     # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     # out = cv2.VideoWriter(videoName,fourcc,10,(2560,1440))
#     out = cv2.VideoWriter(videoName,fourcc,10,(1920,1080))
#     while True:
#         flag, frame = vid.read()
#         # print("frame.shape=",frame.shape)
#         if flag:
#             imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             imgDict = faceDetectionCenterMutilFace(imgRGB)
#             if len(imgDict)>0:
#                for key,value in imgDict.items():
#                     yaw, pitch, roll = angleNoDetection(value)
#                     newYaw = newAngle(yaw,pitch)
#                     # print("newYaw=",newYaw)
#                     txt =  abs(round(180 / np.pi * newYaw,2))
#                     txt = 0 if txt < 20 else 1
#                     cv2.rectangle(frame,(2*key[0],2*key[1]),(2*key[2],2*key[3]),(0,0,255),2)
#                     cv2.putText(frame,str(txt),(2*key[0],2*key[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#                     out.write(frame)
#                     cv2.imshow("frame",frame)
#                     cv2.waitKey(1)
#             else:
#                 continue
#         else:
#             break


## face crop
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     # videoDir = "/home/zhex/Videos/profileFace/hall/test"
#     # videoNames = os.listdir(videoDir)
#     # videoNames.sort(key=lambda x:int(x[:-4]))
#     # for videoName in videoNames:
#     #     videoPath = os.path.join(videoDir,videoName)
#     #     print("videoPath=",videoPath)
#     videoPath = "/home/zhex/Videos/profileFace/luogang/3m/不戴口罩_h3米d1米_d2米_d3米.mp4"
#     vid = cv2.VideoCapture(videoPath)
#     frontalSaveDir = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     profileSaveDir = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     indexF = 0
#     indexP = 0
#     step = 0
#     while True:
#         flag, frame = vid.read()
#         # print("frame.shape=",frame.shape)
#         if flag:
#             step += 1
#             if step % 1 == 0:
#                 imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#                 imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#                 imgDict = faceDetectionCenterMutilFace(imgRGB)
#                 if len(imgDict)>0:
#                    for key,value in imgDict.items():
#                         yaw, pitch, roll = angleNoDetection(value)
#                         newYaw = newAngle(yaw,pitch)
#                         txt =  abs(round(180 / np.pi * newYaw,2))
#                         flag = 0 if txt < 20 else 1
#                         print("flag=",flag)
#                         x1 = 2 * key[0]
#                         y1 = 2 * key[1]
#                         x2 = 2 * key[2]
#                         y2 = 2 * key[3]
#                         if flag == 0:
#                             if not os.path.exists(frontalSaveDir):
#                                 os.makedirs(frontalSaveDir)
#                             frontalFace = frame[y1:y2,x1:x2,:]
#                             cv2.imwrite(os.path.join(frontalSaveDir,"{}.png".format(indexF)),frontalFace)
#                             indexF += 1
#                         else:
#                             if not os.path.exists(profileSaveDir):
#                                 os.makedirs(profileSaveDir)
#                             profileFace = frame[y1:y2, x1:x2, :]
#                             cv2.imwrite(os.path.join(profileSaveDir,"{}.png".format(indexP)), profileFace)
#                             indexP += 1
#                 else:
#                     continue
#         else:
#             break


## test on luogang monitor videos
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     videoPath = "/home/zhex/Videos/profileFace/luogang/3m/不戴口罩_h3米d1米_d2米_d3米.mp4"
#     vid = cv2.VideoCapture(videoPath)
#     videoName = videoPath.split("/")[-1]
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(videoName, fourcc, 10, (2560, 1440))
#     while True:
#         flag, frame = vid.read()
#         # print("frame.shape=",frame.shape)
#         if flag:
#             imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             imgDict = faceDetectionCenterMutilFace(imgRGB)
#             if len(imgDict)>0:
#                for key,value in imgDict.items():
#                     yaw, pitch, roll = angleNoDetection(value)
#                     β = newAngle(yaw,pitch)
#                     # print("β, yaw=",β, yaw)
#                     txt = str(β) + "_" + str(yaw)
#                     print("txt=",txt)
#                     x1 = 2 * key[0]
#                     y1 = 2 * key[1]
#                     x2 = 2 * key[2]
#                     y2 = 2 * key[3]
#                     # x1 = key[0]
#                     # y1 = key[1]
#                     # x2 = key[2]
#                     # y2 = key[3]
#                     cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
#                     cv2.putText(frame,txt,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#                     # out.write(frame)
#                     cv2.imshow("frame",imgBGR)
#                     cv2.waitKey(1)
#             else:
#                 continue
#         else:
#             break


# multi faces
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     videoDir = "/home/zhex/Videos/profileFace/hall/test"
#     videoNames = os.listdir(videoDir)
#     videoNames.sort(key=lambda x:int(x[:-4]))
#     for videoName in videoNames:
#         videoPath = os.path.join(videoDir,videoName)
#         print("videoPath=",videoPath)
#     # videoPath = "/home/zhex/Videos/profileFace/06.avi"
#     # videoName = "06.mp4"
#         vid = cv2.VideoCapture(videoPath)
#         # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         # out = cv2.VideoWriter(videoName,fourcc,10,(2560,1440))
#         # out = cv2.VideoWriter(videoName,fourcc,10,(1920,1080))
#         while True:
#             flag, frame = vid.read()
#             # print("frame.shape=",frame.shape)
#             if flag:
#                 imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#                 imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#                 imgDict = faceDetectionCenterMutilFace(imgRGB)
#                 if len(imgDict)>0:
#                    for key,value in imgDict.items():
#                         yaw, pitch, roll = angleNoDetection(value)
#                         # print("yaw=",yaw)
#                         txt = str(yaw) + "_" + str(pitch) + "_" + str(roll)
#                         cv2.rectangle(frame,(2*key[0],2*key[1]),(2*key[2],2*key[3]),(0,0,255),2)
#                         cv2.putText(frame,txt,(2*key[0]-400,2*key[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#                         # out.write(frame)
#                         cv2.imshow("frame",frame)
#                         cv2.waitKey(1000)
#                 else:
#                     continue
#             else:
#                 break

## multi faces
# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     videoDir = "/home/zhex/Videos/profileFace/outdoor"
#     # for videoName in os.listdir(videoDir):
#     #     videoPath = os.path.join(videoDir,videoName)
#     videoPath = "/home/zhex/Videos/profileFace/outdoor/709.mp4"
#     videoName = "709_2.mp4"
#     vid = cv2.VideoCapture(videoPath)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(videoName,fourcc,25,(2560,1440))
#     while True:
#         flag, frame = vid.read()
#         # print(frame.shape)
#         if flag:
#             imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             imgDict = faceDetectionCenterMutilFace(imgRGB)
#             if len(imgDict)>0:
#                for key,value in imgDict.items():
#                     yaw = angleNoDetection(value)
#                     print("yaw=",yaw)
#                     cv2.putText(frame,str(yaw),(2*key[0],2*key[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
#                     out.write(frame)
#                     # cv2.imshow("frame",frame)
#                     # cv2.waitKey(1)
#             else:
#                 continue
#         else:
#             break

## newest
# if __name__ == '__main__':
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#
#     trainRootPath = "/home/zhex/data/profileNew/train"
#     trainImage = open("trainImage.txt","a")
#     trainLable = open("trainLabel.txt","a")
#
#     num_act_train = 0
#     num_pre_train = 0
#
#     IDsTrain = os.listdir(trainRootPath)
#     IDsTrain.sort(key=lambda x: int(x))
#     for id in tqdm(IDsTrain):
#         imgDir = os.path.join(trainRootPath, id)
#         # print("imgDir=",imgDir)
#         imgPaths = os.listdir(imgDir)
#         # print("imgPaths=",imgPaths)
#         imgPaths.sort(key=lambda x:int(x[:-4]))
#         # print("imgPaths=", imgPaths)
#         for imgName in imgPaths:
#             imgPath = os.path.join(imgDir,imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2])
#             # print("ID=",ID)
#             writePath = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             trainImage.write(writePath + "\n")
#             trainImage.flush()
#             num_act_train += 1
#             imgBGR = cv2.imread(imgPath)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             try:
#                 # coefficient = angleDetection(imgRGB)
#                 yaw = angleNoDetection(imgRGB)
#                 coefficient = yawCoefficient(abs(yaw))
#                 # print("coefficient=",coefficient)
#                 num_pre_train += 1
#                 trainLable.write(ID + " " + str(coefficient)+ "\n")
#                 trainLable.flush()
#             except Exception as e:
#                 print(e)
#
#     print("num_act_train=", num_act_train)
#     print("num_pre_train=", num_pre_train)
#
#     valRootPath = "/home/zhex/data/profileNew/val"
#     valImage = open("valImage.txt","a")
#     valLable = open("valLabel.txt","a")
#     num_act_val = 0
#     num_pre_val = 0
#     IDsVal = os.listdir(valRootPath)
#     IDsVal.sort(key=lambda x: int(x))
#     for id in tqdm(IDsVal):
#         imgDir = os.path.join(valRootPath, id)
#         # print("imgDir=",imgDir)
#         imgPaths = os.listdir(imgDir)
#         # print("imgPaths=",imgPaths)
#         imgPaths.sort(key=lambda x: int(x[:-4]))
#         # print("imgPaths=", imgPaths)
#         for imgName in imgPaths:
#             imgPath = os.path.join(imgDir, imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2])
#             # print("ID=",ID)
#             writePath = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             valImage.write(writePath + "\n")
#             valImage.flush()
#             num_act_val += 1
#             imgBGR = cv2.imread(imgPath)
#             imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
#             try:
#                 # coefficient = angleDetection(imgRGB)
#                 yaw = angleNoDetection(imgRGB)
#                 coefficient = yawCoefficient(abs(yaw))
#                 # print("coefficient=",coefficient)
#                 num_pre_val += 1
#                 valLable.write(ID + " " + str(coefficient) + "\n")
#                 valLable.flush()
#             except Exception as e:
#                 print(e)
#     print("num_act_val=",num_act_val)
#     print("num_pre_val=",num_pre_val)


## deal vgg_face2's dataset
if __name__ == '__main__':
    weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model = createCapsuleModel(weightPath1,weightPath2,weightPath3)
    # rootPath = "/home/zhex/Downloads/data/train"
    rootPath = "/home/zhex/Downloads/data/test"
    # csv_path = "/home/zhex/Downloads/data/bb_landmark/loose_bb_train.csv"
    csv_path = "/home/zhex/Downloads/data/bb_landmark/loose_bb_test.csv"
    filex = open(csv_path,"r")
    next(filex)
    infos = filex.readlines()
    # annotation_path = "/home/zhex/Downloads/data/annotations/train"
    annotation_path = "/home/zhex/Downloads/data/annotations/test"
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    IDs = os.listdir(rootPath)
    IDs.sort(key=lambda x: int(x[1:]))
    index = 0
    for id in tqdm(IDs):
        trainLabel = open(annotation_path + "/" +"{}.txt".format(id), "a")
        imgDir = os.path.join(rootPath, id)
        imgPaths = os.listdir(imgDir)
        imgPaths.sort(key=lambda x:int(x[:-4].split("_")[0]))
        for imgName in imgPaths:
            imgPath = os.path.join(imgDir,imgName)
            info = infos[index].split(",")
            middle_path =  info[0].split('"')[1]
            # f_path =  "/home/zhex/Downloads/data/train/" + middle_path + ".jpg"
            f_path =  "/home/zhex/Downloads/data/test/" + middle_path + ".jpg"
            x1 = int(info[1])-5
            y1 = int(info[2])-5
            x2 = x1 + int(info[3])+5
            y2 = y1 + int(info[4])+5
            if x1< 0 or y1 < 0 or x2 < x1 or y2 < y1:
                continue
            index += 1
            ID = str(imgPath.split("/")[-2])
            writePath = str(imgPath.split("/")[-3]) + "/" + str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
            # print("f_path=",f_path)
            # print("imgPath=",imgPath)
            if f_path == imgPath:
                try:
                    imgBGR = cv2.imread(imgPath)
                    imgBGR = imgBGR[y1:y2,x1:x2,:]
                    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
                    yaw = angleNoDetection(imgRGB)
                    trainLabel.write(writePath + " " + yaw + "\n")
                except Exception as e:
                    print(e)
            else:
                pass

# if __name__ == '__main__':
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     rootPath = "/home/zhex/data/imgs_glintasia"
#     trainImage = open("originImage.txt","a")
#     trainLable = open("originLabel.txt","a")
#     num_act = 0
#     num_pre = 0
#     for root, dirs, names in tqdm(os.walk(rootPath)):
#         names.sort(key=lambda x:int(x[:-4]))
#         for imgName in names:
#             imgPath = os.path.join(root,imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             # print("ID=",ID)
#             trainImage.write(ID + "\n")
#             trainImage.flush()
#             num_act += 1
#             imgBGR = cv2.imread(imgPath)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             try:
#                 # coefficient = angleDetection(imgRGB)
#                 yaw = angleNoDetection(imgRGB)
#                 coefficient = yawCoefficient(abs(yaw))
#                 # print("coefficient=",coefficient)
#                 num_pre += 1
#                 trainLable.write(ID + " " + str(coefficient)+ "\n")
#                 trainLable.flush()
#             except Exception as e:
#                 print(e)
#     print("num_act=",num_act)
#     print("num_pre=",num_pre)


# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     # videoDir = "/home/zhex/Videos/profileFace/outdoor"
#     # for videoName in os.listdir(videoDir):
#     # videoPath = os.path.join(videoDir,videoName)
#     videoPath = "/home/zhex/Videos/profileFace/outdoor/709.mp4"
#     videoName = "709.mp4"
#     vid = cv2.VideoCapture(videoPath)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(videoName,fourcc,25,(2560,1440))
#     while True:
#         flag, frame = vid.read()
#         # print(frame.shape)
#         if flag:
#             imgBGR = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             newImg,box = faceDetectionCenterFace(imgRGB)
#             if newImg != "zero" and box != []:
#                 x1,y1 = box[0],box[1]
#                 # yaw = angleDetection(newImg)
#                 yaw = angleNoDetection(newImg)
#                 print("yaw=",yaw)
#                 cv2.putText(frame,str(yaw),(2*x1,2*y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
#                 out.write(frame)
#                 # cv2.imshow("frame",frame)
#                 # cv2.waitKey(1)
#             else:
#                 continue
#         else:
#             break


# if __name__ == '__main__':
#     cv2.ocl.setUseOpenCL(False)
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     # imageDirF = "/home/zhex/Videos/profileFace/monitor/xiangzhong/frontal"
#     # imageDirP = "/home/zhex/Videos/profileFace/monitor/xiangzhong/profile"
#     # imageDirF = "/home/zhex/Videos/profileFace/monitor/gangxin/frontal"
#     imageDirP = "/home/zhex/Videos/profileFace/monitor/gangxin/profile"
#     # imageDirG = "/home/zhex/Documents/project/profileFace/pic/AI/lingangxin"
#     # imageDirT = "/home/zhex/git_me/recognition/occlusion_face_recognition/testImage"
#     imageNames = os.listdir(imageDirP)
#     imageNames.sort(key = lambda x: int(x[:-4]))
#     yawSum = []
#     for imgName in imageNames:
#         imagePath = os.path.join(imageDirP,imgName)
#         # print("imagePath=",imagePath)
#         imgBGR = cv2.imread(imagePath)
#         imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#         yaw, pitch, roll = angleNoDetection(imgRGB)
#         # print("yaw=",yaw)
#         coefficient = yawCoefficient(abs(yaw))
#         yawSum.append(coefficient)
#         # print("{}_yaw=".format(imgName),coefficient)
#     yawNpy = np.array(yawSum)
#     np.save("yawMonitorProfilelingangxin.npy",yawNpy)
