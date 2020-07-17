import os
import cv2
from math import cos, sin, log
from fsaLib.FSANET_model import *
from faceDetection import faceDetectionCenterFace,faceDetectionCenterMutilFace
from keras.layers import Average

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
    return yaw


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

def createModel(weightPath1,weightPath2,weightPath3):
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
#     img = cv2.imread(imgPath)
#     # yaw = angleDetection(img)
#     yaw = angleNoDetection(img)
#     print("yaw=",yaw)


# if __name__ == '__main__':
#     weightPath1 = 'pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = 'pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = 'pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createModel(weightPath1,weightPath2,weightPath3)
#     rootPath = "/home/zhengxiangzhong/dataset_recognition/imgs_glintasia"
#     trainImage = open("originImage.txt","a")
#     trainLable = open("originLabel.txt","a")
#     num_act = 0
#     num_pre = 0
#     for root, dirs, names in os.walk(rootPath):
#         for imgName in names:
#             imgPath = os.path.join(root,imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             print("ID=",ID)
#             trainImage.write(ID + "\n")
#             trainImage.flush()
#             num_act += 1
#             img = cv2.imread(imgPath)
#             try:
#                 coefficient = angleDetection(img)
#                 print("coefficient=",coefficient)
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
#             img = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#             newImg,box = faceDetectionCenterFace(img)
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

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model = createModel(weightPath1,weightPath2,weightPath3)
    videoDir = "/home/zhex/Videos/profileFace/outdoor"
    # for videoName in os.listdir(videoDir):
    #     videoPath = os.path.join(videoDir,videoName)
    videoPath = "/home/zhex/Videos/profileFace/outdoor/709.mp4"
    videoName = "709_2.mp4"
    vid = cv2.VideoCapture(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(videoName,fourcc,25,(2560,1440))
    while True:
        flag, frame = vid.read()
        # print(frame.shape)
        if flag:
            img = cv2.resize(frame,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            imgDict = faceDetectionCenterMutilFace(img)
            if len(imgDict)>0:
               for key,value in imgDict.items():
                    yaw = angleNoDetection(value)
                    print("yaw=",yaw)
                    cv2.putText(frame,str(yaw),(2*key[0],2*key[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                    out.write(frame)
                    # cv2.imshow("frame",frame)
                    # cv2.waitKey(1)
            else:
                continue
        else:
            break