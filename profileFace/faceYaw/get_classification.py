import os
import cv2
import shutil
import random
from math import tan, cos, sin, asin, log, sqrt, pow
from fsaLib.FSANET_model import *
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
    return round(yaw,2), round(pitch,2), round(roll,2)


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
    pitch = round(pitch,2)
    roll = round(roll,2)
    return yaw, pitch, roll

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


def profile_yaw(origin_frontal_dir,origin_profile_dir,new_frontal_dir,new_profile_dir):
    """
    :param origin_frontal_dir:
    :param origin_profile_dir:
    :param new_frontal_dir:
    :param new_profile_dir:
    :return:
    """
    assert os.path.exists(origin_frontal_dir),"{} is null !!!".format(origin_frontal_dir)
    assert os.path.exists(origin_profile_dir),"{} is null !!!".format(origin_profile_dir)
    if not os.path.exists(new_frontal_dir):
        os.makedirs(new_frontal_dir)
    if not os.path.exists(new_profile_dir):
        os.makedirs(new_profile_dir)

    origin_frontal_names = os.listdir(origin_frontal_dir)
    origin_profile_names = os.listdir(origin_profile_dir)
    index = 0
    for name in origin_frontal_names:
        tmp = name.split(".")[0]
        yaw = int(str(tmp).split("_")[1])
        print("yaw=",yaw)
        image_path = os.path.join(origin_frontal_dir,name)
        if yaw >= 30:##定义新的侧脸角度大于30°即为侧脸
            shutil.move(image_path,new_profile_dir)
            index += 1
        else:
            pass
    print("index=",index)


def equal_dataset(train_frontal_dir,train_profile_dir,middle_dir):
    """
    :param train_frontal_dir:
    :param train_profile_dir:
    :param middle_dir:
    :return:
    """
    assert os.path.exists(train_frontal_dir), "{} is null".format(train_frontal_dir)
    assert os.path.exists(train_profile_dir), "{} is null".format(train_profile_dir)
    if not os.path.exists(middle_dir):
        os.makedirs(middle_dir)
    frontal_names = os.listdir(train_frontal_dir)
    profile_names = os.listdir(train_profile_dir)
    frontal_length = len(frontal_names)
    profile_length = len(profile_names)
    length = profile_length if profile_length < frontal_length else frontal_length
    if length == profile_length:
        middle_names = random.sample(frontal_names,length)
        for name in tqdm(middle_names):
            image_path = os.path.join(train_frontal_dir,name)
            shutil.copy(image_path,middle_dir)
    else:
        middle_names = random.sample(profile_names,length)
        for name in tqdm(middle_names):
            image_path = os.path.join(train_profile_dir,name)
            shutil.copy(image_path,middle_dir)



def split_dataset(train_frontal_dir,train_profile_dir,test_frontal_dir,test_profile_dir):
    """
    :param train_frontal_dir:
    :param train_profile_dir:
    :param test_frontal_dir:
    :param test_profile_dir:
    :return:
    """
    assert os.path.exists(train_frontal_dir),"{} is null".format(train_frontal_dir)
    assert os.path.exists(train_profile_dir),"{} is null".format(train_profile_dir)

    if not os.path.exists(test_frontal_dir):
        os.makedirs(test_frontal_dir)
    if not os.path.exists(test_profile_dir):
        os.makedirs(test_profile_dir)

    frontal_names = os.listdir(train_frontal_dir)
    profile_names = os.listdir(train_profile_dir)
    train_length = len(frontal_names)
    test_length = int(0.3 * train_length)
    print("test_length=",test_length)

    indexF = 0
    for name in tqdm(frontal_names):
        image_path = os.path.join(train_frontal_dir,name)
        shutil.move(image_path,test_frontal_dir)
        indexF += 1
        if indexF == test_length:
            break

    indexP = 0
    for name in tqdm(profile_names):
        image_path = os.path.join(train_profile_dir, name)
        shutil.move(image_path, test_profile_dir)
        indexP += 1
        if indexP == test_length:
            break

    print("indexF,indexP = ",indexF, indexP)

if __name__ == '__main__':
    weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model = createCapsuleModel(weightPath1,weightPath2,weightPath3)
    rootPath = "/home/zhex/data/profiledatasetNoVal/profileAsia/train"
    save_dir = "/home/zhex/Downloads/face_classification_full/images"
    index = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for root, dirs, names in tqdm(os.walk(rootPath)):
        names.sort(key=lambda x:int(x[:-4]))
        for imgName in names:
            imgPath = os.path.join(root,imgName)
            # print("imgPath=",imgPath)
            ID = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
            # print("ID=",ID)
            imgBGR = cv2.imread(imgPath)
            imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
            try:
                yaw,_,_ = angleNoDetection(imgRGB)
                yaw = abs(yaw)
                yaw = int(yaw)
                index += 1
                shutil.copy(imgPath,save_dir)
                old_path = os.path.join(save_dir,imgName)
                new_path = os.path.join(save_dir,"{}_{}.jpg".format(index,yaw))
                os.rename(old_path,new_path)
            except Exception as e:
                print(e)
## step 1
# if __name__ == '__main__':
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createCapsuleModel(weightPath1,weightPath2,weightPath3)
#     rootPath = "/home/zhex/data/profiledatasetNoVal/profileAsia/train"
#     frontal_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     profile_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#     middle_dir = "/home/zhex/Downloads/face_classification_full/train/middle"
#     indexF = 0
#     indexP = 0
#     indexM = 0
#     if not os.path.exists(frontal_dir):
#         os.makedirs(frontal_dir)
#     if not os.path.exists(profile_dir):
#         os.makedirs(profile_dir)
#     if not os.path.exists(middle_dir):
#         os.makedirs(middle_dir)
#     for root, dirs, names in tqdm(os.walk(rootPath)):
#         names.sort(key=lambda x:int(x[:-4]))
#         for imgName in names:
#             imgPath = os.path.join(root,imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             # print("ID=",ID)
#             imgBGR = cv2.imread(imgPath)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             try:
#                 yaw,_,_ = angleNoDetection(imgRGB)
#                 yaw = abs(yaw)
#                 yaw = int(yaw)
#                 if yaw <= 20:
#                     indexF += 1
#                     shutil.copy(imgPath,frontal_dir)
#                     old_path = os.path.join(frontal_dir,imgName)
#                     new_path = os.path.join(frontal_dir,"{}_{}.jpg".format(indexF,yaw))
#                     os.rename(old_path,new_path)
#                 elif yaw >= 45:
#                     indexP += 1
#                     shutil.copy(imgPath,profile_dir)
#                     old_path = os.path.join(profile_dir, imgName)
#                     new_path = os.path.join(profile_dir, "{}_{}.jpg".format(indexP,yaw))
#                     os.rename(old_path, new_path)
#                 else:
#                     indexM += 1
#                     shutil.copy(imgPath, middle_dir)
#                     old_path = os.path.join(middle_dir, imgName)
#                     new_path = os.path.join(middle_dir, "{}_{}.jpg".format(indexM, yaw))
#                     os.rename(old_path, new_path)
#             except Exception as e:
#                 print(e)

# if __name__ == '__main__':
#     weightPath1 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
#     weightPath2 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
#     weightPath3 = '/home/zhex/pre_models/faceYaw/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
#     model = createCapsuleModel(weightPath1,weightPath2,weightPath3)
#     rootPath = "/home/zhex/data/profiledatasetNoVal/profileAsia/train"
#     middle_dir = "/home/zhex/Downloads/middle"
#     indexM = 0
#     if not os.path.exists(middle_dir):
#         os.makedirs(middle_dir)
#     for root, dirs, names in tqdm(os.walk(rootPath)):
#         names.sort(key=lambda x:int(x[:-4]))
#         for imgName in names:
#             imgPath = os.path.join(root,imgName)
#             # print("imgPath=",imgPath)
#             ID = str(imgPath.split("/")[-2]) + "/" + str(imgPath.split("/")[-1])
#             # print("ID=",ID)
#             imgBGR = cv2.imread(imgPath)
#             imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)
#             try:
#                 yaw,_,_ = angleNoDetection(imgRGB)
#                 yaw = abs(yaw)
#                 yaw = int(yaw)
#                 if yaw > 20 and yaw < 45:
#                     indexM += 1
#                     shutil.copy(imgPath, middle_dir)
#                     old_path = os.path.join(middle_dir, imgName)
#                     new_path = os.path.join(middle_dir, "{}_{}.jpg".format(indexM, yaw))
#                     os.rename(old_path, new_path)
#             except Exception as e:
#                 print(e)

## step 2
# if __name__ == "__main__":
#     frontal_dir = "/home/zhex/Downloads/face_classification_full/train/frontal_origin"
#     profile_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#     middle_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     equal_dataset(frontal_dir,profile_dir,middle_dir)


## step 3
# if __name__ == "__main__":
#     train_frontal_dir = "/home/zhex/Downloads/face_classification_full/train/frontal"
#     train_profile_dir = "/home/zhex/Downloads/face_classification_full/train/profile"
#
#     test_frontal_dir = "/home/zhex/Downloads/face_classification_full/test/frontal"
#     test_profile_dir = "/home/zhex/Downloads/face_classification_full/test/profile"
#     split_dataset(train_frontal_dir, train_profile_dir, test_frontal_dir, test_profile_dir)
