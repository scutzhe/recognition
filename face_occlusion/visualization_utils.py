from PIL import ImageDraw
import cv2
import numpy as np

def show_results(img, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline = 'red')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0, p[i + 5] - 1.0),(p[i] + 1.0, p[i + 5] + 1.0)], outline = 'green',width = 0)
    return img_copy

def rotation_img(img, bounding_boxes, facial_landmarks = []):
    '''
    :param img:  PIL's img
    :param bounding_boxes: face's coordinations
    :param facial_landmarks:  face's keywords coordinations
    :return: rotation's face (numpy's img)
    '''
    try:
        if bounding_boxes == [] or facial_landmarks == []:
            return []
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        boxes = np.empty([bounding_boxes.shape[0],4],dtype=float)
        landmarks = np.empty([facial_landmarks.shape[0],4],dtype=float)
        boxes = bounding_boxes[:,:4]
        landmarks[:,0] = facial_landmarks[:,0]
        landmarks[:,1] = facial_landmarks[:,5]
        landmarks[:,2] = facial_landmarks[:,1]
        landmarks[:,3] = facial_landmarks[:,6]

        img_affine = []
        for p, b in zip(landmarks,boxes):
            eye_center = ((p[0] + p[2]) // 2, (p[1] + p[3]) // 2)
            dy = p[3] - p[1]
            dx = p[2] - p[0]
            angle = cv2.fastAtan2(dy, dx)
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            img_rotation = cv2.warpAffine(img, M, dsize=(img.shape[1],img.shape[0]),borderValue=(255,255,255))

            x1 = int(b[0]) - 5
            y1 = int(b[1]) - 5
            x2 = int(b[2]) + 5
            y2 = int(b[3]) + 5
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            img_small = img_rotation[y1:y2,x1:x2]

            img_affine.append(img_small)
        return img_affine
    except Exception as e:
        print('unknown error {}'.format(e))