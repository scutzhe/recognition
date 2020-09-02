import numpy as np
import torch
from torch.autograd import Variable
from get_nets import ONet
from box_utils import nms, calibrate_box, get_image_boxes, convert_to_square


class FaceDetectorLandmark(object):
    def __init__(self, use_cuda=True):
        # LOAD MODELS
        self.onet = ONet()
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.onet = self.onet.cuda()

        self.onet.eval()

    def detect_faces(self, image, min_face_size=20.0, thresholds=(0.6, 0.7, 0.8), nms_thresholds=(0.7, 0.7, 0.7)):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.
            use_cuda:

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size  # 12/20 = 0.6
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # it will be returned
        bounding_boxes =  np.array([[0,0,112,112,1]])
        # STAGE 3
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        # with torch.no_grad():
        #     img_boxes = Variable(torch.FloatTensor(img_boxes))
        img_boxes = torch.from_numpy(img_boxes)

        if self.use_cuda:
            img_boxes = img_boxes.cuda()
            output = self.onet(img_boxes)
            landmarks = output[0].data.cpu().numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.cpu().numpy()  # shape [n_boxes, 4]
            probs = output[2].data.cpu().numpy()  # shape [n_boxes, 2]

        else:
            output = self.onet(img_boxes)
            landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].data.numpy()  # shape [n_boxes, 4]
            probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks
