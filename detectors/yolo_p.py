'''
Perform vehicle detection using models created with the YOLO (You Only Look Once) neural net.
https://pjreddie.com/darknet/yolo/
'''

# pylint: disable=no-member,invalid-name

import cv2
import numpy as np
import settings


with open(settings.YOLO_P_CLASSES_PATH, 'r') as classes_file:
    CLASSES_P = dict(enumerate([line.strip() for line in classes_file.readlines()]))
with open(settings.YOLO_P_CLASSES_OF_INTEREST_PATH, 'r') as coi_file_P:
    CLASSES_OF_INTEREST_P = tuple([line.strip() for line in coi_file_P.readlines()])
conf_threshold_P = settings.YOLO_P_CONFIDENCE_THRESHOLD
net = cv2.dnn.readNet(settings.YOLO_P_WEIGHTS_PATH, settings.YOLO_P_CONFIG_PATH)

def get_bounding_boxes(image):
    '''
    Return a list of bounding boxes of vehicles detected,
    their classes and the confidences of the detections made.
    '''

    # create image blob
    scale_P = 0.00392
    image_blob_P = cv2.dnn.blobFromImage(image, scale_P, (416, 416), (0, 0, 0), True, crop=False)

    # detect objects
    net.setInput(image_blob_P)
    layer_names_P = net.getLayerNames()
    output_layers_P = [layer_names_P[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs_P = net.forward(output_layers_P)

    _classes_P = []
    _confidences_P = []
    boxes_P = []
    nms_threshold_P = 0.4

    for output in outputs_P:
        for detection in output:
            scores_P = detection[5:]
            class_id_P = np.argmax(scores_P)
            confidence = scores_P[class_id_P]
            if confidence > conf_threshold_P and CLASSES_P[class_id_P] in CLASSES_OF_INTEREST_P:
                width = image.shape[1]
                height = image.shape[0]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                _classes_P.append(CLASSES_P[class_id_P])
                _confidences_P.append(float(confidence))
                boxes_P.append([x, y, w, h])

    # remove overlapping bounding boxes
    indices_P = cv2.dnn.NMSBoxes(boxes_P, _confidences_P, conf_threshold_P, nms_threshold_P)

    _bounding_boxes_P = []
    for i in indices_P:
        i = i[0]
        _bounding_boxes_P.append(boxes_P[i])

    return _bounding_boxes_P, _classes_P, _confidences_P