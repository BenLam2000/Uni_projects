import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops
import time

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.class_colour = {
            'background': (0, 0, 0),
            'redapple': (0, 165, 255),
            'greenapple': (0, 255, 255),
            'orange': (0, 255, 0),
            'mango': (0, 0, 255),
            'capsicum': (255, 0, 0)
        }

    def detect_single_image(self, img):
        """
        function:
            detect target(s) in an image
        input:
            img: image, e.g., image read by the cv2.imread() function
        output:
            bboxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info from [x, y, w, h] back to the format of [x1,y1,x2,y2] topleft and bottomright corner
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[bbox[0]], thickness=2)

            # draw class label
            img_out = cv2.putText(img_out, f'{bbox[0]} {bbox[2]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                  self.class_colour[bbox[0]], 2)

        return bboxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        function:
            get bounding box and class label of target(s) in an image as detected by YOLOv8
        input:
            cv_img    : image, e.g., image read by the cv2.imread() function
            model_path: str, e.g., 'yolov8n.pt', trained YOLOv8 model
        output:
            bounding_boxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
        """

        # predict target type and bounding box with your trained YOLO

        tick = time.time()
        # predictions = self.model.predict(cv_img, imgsz=320, verbose=False, conf=0.6)
        predictions = self.model.predict(cv_img, imgsz=320, verbose=False, conf=0.6, classes=[1,2,3,4,5])
        # print(predictions)
        dt = time.time() - tick
        # print(f'Inference Time {dt:.2f}s, approx {1 / dt:.2f}fps', end="\r")
        # inf_time = predictions.speed['inference']
        # print(f'Inference Time {inf_time/1000:.2f}s, approx {1 / (inf_time/1000):.2f}fps')

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            # print(boxes)
            for box in boxes:
                # print(box)
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0]
                box_x = box_cord[0]
                box_y = box_cord[1]
                box_w = box_cord[2]
                box_h = box_cord[3]

                box_label = box.cls  # class label of the box

                bounding_boxes.append([prediction.names[int(box_label)], np.asarray([box_x,box_y,box_w,box_h]), round(float(box.conf),2)])
                # ex: bounding_boxes = ['redapple', array([200, 200, 100, 100]), 0.91]
        return bounding_boxes


# FOR TESTING ONLY
if __name__ == '__main__':
    # Test whole set of images
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    test_set_num = 1
    # for model_num in range(1,6):  # loop through all models (1-4)
    model_num = 5
    yolo = Detector(f'{script_dir}/model/best{model_num}.pt')

    for i in range(0,20):  # loop through all images (0-19)
        img = cv2.imread(f'{script_dir}/test_images/test{test_set_num}/img_{i}.png')

        bboxes, img_out = yolo.detect_single_image(img)

        # print(bboxes)
        # print(len(bboxes))

        # cv2.imshow('yolo detect', img_out)
        # cv2.waitKey(0)

        cv2.imwrite(f'{script_dir}/test_images/test{test_set_num}/img_{i}_model{model_num}_pred_small.png', img_out)


    # # Test single image
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # yolo = Detector(f'{script_dir}/model/model.best.pt')
    # img = cv2.imread(f'{script_dir}/test_images/test1/img_15.png')
    # bboxes, img_out = yolo.detect_single_image(img)
    #
    # cv2.imshow('yolo detect', img_out)
    # cv2.waitKey(0)
