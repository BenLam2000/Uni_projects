import numpy as np
import cv2
import sys
import os.path
import math

import plotly.express as px
import matplotlib.pyplot as plt

class FeatureMatching:
    def __init__(self):
        self.image_folder = 'images'

    # @@@@@@@@@@@@@@@@@@@@ HELPER FUNCTIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ #
    # read an image with colors in RGB order for matplotlib
    def iread(self, filename):
        """
        This function reads an image. Only images in the "images" folder are considered

        :param image: str with name of image to be read.
        :return: a numpy array of size [image_height, image_width] where each [i,j] corresponds to a pixel in the image.
        """
        return cv2.cvtColor(cv2.imread(os.path.join(self.image_folder, filename)), cv2.COLOR_BGR2GRAY)

    # read an image with colors in RGB order for matplotlib
    def iread_color(self, filename):
        """
        This function reads an image. Only images in the "images" folder are considered

        :param image: str with name of image to be read.
        :return: a numpy array of size [image_height, image_width] where each [i,j] corresponds to a pixel in the image.
        """
        return cv2.cvtColor(cv2.imread(os.path.join(self.image_folder, filename)), cv2.COLOR_BGR2RGB)

    def idisp(self, image, cmap = 'gray', height = None):
        """
        Displaying interactive image.
        """
        aspect_ratio = image.shape[1]/image.shape[0]
        width = int(aspect_ratio*height)
        fig = cv2.resize(image, (width, height))
        fig = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)
        cv2.imshow('Feature Matched Image', fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def match_features(self, img1_str, img2_str):
        # Get input image
        img1_color = self.iread_color(img1_str)
        img1_gray = self.iread(img1_str)

        # Get Test Image
        img2_color = self.iread_color(img2_str)
        img2_gray = self.iread(img2_str)

        # Create SIFT Descriptor
        sift = cv2.SIFT_create()

        # Detect SIFT Keypoints
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        # FLANN stands for Fast Library for Approximate Nearest Neighbors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        ratio_test_threshold = 0.9
        good = []
        for m, n in matches:
            if m.distance < ratio_test_threshold * n.distance:
                good.append(m)

        # Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. Otherwise simply show a message saying not enough matches are present.
        MIN_MATCH_COUNT = 5

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            print(src_pts)
            print(dst_pts)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img1_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2_gray = cv2.polylines(img2_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None

        # draw image
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None, **draw_params)
        self.idisp(img3, height=400)


if __name__ == "__main__":
    feature_matching = FeatureMatching()
    feature_matching.match_features('3.png','4.png')


