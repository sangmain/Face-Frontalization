import numpy as np
import cv2
import dlib
from preprocessing.inference import get_suffix, crop_img, parse_roi_box_from_landmark
import glob

STD_SIZE = 128


folder_path = 'D:/Selected_Img'

dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()

def crop_progress(image):

    rects = face_detector(image, 1)


    if len(rects) == 0:
        return

    for rect in rects:
        offset = 0
        top = rect.top()
        bottom = rect.bottom() - 0
        left = rect.left() + offset
        right = rect.right() - offset


        faceBoxRectangleS =  dlib.rectangle(left=left,top=top,right=right, bottom=bottom)

        pts = face_regressor(image, faceBoxRectangleS).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)


        cropped_image = crop_img(image, roi_box)

        cropped_image = cv2.resize(cropped_image, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        
        return cropped_image