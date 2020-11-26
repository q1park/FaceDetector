from scipy.spatial.distance import directed_hausdorff
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from queue import Queue
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac


def whiten(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    return points1, points2


def transformation_from_points(points1, points2):
    """
    https://scikit-image.org/docs/0.14.x/auto_examples/transform/plot_matching.html
    """

    points1, points2 = whiten(points1, points2)

    model = AffineTransform()
    model.estimate(points1, points2)
    model_robust, inliers = ransac((points1, points2), AffineTransform, min_samples=2, residual_threshold=20, max_trials=100)
    
    scale = np.round(model_robust.scale, 2)
    translation = np.round(model_robust.translation, 2)
    rotation = np.round(model_robust.rotation, 2)
    outliers = inliers == False
    
    return scale, translation, rotation, inliers, outliers



def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils.video import VideoStream
import time


cap = cv2.VideoCapture('short.mp4')

idx = 0

L = Queue(maxsize=30)

while(cap.isOpened()):
    idx += 1
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)  # upsample image by 1x


    # for (i, rect) in enumerate(rects):
    if len(rects) == 1:
        rect = rects[0]

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        shape = shape

        if not L.full(): 
            L.put(shape)
        else:
            prev = L.get()
            s, t, r, i, o = transformation_from_points(prev, shape)
            print(s, t, r)
            L.put(shape)
        

    print(idx, "-" * 10)

    # <---------------

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()