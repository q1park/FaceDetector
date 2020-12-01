import face_recognition
import cv2
import os
import imutils
import argparse
from typing import str


def main(fname: str):
    """extract frames that contain faces.
    """

    # https://github.com/ahmetozlu/face_recognition_crop

    input_movie = cv2.VideoCapture(fname)
    counter = 0

    while True:
        ret, frame = input_movie.read()

        # Quit when the input video file ends
        if not ret:
            break

        # Find all the faces in the current frame of video
        # frame = imutils.resize(frame, width=500)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)
        
        # Label the results
        for (top, right, bottom, left) in face_locations:

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # crop image according to face
            crop_img = frame[top:bottom, left:right]
            crop_img = cv2.resize(crop_img,(200, 200))  # sentinel is resizing to 200x200
            cv2.imwrite("cropped/" + str(counter).zfill(4) +".png", crop_img)
            counter +=  1

        cv2.imshow('cropped', frame)
        
        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    input_movie.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file")
    args = parser.parse_args()
    main(args.file)