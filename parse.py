import face_recognition
import cv2
import os
import dlib
import imutils

# https://github.com/ahmetozlu/face_recognition_crop

# Open the input movie file
input_movie = cv2.VideoCapture("assets/short2.mp4")
current_path = os.getcwd()

counter = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    if not ret:
        break

    # Find all the faces and face encodings in the current frame of video
    frame = imutils.resize(frame, width=500)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(frame)
    
    # Label the results
    for (top, right, bottom, left) in face_locations:

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # crop image according to face
        crop_img = frame[top:bottom, left:right]
        cv2.imwrite(current_path + "/cropped/" + str(counter).zfill(3) +".png", crop_img)
        counter +=  1

    cv2.imshow('face_recog_crop', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# All done!
input_movie.release()
cv2.destroyAllWindows()