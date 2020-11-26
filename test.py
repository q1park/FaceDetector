import face_recognition
import cv2
import os

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

# The output video
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
output_movie = cv2.VideoWriter('tbbt_output.mp4', fourcc, 30, (1280, 720))

# Open the input movie file
input_movie = cv2.VideoCapture("assets/short1.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

current_path = os.getcwd()

counter = 0
counter1 = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)

    face_names = []
    
    # Label the results
    for (top, right, bottom, left) in face_locations:

        area = (right - left) * (bottom - top)

        if area < 100:
            pass
        else:

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            crop_img = frame[top:bottom, left:right]
            
            cv2.imwrite(current_path + "/cropped/" + "sheldon"+str(counter)+".png",crop_img)
            counter +=  1
        
    # Write the resulting image to the output video file
    output_movie.write(frame)
    print("Writing frame {} / {}".format(frame_number, length))
    
    cv2.imshow('face_recog_crop', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# All done!
input_movie.release()
cv2.destroyAllWindows()