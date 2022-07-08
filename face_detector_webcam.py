import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haarcascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# webcam
webcam = cv2.VideoCapture(0)

while True:
    # read the current_frame
    successful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if successful_frame_read:
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

        for x, y, w, h in face_coordinates:
            cv2.rectangle(frame, (x, y),
                          (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

        cv2.imshow('Clever Programmer Face Detector', frame)

        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break

webcam.release()

print("Code Completed!!")
