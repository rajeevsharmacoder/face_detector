import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haarcascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# choosing an image to detect faces in
img = cv2.imread('facesinput1.jpeg')

# must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)

# draw rectangles around the faces
# first face only
# cv2.rectangle(img, (face_coordinates[0][0], face_coordinates[0][1]), (face_coordinates[0][0] +
#               face_coordinates[0][2], face_coordinates[0][1] + face_coordinates[0][3]), (0, 255, 0), 2)
# all faces
for x, y, w, h in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (randrange(100, 256), randrange(100, 256), randrange(100, 256)), 2)

# to show the image
cv2.imshow('Clever programmer face detector', img)

# pause code execution
cv2.waitKey()

print("Code Completed!!")
