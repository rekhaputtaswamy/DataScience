import cv2
import os, sys

#image and cascade names are passed as command-line arguments
image_path = sys.argv[1]
cascade_path = sys.argv[2]

#create haar cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

#Read the image
image = cv2.imread(image_path)

#converting image to grayscale since many operations in OpenCV are done in grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces in the image
faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
)

#faces has a list of rectangles in which faces are found. This list is looped over where it thinks it found something
# Draw a rectangle around the faces
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#This function returns 4 values: the x and y location of the rectangle, and the rectangle’s width and height (w , h).
cv2.imshow("Faces found", image)
cv2.waitKey(0)

# Command
#$ python face_detection.py groupies.png haarcascade_frontalface_default.xml

#Reference
#https://realpython.com/face-recognition-with-python/