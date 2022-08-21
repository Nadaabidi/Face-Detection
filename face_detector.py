import cv2
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces in 
img = cv2.imread('D.jpeg')

#To capture video from webcam
#webcam = cv2.VideoCapture(0) # 0 means you're using the default webcam

#Convert image to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw Rectangles Around the Faces
#(x,y) are the top left corner pixels, (x+w, y+h) are the bottom right corner pixels, (0, 255, 0) BGR ref that gives us the color green, and 10 is the thickness of the rectangle

for (x,y,w,h) in face_coordinates:
#(x,y,w,h)=face_coordinates[0]
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256), 10))

#print(face_coordinates)

#Display the image with the faces
cv2.imshow("Nadouch's Face",img)
#wait the execution process
cv2.waitKey()


print("Hello Nadouch")