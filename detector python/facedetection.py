import cv2
from random import randrange

trained_faced_data = cv2.CascadeClassifier('E:\\artificial intelligence\\xml files\\haarcascade_frontalface_default.xml')

img = cv2.imread('images\\two.png')                                                 # read images

grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                               #convert img into black and white                                                   #conver images into black and white

#detect face
face_coordinates = trained_faced_data.detectMultiScale(grayscaled_img)
print(face_coordinates)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2) 
                                
   

cv2.imshow("clever face detector",img)                                   #show image in terminal
cv2.waitKey()

print("code completed")