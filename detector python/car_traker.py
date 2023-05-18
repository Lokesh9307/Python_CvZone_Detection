import cv2

img_file = 'images\car.png'
classifier_file = 'E:\\artificial intelligence\\xml files\\car_detector.xml'

img = cv2.imread(img_file)

grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier_file)
cars = car_tracker.detectMultiScale(grayscaled_img,1.1, 1)

for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w ,y+h),(0,0,255),2)
    
cv2.imshow("Car is tracking...",img)
cv2.waitKey()



print("code running...")


