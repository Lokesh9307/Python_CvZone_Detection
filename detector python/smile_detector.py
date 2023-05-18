import cv2
face_detector = cv2.CascadeClassifier('E:\\artificial intelligence\\xml files\\haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier("E:\\artificial intelligence\\xml files\\smile_detetion.xml")
eye_detector = cv2.CascadeClassifier("E:\\artificial intelligence\\xml files\\eye.xml")
# img = cv2.imread('E:\\artificial intelligence\\images\\two.png')
webcam = cv2.VideoCapture(2)

while True:
    successful_frame_read,frame = webcam.read()

    if not successful_frame_read:
        break

    gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray_scale,scaleFactor=1.3,minNeighbors=5)
    smile = smile_detector.detectMultiScale(gray_scale,1.7,20)
    eyes = eye_detector.detectMultiScale(gray_scale,1.5,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

        for (x,y,w,h) in smile:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)

        if len(smile)>0:
            cv2.putText(frame,'Smiling...',(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,0,0))

        for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
          

    cv2.imshow('smile detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 

webcam.release()
cv2.destroyAllWindows()