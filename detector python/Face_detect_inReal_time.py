import cv2

trained_faced_data = cv2.CascadeClassifier('E:\\artificial intelligence\\xml files\\haarcascade_frontalface_default.xml')

webCam = cv2.VideoCapture(1)
while True:
    successful_frame_read , frame = webCam.read()
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_faced_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w , y+h),(0,255,0),2)

    cv2.imshow('face detecting...',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
       break

key = cv2.waitKey(1)


