import cv2

video = cv2.VideoCapture('images\\track.mp4')

car_tracker = cv2.CascadeClassifier('E:\\artificial intelligence\\xml files\\car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('E:\\artificial intelligence\\xml files\\full_body.xml')

while True:
    (read_successful,frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(frame)
    pedestrians = pedestrian_tracker.detectMultiScale(frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),2)

    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,255),2)

    cv2.imshow('Self Driving Car',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
# key = cv2.waitKey(1)
video.release()