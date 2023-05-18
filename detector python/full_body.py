import cv2
import mediapipe as mp
webCam = cv2.VideoCapture(1)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


with mp_holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    while webCam.isOpened():
       ret ,frame = webCam.read()
# recolor feed
       image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
# make detection
       results = holistic.process(image)

    #    print(results.face_landmarks)

# face_landmarks, pose_landmarks,left_hand_landmarks,right_hand_landmarks....!

# recolor image to RGB to BGR for rendering
       image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
# draw face landmarks
       mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80,56,256), thickness=1, circle_radius=1)
                                )
                                                            
# right_hand_landmarks
       mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(80,56,256), thickness=2, circle_radius=1))
# left_hand_landmarks
       mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(80,56,256), thickness=2, circle_radius=1))
# pose_landmarks,
       mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(127,0,63), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(170,212,255), thickness=2, circle_radius=1))

       cv2.imshow("HOLISTIC MODEL DETECTION" ,image)

       if cv2.waitKey(10) & 0xFF ==ord('q'):
          break

webCam.release()
cv2.destroyAllWindows()

    
