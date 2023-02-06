import cv2
import mediapipe as mp 
import time 


cap =cv2.VideoCapture(0) 
pTime = 0
cTime = 0

mpFaceDetection = mp.solutions.face_detection # moduł wykrywania twarzy
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True: 
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    results = faceDetection.process(imgRGB)
    
    if results.detections: 
        for id, detection in enumerate(results.detections):
            print(id, detection)
            #mpDraw.draw_detection(img, detection)
            
            h, w, c =img.shape
            
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            
            cv2.rectangle(img,bbox, (255,0,0),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 2)
    cv2.imshow('Obraz', img)
    
    key = cv2.waitKey(1)
    if key==27:   #sprawia że esc zamyka okno, 27-klawisz esc
       break
   
cap.release()  
cv2.destroyAllWindows()