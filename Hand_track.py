import cv2
import mediapipe as mp
import time 
import numpy as np

cam = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # wybiera moduł rąk z mediapipe
hands = mpHands.Hands() # w domsle sledzi 2 ręce 
mpDraw = mp.solutions.drawing_utils # funkcja do rysowania rąk

pTime = 0
cTime = 0

while True: 
    success, img = cam.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #zamiania obraz na RGB bo tylko taki czyta mp
    result = hands.process(imgRGB) # obiekt zawierający wyniki
    
    #print(result.multi_hand_landmarks)
    
    #pętla rozdzielenia wyników z poszczególnych rąk
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks: 
            #pętla która przypisuje wartoci do poszczególnych punktów (0-20)
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id, ':', cx,cy)
                
                if id==0:
                    cv2.circle(img, (cx,cy),10, (255,255,0), cv2.FILLED)
                if id==8: 
                    cv2.circle(img, (cx,cy),10, (255,100,100), cv2.FILLED)
                
                
                    
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # rysuje punkty
            
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,255), 2)
    
    cv2.imshow('Obraz', img)
    
    key = cv2.waitKey(1)
    if key==27:   #sprawia że esc zamyka okno, 27-klawisz esc
       break
   
cam.release()  
cv2.destroyAllWindows()
