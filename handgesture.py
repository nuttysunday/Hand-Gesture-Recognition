#IMPORTS

from cv2 import cv2 
import time 
import mediapipe as mp
import math
import vlc
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vRange = volume.GetVolumeRange()
volper = 0  
minV = vRange[0]
maxV = vRange[1] 
media = vlc.MediaPlayer("Nature.mp4")
media.play()

wCam,hCam = 640,480
vol = 0
volbar = 350

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
    pause_flag = True
    while(True):
        success, img = cap.read()
    
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img,(380,75),(630,350),(153, 255, 204),1)
        x_roi1,y_roi1 = 380,75
        x_roi2,y_roi2 = 630,350
        
        cv2.rectangle(img,(125,75),(225,350),(153, 255, 204),1)
        x_roi3,y_roi3 = 125,75
        x_roi4,y_roi4 = 225,350
        
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x11,y11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y* hCam) 
            x10,y10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y* hCam)

            if (x_roi1<x10) and (x11<x_roi2):
                cv2.putText(img,'Play/Pause controller',(355,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
                x3,y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y* hCam) 
                x4,y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y* hCam)

                cv2.line(img,(x3,y3),(x4,y4),(159, 226, 191),3)

                length1 = math.hypot(x3-x4,y3-y4)

                #pause 75-90
                #play > 150

                if length1>200 and pause_flag == False:
                    pause_flag = True
                    media.play()
                    cv2.putText(img,'Play ',(390,150),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
                elif length1<110 and pause_flag == True:
                    pause_flag = False
                    media.pause()
                    cv2.putText(img,'Pause',(390,150),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)

            elif (x_roi3<x10) and (x11<x_roi4):

                cv2.putText(img,'Volume controller',(125,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)
                x1,y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*wCam) ,int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y* hCam) 
                x2,y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*wCam),int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y* hCam)
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                cv2.circle(img,(x1,y1),15,(204, 204, 255),cv2.FILLED)
                cv2.circle(img,(x2,y2),15,(204, 204, 255),cv2.FILLED)
                cv2.circle(img,(cx,cy),15,(204, 204, 255),cv2.FILLED)

                cv2.line(img,(x1,y1),(x2,y2),(159, 226, 191),3)

                length = math.hypot(x2-x1,y2-y1)
                #print(length)

                if length<30:
                    cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

                #hand range = 30-200
                #vol range = -65 - 0

                vol = np.interp(length,[30,200],[minV,maxV])
                volbar = np.interp(length,[30,200],[350,100])
                volper = np.interp(length,[30,200],[0,100])
                #print(vol)
                volume.SetMasterVolumeLevel(vol,None)

            cv2.rectangle(img,(50,100),(85,350),(153, 204, 255),3)
            cv2.rectangle(img,(50,int(volbar)),(85,350),(102, 255, 255),cv2.FILLED)
            cv2.putText(img,f'{int(volper)}%',(40,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(26, 26, 26),2)

        cv2.imshow('HAND GESTURE RECOGNITION', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()