# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2 # open cv
import numpy as np # mathemartical calculation in the model
import mediapipe as mp # using the models 
import tensorflow as tf # machine learning
from tensorflow.keras.models import load_model 

import Tello # Tello.py 

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture') # getting the data from the model

# Load class names
f = open('gesture.names', 'r') # The gestures defined here are used here in the gesture.names file
classNames = f.read().split('\n') 
f.close()
print(classNames)

i = 0
tello = Tello.Controller()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # opening the webcam

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb) # The frame we process before is been used here.

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks]) # The data collected from model is been processed and give the final result.
            # print(prediction)
            classID = np.argmax(prediction)       # the index of the predicted guesture     
            className = classNames[classID]       # name of the predicted guesture.

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA) # name the prediction on the window

    # Show the final output
    cv2.imshow("Output", frame) 
    
    # couter
    if i > 20:
        
        if className == "":
            tello.send_command("command")
        else:
            tello.send_command(className)
        i = 0
        
    i += 1
    
    # exit the window.
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()