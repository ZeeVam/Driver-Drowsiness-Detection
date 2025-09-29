import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import pygame

# --- Initialization ---

# Initialize Pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav.mp3')

# Load the trained model and dlib's models
print("Loading models...")
model = load_model('drowsiness_model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# --- Constants ---
EYE_CLOSED_COUNTER_THRESHOLD = 10
COUNTER = 0

# --- Start Video Stream ---
print("Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray, 0)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye coordinates
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Crop eye regions and preprocess
        try:
            leftEyeROI = frame[leftEye[1][1]:leftEye[4][1], leftEye[0][0]:leftEye[3][0]]
            rightEyeROI = frame[rightEye[1][1]:rightEye[4][1], rightEye[0][0]:rightEye[3][0]]

            leftEyeGray = cv2.cvtColor(leftEyeROI, cv2.COLOR_BGR2GRAY)
            leftEyeResized = cv2.resize(leftEyeGray, (24, 24))
            leftEyeNormalized = leftEyeResized / 255.0
            leftEyeFinal = np.expand_dims(leftEyeNormalized, axis=-1)
            leftEyeFinal = np.expand_dims(leftEyeFinal, axis=0)
            
            rightEyeGray = cv2.cvtColor(rightEyeROI, cv2.COLOR_BGR2GRAY)
            rightEyeResized = cv2.resize(rightEyeGray, (24, 24))
            rightEyeNormalized = rightEyeResized / 255.0
            rightEyeFinal = np.expand_dims(rightEyeNormalized, axis=-1)
            rightEyeFinal = np.expand_dims(rightEyeFinal, axis=0)
            
            prediction_left = model.predict(leftEyeFinal)
            prediction_right = model.predict(rightEyeFinal)

            cv2.putText(frame, f"Left Score: {prediction_left[0][0]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right Score: {prediction_right[0][0]:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            continue

        # Drowsiness logic
        if prediction_left < 0.5 and prediction_right < 0.5:
            COUNTER += 1
            
            if COUNTER >= EYE_CLOSED_COUNTER_THRESHOLD:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            pygame.mixer.music.stop()
        
        # Visualization
        left_eye_hull = cv2.convexHull(leftEye)
        right_eye_hull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    cv2.imshow("Drowsiness Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()