import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion model
emotion_model_path = 'models/emotion_model.h5'
emotion_model = load_model(emotion_model_path)

# Load age and gender models
age_prototxt_path = 'models/age_deploy.prototxt'
age_model_path = 'models/age_net.caffemodel'
gender_prototxt_path = 'models/deploy.prototxt'
gender_model_path = 'models/gender_net.caffemodel'

age_net = cv2.dnn.readNetFromCaffe(age_prototxt_path, age_model_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt_path, gender_model_path)

# Labels for the models
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
gender_labels = ['Male', 'Female']
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_rgb = frame[y:y+h, x:x+w]

        # Prepare face for emotion prediction
        emotion_face = cv2.resize(face, (48, 48))
        emotion_face = emotion_face.astype('float32') / 255
        emotion_face = np.expand_dims(emotion_face, axis=0)
        emotion_face = np.expand_dims(emotion_face, axis=-1)

        # Directly use the reshaped tensor without flattening
        emotion_prediction = emotion_model.predict(emotion_face)
        max_index = np.argmax(emotion_prediction[0])
        emotion = emotion_labels[max_index]

        # Prepare face for age and gender prediction
        face_blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = gender_labels[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = age_labels[age_preds[0].argmax()]

        # Display the results
        label = f"{emotion}, {gender}, {age}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Face Emotion, Age, and Gender Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
