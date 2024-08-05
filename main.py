import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('eemotion_detection_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess an image for the model
def preprocess_image(image):
    resized = cv2.resize(image, (48, 48))
    normalized = resized.astype('float32') / 255.0
    reshaped = np.expand_dims(normalized, axis=0)
    reshaped = np.expand_dims(reshaped, axis=-1)
    return reshaped

# Start video capture using the default method
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        preprocessed_image = preprocess_image(roi_gray)
        
        # Predict emotion
        prediction = model.predict(preprocessed_image)
        emotion = emotion_labels[np.argmax(prediction)]
        
        # Display the emotion
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
