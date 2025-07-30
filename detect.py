import streamlit as st
import cv2
import numpy as np
from keras.preprocessing.image import load_img
from keras.models import load_model
import os
import pandas as pd

# Load the model
model = load_model("emotiondetector.keras")

# Labels for predictions
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to process image
def ef(image):
    img = load_img(image, color_mode="grayscale")  # Changed grayscale=True to color_mode="grayscale"
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to capture and process video input
def capture_video():
    # Load face detector from OpenCV
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    
    webcam = cv2.VideoCapture(0)
    
    while True:
        i, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        
        try: 
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = ef(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            cv2.imshow("Output", im)
            cv2.waitKey(27)
        except cv2.error:
            pass

# Streamlit UI
def main():
    st.title("Emotion Detection")
    st.write("This is a simple emotion detection model using deep learning.")
    
    img_path = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if img_path is not None:
        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        img = ef(img_path)  # Process the image
        pred = model.predict(img)
        pred_label = labels[pred.argmax()]
        st.write(f"Predicted emotion: {pred_label}")
    
    if st.button("Start Real-Time Video"):
        capture_video()

if __name__ == "__main__":
    main()
