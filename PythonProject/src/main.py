import os
import time
from datetime import datetime
import keras
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt

img_size = (80, 80)
gender_class_names = ['Male', 'Female']

def setup_camera():
    for index in range(3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                return cap
            cap.release()
    raise IOError("No working webcam found")

def preprocess_image(image):
    img = cv2.resize(image, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img



def predict_age_gender(frame):
    processed_image = preprocess_image(frame)

    gender_pred = gender_model.predict(processed_image, verbose=0)
    age_pred = age_model.predict(processed_image, verbose=0)

    gender_idx = np.argmax(gender_pred[0])
    gender = gender_class_names[gender_idx]

    age_idx = np.argmax(age_pred[0])
    age = age_idx

    return age, gender


def main():
    try:
        cap = setup_camera()

        last_capture_time = time.time()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                try:
                    current_time = time.time()
                    if current_time - last_capture_time >= 5 :
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        age, gender = predict_age_gender(face_img)
                        last_capture_time = current_time


                    text = f"Age: {age}, Gender: {gender}"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                except Exception as e:
                    print(f"Prediction error: {str(e)}")

            cv2.imshow('Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                shutil.rmtree('captured_images')
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()



age_model = keras.models.load_model('agemodel.h5')
gender_model = keras.models.load_model('gendermodel.h5')
main()