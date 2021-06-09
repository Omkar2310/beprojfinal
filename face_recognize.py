# facerec.py
import os
import cv2
import torch
import time
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'
learn = load_learner("dataset_with_mask")
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
(width, height) = (130, 100)
while True:
    (_, im) = webcam.read()
    cv2.imwrite("pic101.jpg", im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        img = open_image("pic101.jpg")
        # (image.jpg is any random image.)
        # img.show(figsize=(3, 3))
        pred_class, preds_idx, outputs = learn.predict(img)
        # Try to recognize the face
        print("Recognized as : ", pred_class)
        print(outputs[preds_idx])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(im, '%s - %.0f' % (pred_class, outputs[preds_idx]),
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.imwrite("pic101.jpg", im)
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
